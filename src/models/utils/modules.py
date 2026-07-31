import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_queries_or_keys(x, pos, base=10000):
    B, num_heads, N, D = x.size()
    assert D%2 == 0

    #compute frequency and angels
    omega = torch.arange(D//2, dtype=x.dtype, device=x.device)
    omega /= D/2.0
    omega = 1.0/ base**omega
    freq = torch.einsum("..., f -> ... f", pos, omega)

    # rotation matrix to multiply with
    emb_sin = freq.sin()
    emb_cos = freq.cos()

    emb_sin = emb_sin.repeat_interleave(2, dim=-1)
    emb_cos = emb_cos.repeat_interleave(2,dim=-1)

    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)

    return (x * emb_cos) + (y * emb_sin)


class RoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        grid_size=14,
        is_causal=False,
        base = 10000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_sdpa = use_sdpa
        self.grid_size = grid_size
        self.is_causal = is_causal
        self.base = base

        #spliiting the, dim for 3d rope
        self.d_dim = int(2*((head_dim//3))//2)
        self.h_dim = int(2*((head_dim//3))//2)
        self.w_dim = int(2*((head_dim//3))//2)

        
        

    # get (t,h,w)
    def _get_frame_pos(self,ids, H_patches=None, W_patches=None  ):
        #nbr of patches per frame:
        if H_patches is  None or W_patches is None :
            tokens_per_frame = int(self.grid_size*self.grid_size)
        else:
            tokens_per_frame = int(H_patches*W_patches)
        return ids // tokens_per_frame
    def _get_height_pos(self, ids, H_patches=None, W_patches=None ):
        #remove frame component from ids
        if H_patches is  None or W_patches is None :
            tokens_per_frame = int(self.grid_size*self.grid_size)
            tokelns_per_row = self.grid_size #nbr of patchs(img_size//patch_size)
        else:
            tokens_per_frame = int(H_patches*W_patches)
            tokens_per_row = W_patches
        frame_ids = ids//tokens_per_frame
        ids = ids - frame_ids*tokens_per_frame

        return ids//tokens_per_row
    def seperate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is  None or W_patches is None :
            tokens_per_frame = int(self.grid_size*self.grid_size)
            tokens_per_row = self.grid_size #nbr of patchs(img_size//patch_size)
        else:
            tokens_per_frame = int(H_patches*W_patches)
            tokens_per_row = W_patches

        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)


        #-- get width ids
        width_ids = ids - (frame_ids*tokens_per_frame) - (height_ids*tokens_per_row)

        return frame_ids, height_ids, width_ids


    def _unmasked_positions(self, T, H_patches, W_patches, device):
        """Directly build (t, h, w) grids for the no-mask case -- no encode/decode roundtrip."""
        shape_key = (T, H_patches, W_patches)
        if self._cached_shape_key != shape_key:
            t_ids = torch.arange(T, device=device).repeat_interleave(H_patches * W_patches)
            h_ids = torch.arange(H_patches, device=device).repeat_interleave(W_patches).repeat(T)
            w_ids = torch.arange(W_patches, device=device).repeat(H_patches * T)
            self._cached_positions = (t_ids.float(), h_ids.float(), w_ids.float())
            self._cached_shape_key = shape_key
        return self._cached_positions

    # ---------- forward ----------

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None):
        B, N, C = x.shape

        qkv = self.qkv(x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        # -- Stage 2: figure out each token's (t, h, w)
        if mask is not None:
            # tokens were removed/reordered by JEPA masking -- must use real original indices
            mask = mask.unsqueeze(1)  # [B, 1, N_keep] -- broadcasts over heads, no need to .repeat
            d_pos, h_pos, w_pos = self.separate_positions(mask, H_patches, W_patches)
        else:
            # nothing removed -- positions can be built directly and cached
            d_pos, h_pos, w_pos = self._unmasked_positions(T, H_patches, W_patches, x.device)

        # -- Stage 3: rotate each axis-chunk of Q and K independently
        s = 0
        qd = rotate_queries_or_keys(q[..., s:s + self.d_dim], pos=d_pos, base=self.base)
        kd = rotate_queries_or_keys(k[..., s:s + self.d_dim], pos=d_pos, base=self.base)
        s += self.d_dim

        qh = rotate_queries_or_keys(q[..., s:s + self.h_dim], pos=h_pos, base=self.base)
        kh = rotate_queries_or_keys(k[..., s:s + self.h_dim], pos=h_pos, base=self.base)
        s += self.h_dim

        qw = rotate_queries_or_keys(q[..., s:s + self.w_dim], pos=w_pos, base=self.base)
        kw = rotate_queries_or_keys(k[..., s:s + self.w_dim], pos=w_pos, base=self.base)
        s += self.w_dim

        # -- Stage 4: recombine rotated chunks (+ untouched remainder, if head_dim doesn't split evenly)
        if s < self.head_dim:
            qr = q[..., s:]
            kr = k[..., s:]
            q = torch.cat([qd, qh, qw, qr], dim=-1)
            k = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q = torch.cat([qd, qh, qw], dim=-1)
            k = torch.cat([kd, kh, kw], dim=-1)

        # -- Stage 5: standard attention, unchanged from plain Attention
        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ACRoPEAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        base=10000,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_sdpa = use_sdpa
        self.is_causal = is_causal
        self.grid_size = grid_size
        self.base = base

        self.d_dim = int(2 * ((head_dim // 3) // 2))
        self.h_dim = int(2 * ((head_dim // 3) // 2))
        self.w_dim = int(2 * ((head_dim // 3) // 2))

        self._cached_shape_key = None
        self._cached_positions = None

    # ---------- position decomposition: identical to RoPEAttention ----------

    def _get_frame_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
        else:
            tokens_per_frame = int(H_patches * W_patches)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        ids = ids - tokens_per_frame * frame_ids
        return ids // tokens_per_row

    def separate_positions(self, ids, H_patches=None, W_patches=None):
        if H_patches is None or W_patches is None:
            tokens_per_frame = int(self.grid_size * self.grid_size)
            tokens_per_row = self.grid_size
        else:
            tokens_per_frame = int(H_patches * W_patches)
            tokens_per_row = W_patches
        frame_ids = self._get_frame_pos(ids, H_patches, W_patches)
        height_ids = self._get_height_pos(ids, H_patches, W_patches)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids.float(), height_ids.float(), width_ids.float()

    def _patch_positions(self, T, H_patches, W_patches, device):
        if H_patches is None or W_patches is None:
            grid_H = grid_W = self.grid_size
        else:
            grid_H, grid_W = H_patches, W_patches

        shape_key = (T, grid_H, grid_W)
        if self._cached_shape_key != shape_key:
            ids = torch.arange(int(T * grid_H * grid_W), device=device)
            self._cached_positions = self.separate_positions(ids, grid_H, grid_W)
            self._cached_shape_key = shape_key
        return self._cached_positions

    # ---------- forward ----------

    def forward(self, x, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, action_tokens=0):
        B, N, C = x.shape

        # -- Stage 2: split action tokens from patch tokens (per frame), BEFORE computing qkv,
        #    so each group gets its own qkv projection call.
        if action_tokens > 0:
            x = x.view(B, T, action_tokens + H_patches * W_patches, C)  # [B, T, K+H*W, C]
            action_x = x[:, :, :action_tokens, :].flatten(1, 2)          # [B, T*K, C]
            patch_x = x[:, :, action_tokens:, :].flatten(1, 2)           # [B, T*H*W, C]
        else:
            patch_x = x
            action_x = None

        # -- Stage 3: rotate action tokens (temporal position only, full head_dim)
        if action_x is not None:
            qkv_a = self.qkv(action_x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
            q_a, k_a, v_a = qkv_a[0], qkv_a[1], qkv_a[2]     # [B, num_heads, T*K, head_dim]

            frame_pos = torch.arange(T, device=x.device).repeat_interleave(action_tokens).float()
            q_a = rotate_queries_or_keys(q_a, pos=frame_pos, base=self.base)
            k_a = rotate_queries_or_keys(k_a, pos=frame_pos, base=self.base)

        # -- Stage 4: rotate patch tokens (d/h/w split, exactly like RoPEAttention)
        qkv_p = self.qkv(patch_x).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q_p, k_p, v_p = qkv_p[0], qkv_p[1], qkv_p[2]         # [B, num_heads, T*H*W, head_dim]

        if mask is not None:
            mask_pos = mask.unsqueeze(1)
            d_pos, h_pos, w_pos = self.separate_positions(mask_pos, H_patches, W_patches)
        else:
            d_pos, h_pos, w_pos = self._patch_positions(T, H_patches, W_patches, x.device)

        s = 0
        qd = rotate_queries_or_keys(q_p[..., s:s + self.d_dim], pos=d_pos, base=self.base)
        kd = rotate_queries_or_keys(k_p[..., s:s + self.d_dim], pos=d_pos, base=self.base)
        s += self.d_dim
        qh = rotate_queries_or_keys(q_p[..., s:s + self.h_dim], pos=h_pos, base=self.base)
        kh = rotate_queries_or_keys(k_p[..., s:s + self.h_dim], pos=h_pos, base=self.base)
        s += self.h_dim
        qw = rotate_queries_or_keys(q_p[..., s:s + self.w_dim], pos=w_pos, base=self.base)
        kw = rotate_queries_or_keys(k_p[..., s:s + self.w_dim], pos=w_pos, base=self.base)
        s += self.w_dim

        if s < self.head_dim:
            qr, kr = q_p[..., s:], k_p[..., s:]
            q_p = torch.cat([qd, qh, qw, qr], dim=-1)
            k_p = torch.cat([kd, kh, kw, kr], dim=-1)
        else:
            q_p = torch.cat([qd, qh, qw], dim=-1)
            k_p = torch.cat([kd, kh, kw], dim=-1)

        # -- Stage 5: merge action + patch tokens back, per frame, restoring original order
        if action_tokens > 0:
            def merge(t_action, t_patch):
                t_action = t_action.view(B, self.num_heads, T, action_tokens, -1)
                t_patch = t_patch.view(B, self.num_heads, T, H_patches * W_patches, -1)
                return torch.cat([t_action, t_patch], dim=3).flatten(2, 3)

            q = merge(q_a, q_p)
            k = merge(k_a, k_p)
            v = merge(v_a, v_p)
        else:
            q, k, v = q_p, k_p, v_p

        # -- Stage 6: standard attention, identical to RoPEAttention
        if attn_mask is not None or self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    




class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=None,
                 hidden_features=None,
                 act_layer=nn.GELU,
                 drop = 0.0
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

        
        
def build_action_block_causal_attention_mask(T,H,W,add_tokens=1):
    N_T = add_tokens + (H*W)
    N = T*N_T
    mask = torch.zeros(N,N).bool()
    mask_block = torch.ones(N_T,N_T).bool()
    local_window_time = T

    for t1 in range(T):
        for t2 in range(max(0, t1 - local_window_time + 1), t1 + 1):
            mask[t1 * N_T : (t1 + 1) * N_T, t2 * N_T : (t2 + 1) * N_T] = mask_block

    return mask


     

class Attention(nn.Module):
    def __init__(self,
                 dim=768,
                 num_heads=12,
                 qk_scale=None,
                 qkv_bias=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 use_sdpa=True,
                 is_causal=False):                          
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or dim**-0.5
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.proj = nn.Linear(dim,dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.is_causal = is_causal
        self.use_sdpa = use_sdpa
        self.proj_drop_prob = proj_drop

    def forward(self,x,mask=None,attn_mask=None):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob, is_causal=self.is_causal, attn_mask=attn_mask
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
           
        

class Block (nn.Module):
    def __init__(self,
                 dim=768,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qk_scale=None,
                 qkv_bias=True,
                 attn_drop=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_sdpa=True,
                 is_causal=False,
                 grid_size=16,
                 proj_drop=0.0,
                 use_rope=False,
                 **kwargs,
                 ):
        super().__init__()
        self.norm1 =norm_layer(dim)
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=proj_drop)
               
            
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,hidden_features=mlp_hidden_dim, act_layer=act_layer)
    def forward(self,x,mask=None,attn_mask=None,T=None, H_patches=None, W_patches=None):
        y = self.norm1(x)
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

class ACBlock(nn.Module):
    def __init__(self,
                     dim=768,
                     num_heads=12,
                     mlp_ratio=4.0,
                     qk_scale=None,
                     qkv_bias=False,
                     attn_drop=0.0,
                     act_layer=nn.GELU,
                     norm_layer=nn.LayerNorm,
                     use_sdpa=True,
                     is_causal=False,
                     grid_size=16,
                     proj_drop=0.0,
                     use_rope=False,
                     **kwargs,
                     ):
            super().__init__()
            self.norm1 =norm_layer(dim)
            if use_rope:
                self.attn = ACRoPEAttention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    grid_size=grid_size,
                    proj_drop=proj_drop)
                   
                
            else:
                self.attn = Attention(
                    dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal)
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = MLP(in_features=dim,hidden_features=mlp_hidden_dim, act_layer=act_layer)
    def forward(self,x,mask=None,attn_mask=None,T=None, H_patches=None, W_patches=None, action_tokens=0):
        y = self.norm1(x)
        if isinstance(self.attn, ACRoPEAttention):
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask, T=T, H_patches=H_patches, W_patches=W_patches,action_tokens=action_tokens)
        else:
            y = self.attn(self.norm1(x), mask=mask, attn_mask=attn_mask)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x








