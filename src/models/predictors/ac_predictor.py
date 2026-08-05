import math 


import torch
import torch.nn as nn

from functools import partial
from src.models.utils.modules import ACBlock
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.models.utils.tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    def __init__(self,
                 img_size=(224,224),
                 patch_size=16,
                 in_chans=3,
                 num_frames=1,
                 tubelet_size=2,
                 embed_dim=768,
                 pred_embed_dim=1024,
                 num_heads=12,
                 depth=24,
                 mlp_ratio=4.0,
                 qk_scale=None,
                 qkv_bias=True,
                 is_causal=True,
                 use_sdpa=False,
                 attn_drop=0.0,
                 norm_layer=nn.LayerNorm,
                 use_rope=True,
                 init_std=0.02,
                 use_activation_checkpointing=False,
                 action_dim=7,
                 state_dim=7,
                 extrinsics_dim=7,

                 use_extrinsics=False,
                 **kwargs
                 ):
        super().__init__()

        self.is_causal = is_causal
        self.use_extrinsics = use_extrinsics

        if type(img_size)==int:
            img_size = (img_size,img_size)
        self.img_height,self.img_width = img_size
        #video case
        self.is_video = num_frames>1
        self.use_activation_checkpointing = use_activation_checkpointing
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        #from embed_dim-->pred_embed_dim 
        #4proj(patches,state,action,extrinsics)
        
        self.predictor_embed = nn.Linear(embed_dim,pred_embed_dim,bias=True)
        self.action_embed = nn.Linear(action_dim,pred_embed_dim,bias=True)
        self.state_embed = nn.Linear(state_dim,pred_embed_dim,bias=True)
        if self.use_extrinsics:
            self.extrinsics_embed = nn.Linear(extrinsics_dim,pred_embed_dim,bias=True)

        #reshape the 4 proj into per-frame structure
        self.grid_height = img_size[0]//patch_size
        self.grid_width = img_size[1]//patch_size
        #Attention blocks
        self.use_rope = use_rope  
        self.predictor_blocks = nn.ModuleList(
            [
                ACBlock(
                    use_rope=use_rope,
                    dim=pred_embed_dim,
                    grid_size = self.grid_height,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    act_layer=nn.GELU,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    is_causal=False,
                    use_sdpa=use_sdpa
                ) for _ in range(depth)
            ]     
        )
        #normalize and back to targert size
        self.predictor_norm = norm_layer(pred_embed_dim)
        self.predictor_proj = nn.Linear(pred_embed_dim,embed_dim,bias=True)
        #weight initialization
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m,nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):  #to review other approach
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self,x,actions,states,extrinsics=None,T=None):
        # x: context token 
        B,N,C = x.shape
        H_patches, W_patches = self.grid_height, self.grid_width
        K = 3 if self.use_extrinsics else 2
        #modality proj and reshape per frame
        visual = self.predictor_embed(x)
        state_tok = self.state_embed(states).unsqueeze(2)
        action_tok = self.action_embed(actions).unsqueeze(2)
        if self.use_extrinsics:
            assert extrinsics is not None, "use_extrinsics=True but no extrinsics passed"
            extr_tok = self.extrinsics_embed(extrinsics).unsqueeze(2)

        #--reshape into per frame structure
        visual = visual.view(B, T, H_patches * W_patches, -1)
        #concatenation
        if self.use_extrinsics:
            frame_tokens = torch.cat([action_tok, state_tok, extr_tok, visual], dim=2)
        else:
            frame_tokens = torch.cat([action_tok, state_tok, visual], dim=2)
        # shape --> [B, T, K + H*W, pred_embed_dim]
        #flattening
        x = frame_tokens.flatten(1,2)
        # -- build causal mask:
        attn_mask = None
        if self.is_causal:
             attn_mask = build_action_block_causal_attention_mask(T, H_patches, W_patches, add_tokens=K)
             attn_mask = attn_mask.to(x.device)
        

        #fwd prop 
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    action_tokens=K,
                    use_reentrant=False
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    action_tokens=K,
                )
        # drop the conditionning tokens
        x = x.view(B, T, K + H_patches * W_patches, -1)
        x = x[:, :, K:, :]                  # discard action/state/(extrinsics) slots, keep patches only
        x = x.flatten(1, 2)                 # [B, T*H*W, pred_embed_dim]

        # --- final projection back to encoder's embed_dim ---
        x = self.predictor_proj(x)          # [B, T*H*W, embed_dim]

        return x


def vit_ac_predictor(**kwargs):
    model = VisionTransformerPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model