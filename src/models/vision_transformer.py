import torch
import math
import torch.nn as nn
from functools import partial
#importation

from src.models.utils.modules import Block
from src.masks.utils import apply_masks
from src.utils import trunc_normal_

class PatchEmbed(nn.Module):

    def __init__(self,
                 patch_size=16,
                 embed_dim=768,
                 in_chans=3):
        super().__init__()
        self.patch_size=patch_size
        self.proj= nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        x=self.proj(x).flatten(2).transpose(1,2)

        return x
class PatchEmbed3D(nn.Module):
    def __init__(self,
                 patch_size=16,
                 tubelet_size=2,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.patch_size=patch_size
        self.tubelet_size=tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size,patch_size,patch_size),
            stride=(tubelet_size,patch_size,patch_size)
        )
    def forward(self,x,**kwargs):
        x = self.proj(x).flatten(2).transpose(1,2)
        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 img_size=(224,224),
                 patch_size=16,
                 in_chans=3,
                 depth=12,
                 embed_dim=768,
                 num_frames=1,
                 tubelet_size=2,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qk_scale=None,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 use_activation_checkpointing=False,
                 out_layers=None,
                 use_rope=False,
                 attn_drop_rate=0.0,
                 init_std=0.02,
                 **kwargs
                 ):
        super().__init__()

        self.num_features=self.embed_dim=embed_dim
        
        self.num_heads=num_heads

        if type(img_size) is int:
            img_size = (img_size,img_size)

        self.img_height=img_size[0]
        self.img_width=img_size[1]
        self.patch_size=patch_size
        self.num_frames=num_frames
        self.tubelet_size=tubelet_size
        self.use_activation_checkpointing=use_activation_checkpointing
        self.is_video = num_frames > 1
        self.out_layers = out_layers

        #modality tokenization
           #video 
        if self.is_video:
            self.patch_embed=PatchEmbed3D(patch_size=patch_size,tubelet_size=tubelet_size,embed_dim=embed_dim,in_chans=in_chans)
            self.num_patches=(num_frames//tubelet_size)*(img_size[0]//patch_size)*(img_size[1]//patch_size)

        else:
            self.patch_embed=PatchEmbed(patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
            self.num_patches=(img_size[0]//patch_size)*(img_size[1]//patch_size)

        #Attention blocks

        self.blocks = nn.ModuleList(
            [
                Block(
            grid_size=img_size[0]//patch_size,
            grid_depth=num_frames//tubelet_size,
            use_rope=use_rope,
            num_heads=num_heads,
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer,
            attn_drop=attn_drop_rate
            ) 
            for _ in range(depth)
            ]
        )

        self.norm=norm_layer(embed_dim)

        #initializing weights
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

            for layer_id, layer in enumerate(self.blocks):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self,x,masks=None):
        if masks is not None and not isinstance(masks,list):
            masks=[masks]
        #tokenize inputss
        #image
        if x.ndim==4:
            _,_,H,W = x.shape
            T=1
        #video
        elif x.ndim==5:
            _,_,T,H,W = x.shape
            T = T // self.tubelet_size
        H_patches = H//self.patch_size
        W_patches = W//self.patch_size

        x=self.patch_embed(x)
        #mask away unwanted tokens
        if masks is not None:
            x = apply_masks(x,masks)
            masks = torch.cat(masks,dim=0)

        #fwd
        outs = []

        for i,blk in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                blk, x, masks, None, T=T, H_patches=H_patches, W_patches=W_patches, use_reentrant=False
            )
            else:
                x=blk(x,mask=masks,attn_mask=None,T=T,H_patches=H_patches,W_patches=W_patches)

            if self.out_layers is not None and i in self.out_layers :
                outs.append(self.norm(x))
        if self.out_layers is not None:
            return outs 
        if self.norm is not None:
            x = self.norm(x)

        return x


def vit_small_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
def vit_base_rope(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def vit_large_rope(patch_size=16, **kwargs):
    model=VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
       
        **kwargs
    )
    return model

def vit_huge_rope(patch_size=16,**kwargs):
    model=VisionTransformer(
            patch_size=patch_size,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_rope=True,
            **kwargs
        )
    return model









            
            









