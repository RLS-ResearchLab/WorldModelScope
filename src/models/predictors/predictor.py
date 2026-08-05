import math
import torch
import torch.nn as nn
from functools import partial

from src.models.utils.modules import Block
from src.models.utils.tensors import trunc_normal_


class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qk_scale=None,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        use_rope=True,
        init_std=0.02,
        use_activation_checkpointing=False,
        **kwargs,
    ):
        super().__init__()

        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        # -- project encoder's embed_dim down into the predictor's own working dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # -- single learned token standing in for "unknown content" at every target position
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    dim=predictor_embed_dim,
                    grid_size=self.grid_height,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, context_tokens, masks_ctxt, masks_target, T=None, H_patches=None, W_patches=None):
        """
        context_tokens: [B, N_ctxt, embed_dim]   -- encoder output, context-masked
        masks_ctxt:     [B, N_ctxt]               -- original grid indices the context tokens came from
        masks_target:   [B, N_target]             -- original grid indices to be predicted
        """
        B = context_tokens.shape[0]

        # -- project context tokens into predictor's working dim
        x = self.predictor_embed(context_tokens)

        # -- build mask-token placeho.0l0ders for every target position
        pred_tokens = self.mask_token.repeat(B, masks_target.shape[1], 1)

        # -- concatenate context + target placeholders into one sequence
        x = torch.cat([x, pred_tokens], dim=1)

        # -- concatenate their true grid indices too, so RoPE knows real positions of both
        combined_mask = torch.cat([masks_ctxt, masks_target], dim=1)

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, combined_mask, None, T, H_patches, W_patches, use_reentrant=False
                )
            else:
                x = blk(x, mask=combined_mask, attn_mask=None, T=T, H_patches=H_patches, W_patches=W_patches)

        x = self.predictor_norm(x)

        # -- keep only the predictions at target positions, discard context-token outputs
        x = x[:, -masks_target.shape[1] :, :]
        x = self.predictor_proj(x)

        return x


def vit_predictor(**kwargs):
    model = VisionTransformerPredictor(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model