import torch
import torch.nn as nn

from models.base.base_encoder import BaseEncoder



class DINOEncoder(BaseEncoder):
    """
    DINOv2 Vision Transformer encoder.

    Input:
        images:
            B x 3 x H x W

    Output:
        latent representation:
            B x N x D

    B = batch size
    N = number of patches
    D = embedding dimension
    """


    def __init__(
        self,
        model_name="dinov2_vit_base",
        pretrained=True,
        freeze=True
    ):

        super().__init__()



        # Load DINOv2 backbone
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name
        )


        self.embed_dim = self.backbone.embed_dim



        if freeze:
            self.freeze()



    def encode(self,x):

        features = self.backbone.forward_features(x)

        cls_token = features["x_norm_clstoken"]

        patch_tokens = features["x_norm_patchtokens"]


        return {
            "cls": cls_token,
            "patches": patch_tokens
        }