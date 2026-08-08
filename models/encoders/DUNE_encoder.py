import torch
import torch.nn as nn


class DUNEEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = torch.hub.load(
            "naver/dune",
            "dune_vitsmall_14_336_encoder"
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        self.embed_dim = 384
        self.patch_size = 14

    @torch.no_grad()
    def forward(self, images):

        outputs = self.model(images)

        return outputs["x_norm_patchtokens"]