import torch
import torch.nn as nn


class EUPEEncoder(nn.Module):

    def __init__(
        self,
        repo_dir,
        checkpoint_path,
        freeze=True,
    ):
        super().__init__()

        self.model = torch.hub.load(
            repo_dir,
            "eupe_vits16",
            source="local",
            weights=checkpoint_path,
        )

        self.embed_dim = 384
        self.patch_size = 16

        if freeze:
            self.freeze()

    def freeze(self):

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, images):

        outputs = self.model.forward_features(images)

        patch_tokens = outputs["x_norm_patchtokens"]

        return patch_tokens