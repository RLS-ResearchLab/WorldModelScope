import torch 
import torch.nn as nn
import timm
class EUPEEncoder(nn.Module):
    def __init__(self, img_size: int = 224):
        super().__init__()

        self.model = timm.create_model(
            "vit_small_patch16_dinov3_qkvb.eupe_lvd1689m",
            pretrained=True,
            num_classes=0,
            img_size=img_size,   # <-- force 224 so patch grid = 14x14 = 196,
                                  #     matching predictor's num_patches=197 (196+action)
        )

        self.model.eval()

        self.embed_dim = 384
        self.patch_size = 16
        self.img_size = img_size

        self.num_prefix_tokens = self.model.num_prefix_tokens

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            raise ValueError(
                f"EUPEEncoder was built for {self.img_size}x{self.img_size} "
                f"input (to match the predictor's num_patches), but got "
                f"{tuple(x.shape[-2:])}. Resize upstream instead of changing "
                f"the encoder's img_size."
            )

        features = self.model.forward_features(x)
        patch_tokens = features[:, self.num_prefix_tokens:, :]
        return patch_tokens