import torch 
import torch.nn as nn

class VGGTEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = VGGT.from_pretrained(
            "facebook/VGGT-1B"
        )

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, images):

        # images: [B,1,3,518,518]

        aggregated_tokens, patch_start_idx = (
            self.model.aggregator(images)
        )

        features = aggregated_tokens[-1]

        features = features[
            :, :, patch_start_idx:, :
        ]

        return features