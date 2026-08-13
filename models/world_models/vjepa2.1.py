import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Projects encoder features to a common dimension.

    Input:
        [B, N, D_in]

    Output:
        [B, N, D_out]
    """
 
    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            output_dim,
        )

    def forward(self, x):
        return self.projection(x)