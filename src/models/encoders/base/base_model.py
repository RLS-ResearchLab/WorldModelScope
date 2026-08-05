import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Parent class for all models.

    Provides:
    - checkpoint saving/loading
    - freezing
    - unfreezing
    - parameter counting
    """

    def __init__(self):
        super().__init__()


    def save_checkpoint(self, path):
        """
        Save model weights.
        """

        torch.save(
            self.state_dict(),
            path
        )

        print(f"Checkpoint saved at {path}")



    def load_checkpoint(self, path, device="cpu"):
        """
        Load model weights.
        """

        checkpoint = torch.load(
            path,
            map_location=device
        )

        self.load_state_dict(checkpoint)

        print(f"Checkpoint loaded from {path}")



    def freeze(self):
        """
        Freeze all parameters.
        """

        for param in self.parameters():
            param.requires_grad = False



    def unfreeze(self):
        """
        Enable training again.
        """

        for param in self.parameters():
            param.requires_grad = True



    def count_parameters(self):

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


        return {
            "total": total,
            "trainable": trainable
        }