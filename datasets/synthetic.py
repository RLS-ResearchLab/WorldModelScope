# src/data/synthetic.py
import torch
from torch.utils.data import Dataset


class SyntheticVideoDataset(Dataset):
    """Random video clips -- purely for verifying the training pipeline runs and loss decreases.
    Not meant to produce meaningful representations; swap for real video once this works."""

    def __init__(self, num_samples=200, num_frames=4, img_size=64, in_chans=3):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.img_size = img_size
        self.in_chans = in_chans

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.in_chans, self.num_frames, self.img_size, self.img_size)