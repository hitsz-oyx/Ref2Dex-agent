"""Small BC-trained object encoder shared by tau/no-tau executor profiles."""
import torch
from torch import nn


class ObjectShapeEncoder(nn.Module):
    output_dim=64

    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(3,64),nn.ELU(),nn.Linear(64,128),nn.ELU(),nn.Linear(128,64))

    def forward(self,points):
        if points.ndim!=3 or points.shape[-1]!=3 or points.shape[1]<3:raise ValueError('batched canonical object points in metres required')
        # Fixed physical scale retains size, unlike per-object unit-sphere normalization.
        return self.point(points/.1).amax(1)

    def freeze(self):
        self.eval();self.requires_grad_(False)
