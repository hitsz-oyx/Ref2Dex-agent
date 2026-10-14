"""Same Chunk-R trunk; an explicit output shape for branch-preserving targets."""
from torch import nn
import torch
from .paths import add_dependencies
add_dependencies()
from trajectory_policy.chunk_retargeter import ChunkRetargeter
from .action_targets import NATIVE, action_dimension


def make_retargeter(representation=NATIVE,width=128):
    model=ChunkRetargeter(width)
    dimensions=action_dimension(representation)
    if dimensions!=12:
        model.head=nn.Linear(width,dimensions)
        nn.init.zeros_(model.head.weight);nn.init.zeros_(model.head.bias)
        model.action_mean=torch.zeros(8,dimensions)
        model.action_scale=torch.ones(8,dimensions)
    model.action_representation=representation
    return model
