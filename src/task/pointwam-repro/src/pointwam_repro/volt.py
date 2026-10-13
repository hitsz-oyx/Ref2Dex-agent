"""Volt encoder blocks used by PointWAM.

Each block is pre-norm, bidirectional attention, and a GELU MLP with the
usual 4x width. Stochastic depth rises linearly from 0 to the configured
maximum. Axial RoPE is applied only after the instruction prefix.
"""

from __future__ import annotations

import torch
from torch import nn

from pointwam_repro.rope import apply_axial_rope


class DropPath(nn.Module):
    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = float(rate)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0.0:
            return values
        keep = 1.0 - self.rate
        mask = values.new_empty((values.shape[0], 1, 1)).bernoulli_(keep)
        return values * mask / keep


class VoltBlock(nn.Module):
    def __init__(self, width: int, heads: int, drop_rate: float, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must divide evenly across heads")
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        hidden = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.drop_attn = DropPath(drop_rate)
        self.drop_mlp = DropPath(drop_rate)

    def forward(self, tokens: torch.Tensor, vision_coords: torch.Tensor, n_text: int) -> torch.Tensor:
        hidden = self.norm1(tokens)
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        vision_query, vision_key = apply_axial_rope(query[:, :, n_text:], key[:, :, n_text:], vision_coords)
        query = torch.cat((query[:, :, :n_text], vision_query), dim=2)
        key = torch.cat((key[:, :, :n_text], vision_key), dim=2)
        attended = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.drop_attn(self.proj(attended))
        update = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm2(tokens))))
        return tokens + self.drop_mlp(update)


class VoltEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, drop_path_max: float = 0.3) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("Volt needs at least one layer")
        if layers == 1:
            rates = [0.0]
        else:
            rates = [drop_path_max * index / (layers - 1) for index in range(layers)]
        self.drop_rates = tuple(rates)
        self.blocks = nn.ModuleList(
            VoltBlock(width, heads, rate) for rate in rates
        )

    def forward(self, tokens: torch.Tensor, vision_coords: torch.Tensor, n_text: int) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must be [batch, length, width]")
        if n_text < 1 or n_text >= tokens.shape[1]:
            raise ValueError("the sequence needs instruction tokens and at least one vision token")
        hidden = tokens
        for block in self.blocks:
            hidden = block(hidden, vision_coords, n_text)
        return hidden
