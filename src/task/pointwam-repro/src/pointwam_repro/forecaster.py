"""Reduced scene/hand forecaster with the paper's patch interface.

Frozen Mosaic3D and Recap-CLIP are not loaded. Scene features come from a
learned XYZ/RGB MLP, and language is a single learned token. Two transformer
layers are a local smoke width, not Volt-H.
"""

from __future__ import annotations

import torch
from torch import nn

from pointwam_repro.tokenize import pack_patches


class _Head(nn.Module):
    def __init__(self, width: int, feature_dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(width + feature_dim, width),
            nn.GELU(),
            nn.Linear(width, horizon * 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, token: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat((token, feature), dim=-1))
        return raw.reshape(raw.shape[0], self.horizon, 3)


class ReducedPatchForecaster(nn.Module):
    def __init__(self, feature_dim: int = 16, width: int = 64, layers: int = 2, horizon: int = 30) -> None:
        super().__init__()
        if width % 4 or layers < 1 or horizon < 1:
            raise ValueError("invalid reduced forecaster size")
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.scene_feature = nn.Sequential(nn.Linear(6, feature_dim), nn.GELU())
        self.hand_id = nn.Embedding(2, feature_dim)
        self.hand_index = nn.Embedding(10, feature_dim)
        self.patch_proj = nn.Linear(125 * feature_dim, width)
        self.language = nn.Parameter(torch.zeros(width))
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.scene_head = _Head(width, feature_dim, horizon)
        self.hand_head = _Head(width, feature_dim, horizon)

    def forward_clip(
        self,
        scene_xyz: torch.Tensor,
        scene_rgb: torch.Tensor,
        scene_valid: torch.Tensor,
        hand_xyz: torch.Tensor,
        hand_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forecast displacements for one current cloud and two hands."""
        if scene_xyz.ndim != 2 or hand_xyz.shape != (2, 10, 3):
            raise ValueError("expected one scene cloud and two 10-point hands")
        scene_keep = scene_valid.nonzero(as_tuple=False).flatten()
        hand_keep = hand_valid.nonzero(as_tuple=False)
        scene_feat = self.scene_feature(torch.cat((
            scene_xyz, scene_rgb.to(scene_xyz.dtype) / 255), dim=-1))
        hand_feat = self.hand_id(hand_keep[:, 0]) + self.hand_index(hand_keep[:, 1])
        xyz = torch.cat((scene_xyz[scene_keep], hand_xyz[hand_valid]), dim=0)
        features = torch.cat((scene_feat[scene_keep], hand_feat), dim=0)
        packed = pack_patches(features, xyz)
        vision = self.patch_proj(packed["tokens"].flatten(1))
        sequence = torch.cat((self.language.view(1, -1), vision), dim=0).unsqueeze(0)
        encoded = self.encoder(sequence)[0, 1:]
        point_token = encoded[packed["point_patch"]]
        scene_out = scene_xyz.new_zeros((scene_xyz.shape[0], self.horizon, 3))
        hand_out = scene_xyz.new_zeros((2, 10, self.horizon, 3))
        if len(scene_keep):
            scene_out[scene_keep] = self.scene_head(
                point_token[: len(scene_keep)], features[: len(scene_keep)])
        if len(hand_keep):
            predicted = self.hand_head(point_token[len(scene_keep):], features[len(scene_keep):])
            hand_out[hand_keep[:, 0], hand_keep[:, 1]] = predicted
        return {
            "scene_displacement": scene_out.transpose(0, 1),
            "hand_displacement": hand_out.permute(2, 0, 1, 3),
        }
