"""PointWAM trajectory forecaster from Appendix B.3 and Table 4.

Official training code is unpublished. This module follows the paper's stem,
5^3 patches, Volt-H sizes, detokenizer, zero-initialized fusion residual, and
separate scene/hand heads. Frozen Mosaic3D and Recap-CLIP stay outside and
are passed in as features.

Published sizes are 32 layers, width 1280, 20 heads, stem width 64, and a
1 cm voxel. The detokenized voxel width stays at 64: expanding every token to
125 vectors of width 1280 would add about 200M parameters beyond the stated
651M trainable budget.
"""

from __future__ import annotations

import torch
from torch import nn

from pointwam_repro.sparse import SparseStem, mean_pool_voxels
from pointwam_repro.tokenize import voxel_patch_slots
from pointwam_repro.volt import VoltEncoder

PAPER_SCENE_DIM = 768
PAPER_TEXT_DIM = 1024
PAPER_WIDTH = 1280
PAPER_LAYERS = 32
PAPER_HEADS = 20
PAPER_STEM_DIM = 64
PAPER_HORIZON = 30


def paper_adamw(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01) -> torch.optim.Optimizer:
    """Table 4 AdamW. Betas are the PyTorch default; the paper does not state them."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))


class PointWAMForecaster(nn.Module):
    def __init__(
        self,
        scene_dim: int = PAPER_SCENE_DIM,
        text_dim: int = PAPER_TEXT_DIM,
        width: int = PAPER_WIDTH,
        layers: int = PAPER_LAYERS,
        heads: int = PAPER_HEADS,
        stem_dim: int = PAPER_STEM_DIM,
        horizon: int = PAPER_HORIZON,
        drop_path_max: float = 0.3,
        voxel_m: float = 0.01,
        patch_size: int = 5,
    ) -> None:
        super().__init__()
        if patch_size != 5 or horizon < 1:
            raise ValueError("PointWAM patches are 5 voxels and the horizon is positive")
        self.scene_dim = scene_dim
        self.text_dim = text_dim
        self.stem_dim = stem_dim
        self.horizon = horizon
        self.voxel_m = voxel_m
        self.patch_size = patch_size
        self.hand_mlp = nn.Sequential(
            nn.Linear(11, scene_dim),
            nn.GELU(),
            nn.Linear(scene_dim, scene_dim),
        )
        self.stem = SparseStem(scene_dim, stem_dim)
        slots = patch_size ** 3
        self.patch_proj = nn.Linear(slots * stem_dim, width)
        self.text_proj = nn.Linear(text_dim, width)
        self.backbone = VoltEncoder(width, layers, heads, drop_path_max)
        self.detoken_in = nn.Sequential(
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.expand = nn.Linear(width, slots * stem_dim)
        self.fuse_in = nn.Linear(stem_dim * 2, stem_dim)
        self.fuse_norm = nn.LayerNorm(stem_dim)
        self.fuse_out = nn.Linear(stem_dim, stem_dim)
        nn.init.zeros_(self.fuse_out.weight)
        nn.init.zeros_(self.fuse_out.bias)
        self.out_norm = nn.LayerNorm(stem_dim)
        self.scene_head = nn.Sequential(
            nn.Linear(stem_dim, stem_dim),
            nn.GELU(),
            nn.Linear(stem_dim, horizon * 3),
        )
        self.hand_head = nn.Sequential(
            nn.Linear(stem_dim, stem_dim),
            nn.GELU(),
            nn.Linear(stem_dim, horizon * 3),
        )

    def fuse(self, decoded: torch.Tensor, stem: torch.Tensor) -> torch.Tensor:
        """Residual fusion. The last linear starts at zero, so training starts at the detokenized features."""
        hidden = self.fuse_in(torch.cat((decoded, stem), dim=-1))
        update = self.fuse_out(torch.nn.functional.gelu(self.fuse_norm(hidden)))
        return torch.nn.functional.gelu(self.out_norm(decoded + update))

    def forward_observation(
        self,
        scene_xyz: torch.Tensor,
        scene_features: torch.Tensor,
        scene_valid: torch.Tensor,
        hand_xyz: torch.Tensor,
        hand_valid: torch.Tensor,
        text_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forecast standardized displacements for one observation."""
        if scene_xyz.ndim != 2 or scene_features.shape != (scene_xyz.shape[0], self.scene_dim):
            raise ValueError("scene points and Mosaic-sized features must align")
        if hand_xyz.shape != (2, 10, 3) or hand_valid.shape != (2, 10):
            raise ValueError("hands must be two groups of ten keypoints")
        if text_features.ndim != 2 or text_features.shape[-1] != self.text_dim or text_features.shape[0] < 1:
            raise ValueError("text features must be [L, text_dim] with L >= 1")
        scene_keep = scene_valid.nonzero(as_tuple=False).flatten()
        hand_keep = hand_valid.nonzero(as_tuple=False)
        if len(scene_keep) + len(hand_keep) == 0:
            raise ValueError("observation has no valid scene or hand point")
        dtype = self.patch_proj.weight.dtype
        scene_feat = scene_features.to(dtype=dtype)[scene_keep]
        hand_feat = self._hand_features(hand_keep)
        xyz = torch.cat((
            scene_xyz.to(dtype=dtype)[scene_keep],
            hand_xyz.to(dtype=dtype)[hand_keep[:, 0], hand_keep[:, 1]],
        ), dim=0)
        features = torch.cat((scene_feat, hand_feat), dim=0)
        voxel = torch.floor(xyz / self.voxel_m).to(torch.int64)
        pooled, voxel_coords, inverse = mean_pool_voxels(features, voxel)
        stem = self.stem(pooled, voxel_coords)
        point_stem = stem[inverse]
        packed = _pack_stem(stem, voxel_coords, self.patch_size)
        vision = self.patch_proj(packed["tokens"].flatten(1))
        text = self.text_proj(text_features.to(dtype=dtype))
        sequence = torch.cat((text, vision), dim=0).unsqueeze(0)
        encoded = self.backbone(sequence, packed["patch_xyz"], text.shape[0])
        vision_out = encoded[0, text.shape[0]:]
        point_patch = packed["inverse"][inverse]
        point_slot = packed["slot"][inverse]
        decoded_tokens = self.detoken_in(vision_out)
        expanded = self.expand(decoded_tokens).view(vision_out.shape[0], self.patch_size ** 3, self.stem_dim)
        decoded = expanded[point_patch, point_slot]
        fused = self.fuse(decoded, point_stem)
        scene_out = scene_xyz.new_zeros((self.horizon, scene_xyz.shape[0], 3), dtype=dtype)
        hand_out = scene_xyz.new_zeros((self.horizon, 2, 10, 3), dtype=dtype)
        n_scene = len(scene_keep)
        if n_scene:
            scene_out[:, scene_keep] = self._head(self.scene_head, fused[:n_scene]).to(scene_out.dtype)
        if len(hand_keep):
            hand_values = self._head(self.hand_head, fused[n_scene:]).to(hand_out.dtype)
            hand_out[:, hand_keep[:, 0], hand_keep[:, 1]] = hand_values
        return {"scene_displacement": scene_out, "hand_displacement": hand_out}

    def _hand_features(self, hand_keep: torch.Tensor) -> torch.Tensor:
        if len(hand_keep) == 0:
            return self.patch_proj.weight.new_zeros((0, self.scene_dim))
        index = torch.nn.functional.one_hot(hand_keep[:, 1].long(), 10)
        identity = hand_keep[:, 0].to(self.patch_proj.weight.dtype).unsqueeze(-1)
        raw = torch.cat((index.to(identity.dtype), identity), dim=-1)
        return self.hand_mlp(raw)

    def _head(self, head: nn.Module, features: torch.Tensor) -> torch.Tensor:
        raw = head(features).view(features.shape[0], self.horizon, 3)
        return raw.transpose(0, 1)


def _pack_stem(features: torch.Tensor, coords: torch.Tensor, patch_size: int) -> dict[str, torch.Tensor]:
    patch, slot = voxel_patch_slots(coords, patch_size)
    unique, inverse = torch.unique(patch, dim=0, return_inverse=True)
    order = torch.arange(unique.shape[0], device=unique.device)
    for axis in (2, 1, 0):
        order = order[torch.argsort(unique[order, axis], stable=True)]
    unique = unique[order]
    remap = torch.empty_like(order)
    remap[order] = torch.arange(order.shape[0], device=order.device)
    inverse = remap[inverse]
    tokens = features.new_zeros((unique.shape[0], patch_size ** 3, features.shape[1]))
    tokens[inverse, slot] = features.to(tokens.dtype)
    return {"tokens": tokens, "patch_xyz": unique, "inverse": inverse, "slot": slot}
