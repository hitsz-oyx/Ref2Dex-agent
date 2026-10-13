"""Per-horizon displacement standardization from Appendix B.5.

Scene coordinates are clipped to [-0.5, 0.5] m before the statistics and
before the loss. Hand coordinates are not clipped. A coordinate with no
variation uses standard deviation 1, so its standardized target is 0.
Fine-tuning is expected to keep these pre-training statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SCENE_CLIP_M = 0.5


@dataclass
class HorizonStats:
    scene_mean: torch.Tensor
    scene_std: torch.Tensor
    hand_mean: torch.Tensor
    hand_std: torch.Tensor

    def standardize_scene(self, displacement: torch.Tensor) -> torch.Tensor:
        clipped = displacement.clamp(-SCENE_CLIP_M, SCENE_CLIP_M)
        return _apply(clipped, self.scene_mean, self.scene_std)

    def standardize_hand(self, displacement: torch.Tensor) -> torch.Tensor:
        return _apply(displacement, self.hand_mean, self.hand_std)

    def unstandardize_scene(self, standardized: torch.Tensor) -> torch.Tensor:
        return _apply(standardized, self.scene_mean, self.scene_std, inverse=True)

    def unstandardize_hand(self, standardized: torch.Tensor) -> torch.Tensor:
        return _apply(standardized, self.hand_mean, self.hand_std, inverse=True)


def _moment(values: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate moments. ``values`` is ``[T, ..., 3]`` and ``valid`` is ``[T, ...]``."""
    mask = valid & torch.isfinite(values).all(dim=-1)
    safe = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    total = mask.to(values.dtype).sum(dim=tuple(range(1, mask.ndim)))
    summed = safe.sum(dim=tuple(range(1, safe.ndim - 1)))
    squared = (safe ** 2).sum(dim=tuple(range(1, safe.ndim - 1)))
    return summed, squared, total


def _finalize(summed: torch.Tensor, squared: torch.Tensor, count: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = count.clamp_min(1).unsqueeze(-1)
    mean = summed / denominator
    variance = (squared / denominator - mean ** 2).clamp_min(0)
    std = variance.sqrt()
    std = torch.where((count.unsqueeze(-1) > 0) & (std > 1e-8), std, torch.ones_like(std))
    mean = torch.where(count.unsqueeze(-1) > 0, mean, torch.zeros_like(mean))
    return mean, std


def fit_horizon_stats(examples: list[dict]) -> HorizonStats:
    """Estimate scene and hand statistics from fit clips only."""
    scene_sum = scene_sq = hand_sum = hand_sq = None
    scene_count = hand_count = None
    for example in examples:
        scene = torch.as_tensor(example["scene_displacement"], dtype=torch.float64)
        scene = scene.clamp(-SCENE_CLIP_M, SCENE_CLIP_M)
        scene_valid = torch.as_tensor(example["scene_target_valid"], dtype=torch.bool)
        hand = torch.as_tensor(example["hand_displacement"], dtype=torch.float64)
        hand_valid = torch.as_tensor(example["hand_target_valid"], dtype=torch.bool)
        s_sum, s_sq, s_count = _moment(scene, scene_valid)
        h_sum, h_sq, h_count = _moment(hand, hand_valid)
        scene_sum = s_sum if scene_sum is None else scene_sum + s_sum
        scene_sq = s_sq if scene_sq is None else scene_sq + s_sq
        scene_count = s_count if scene_count is None else scene_count + s_count
        hand_sum = h_sum if hand_sum is None else hand_sum + h_sum
        hand_sq = h_sq if hand_sq is None else hand_sq + h_sq
        hand_count = h_count if hand_count is None else hand_count + h_count
    if scene_sum is None or hand_sum is None or scene_count is None or hand_count is None:
        raise ValueError("statistics need at least one clip")
    scene_mean, scene_std = _finalize(scene_sum, scene_sq, scene_count)
    hand_mean, hand_std = _finalize(hand_sum, hand_sq, hand_count)
    return HorizonStats(
        scene_mean.float(), scene_std.float(), hand_mean.float(), hand_std.float())


def _apply(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    if values.shape[-1] != 3 or values.shape[0] != mean.shape[0]:
        raise ValueError("standardization expects time on axis 0 and xyz on the last axis")
    view = [mean.shape[0]] + [1] * (values.ndim - 2) + [3]
    center = mean.to(device=values.device, dtype=values.dtype).reshape(view)
    scale = std.to(device=values.device, dtype=values.dtype).reshape(view)
    if inverse:
        return values * scale + center
    return (values - center) / scale
