"""Vectorize the existing 256-token point selection without changing point IDs."""
import torch


def vectorized_scene_tokens(self, packed, valid):
    if self.token_count != 256:
        raise ValueError('integer-exact sampler is bounded to the current256-token contract')
    batch, points, hidden = packed.shape
    counts = valid.sum(1)
    if not bool((counts > 0).all()):
        raise ValueError('scene encoder received no valid point')
    positions = torch.arange(points, device=packed.device).expand(batch, -1)
    ordered = torch.where(valid, positions, points).sort(1).values
    slots = torch.arange(256, device=packed.device).expand(batch, -1)
    # Denominator255 is odd, so integer nearest rounding has no half ties.
    # Existing float32 linspace+round gives these IDs in the bounded workspace.
    selected_rank = (slots * (counts[:, None] - 1) + 127) // 255
    selected_rank = torch.where(counts[:, None] > 256, selected_rank, slots)
    selected = ordered.gather(1, selected_rank.clamp_max(points - 1)).clamp_max(points - 1)
    mask = slots < counts[:, None].clamp_max(256)
    tokens = packed.gather(1, selected[..., None].expand(-1, -1, hidden))
    return torch.where(mask[..., None], tokens, torch.zeros_like(tokens)), mask


def install_vectorized_sampling():
    from .model import SceneHistoryEncoder
    original = SceneHistoryEncoder._sample_tokens
    SceneHistoryEncoder._sample_tokens = vectorized_scene_tokens
    return original
