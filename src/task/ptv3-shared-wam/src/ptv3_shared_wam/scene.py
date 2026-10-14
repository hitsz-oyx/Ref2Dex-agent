"""The unchanged PTv3-small architecture receives ONLY scene points."""
import torch

from .paths import add_dependencies

add_dependencies()
from goal_point_wam.model import SceneHistoryEncoder, _sparse_context
from oakink_wm.model import mean_groups
from oakink_wm.pointworld import fixed_workspace_grid


class SceneEncoder(SceneHistoryEncoder):
    def __init__(self, stats, config):
        super().__init__(stats, patch_size=config.get("patch_size", 128),
                         token_count=config.get("scene_tokens", 256),
                         voxel_origin_m=config.get("voxel_origin_m", [-2., -2., -2.]),
                         voxel_size_m=config.get("voxel_m", .01))
        # No inherited object-pose head or unused trainable object projector.
        del self.object_proj
        if config.get('vectorized_padding', True):
            from types import MethodType
            from ptv3.ptv3 import SerializedAttention
            from .padding import vectorized_padding
            for module in self.backbone.modules():
                if isinstance(module, SerializedAttention):
                    module.get_padding_and_inverse = MethodType(vectorized_padding,module)

    def _sample_tokens(self, packed, valid):
        """Match legacy rounded evenly spaced valid IDs without per-row nonzero."""
        batch, points, channels = packed.shape
        counts = valid.sum(1)
        if (counts == 0).any():
            raise ValueError("scene encoder received no valid point")
        ids = torch.arange(points, device=packed.device).expand(batch, -1)
        ordered = torch.where(valid, ids, points).sort(1).values
        slot = torch.arange(self.token_count, device=packed.device)
        if self.token_count == 1:
            ranks = torch.zeros((batch, 1), dtype=torch.long, device=packed.device)
        else:
            even = (counts[:, None]-1).float() * (slot.float() / (self.token_count-1))
            ranks = torch.where(counts[:, None]>self.token_count, even.round().long(), slot[None])
        mask = slot[None] < counts[:, None].clamp_max(self.token_count)
        selected = ordered.gather(1, ranks.clamp_max(points-1)).clamp_max(points-1)
        tokens = packed.gather(1, selected[..., None].expand(-1, -1, channels))
        return torch.where(mask[..., None], tokens, 0.), mask

    def forward(self, batch):
        xyz = batch["xyz"].float()
        # Hands never share the sparse voxel aggregation with object surfaces.
        valid = batch["point_valid"].bool() & (batch["features"][..., 14] < .5)
        if not valid.any(1).all():
            raise ValueError("each scene needs at least one observed non-hand point")
        features = torch.where(valid[..., None], batch["features"].float(), 0.)
        projected = self.scene_proj((features - self.scene_mean) / self.scene_std)
        ids = torch.arange(len(xyz), device=xyz.device)[:, None].expand_as(valid)
        with _sparse_context(xyz):
            grid = fixed_workspace_grid(xyz, valid, self.voxel_origin_m, self.voxel_size_m)
            keys = torch.cat((ids[valid][:, None], grid[valid]), -1)
            unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
            encoded = self.backbone(dict(
                coord=mean_groups(xyz[valid], inverse, len(unique)),
                feat=mean_groups(projected.float()[valid], inverse, len(unique)),
                grid_coord=unique[:, 1:].int(), batch=unique[:, 0], grid_size=self.voxel_size_m))
        packed = encoded.feat.new_zeros((*valid.shape, 128))
        packed[valid] = encoded.feat[inverse]
        # One selection applied to both geometry and feature tokens.
        tokens, token_mask = self._sample_tokens(torch.cat((packed, xyz), -1), valid)
        return dict(tokens=tokens[..., :128], xyz=tokens[..., 128:], mask=token_mask)
