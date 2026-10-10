"""PTv3 point-flow pretraining with the native scene encoding topology."""
import torch
from torch import nn

from .model import mean_groups
from .pointworld import PointWorldWM, fixed_workspace_grid


class VideoPointWorldWM(PointWorldWM):
    def __init__(self, scene_mean, scene_std, patch_size=128):
        stats = dict(scene_mean=scene_mean, scene_std=scene_std,
                     action_mean=[0.] * 9, action_std=[1.] * 9,
                     flow_mean=[[0.] * 3] * 24, flow_std=[[1.] * 3] * 24,
                     translation_mean=[[0.] * 3] * 24, translation_std=[[1.] * 3] * 24,
                     rotation_scale=[1.] * 24)
        super().__init__(stats, patch_size)
        # The native backbone/scene projection can be transferred later, but
        # rigid effects and observed-future-hand modules do not run in this arm.
        for name in ('action_proj', 'hand', 'keypoint', 'action_type',
                     'hand_gamma', 'hand_beta', 'object_proj'):
            delattr(self, name)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs):
        # Deliberately accept only the observed input contract.
        if set(inputs) != {'xyz', 'features', 'point_valid'}:
            raise ValueError('video forward accepts observed inputs only')
        xyz, valid = inputs['xyz'], inputs['point_valid']
        features = torch.where(valid[..., None], inputs['features'],
                               torch.zeros_like(inputs['features']))
        features = (features - self.scene_mean) / self.scene_std
        scene = self.scene_proj(features)
        bids = torch.arange(len(xyz), device=xyz.device)[:, None].expand_as(valid)
        with torch.autocast('cuda', enabled=False):
            grid = fixed_workspace_grid(xyz, valid, self.voxel_origin_m, self.voxel_size_m)[valid]
            keys = torch.cat((bids[valid][:, None], grid), dim=-1)
            unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
            point = self.backbone(dict(coord=mean_groups(xyz[valid].float(), inverse, len(unique)),
                                       feat=mean_groups(scene[valid].float(), inverse, len(unique)),
                                       grid_coord=unique[:, 1:].int(), batch=unique[:, 0],
                                       grid_size=self.voxel_size_m))
        packed = point.feat.new_zeros((*valid.shape, 128))
        packed[valid] = point.feat[inverse]
        local = packed + scene * self.skip_gamma + self.skip_beta
        flow = self.head(local[:, :, None] + self.time.weight[None, None]).float() * .01
        return flow
