"""PTv3 point-flow pretraining with the native scene encoding topology."""
import torch
from torch import nn

from .model import mean_groups
from .pointworld import PointWorldWM, fixed_workspace_grid
from .video_points import MOTION_FEATURES, MOTION_INTERVALS


class VideoPointWorldWM(PointWorldWM):
    def __init__(self, scene_mean, scene_std, patch_size=128, *,
                 motion_mean=None, motion_std=None, motion_token=False):
        stats = dict(scene_mean=scene_mean, scene_std=scene_std,
                     action_mean=[0.] * 9, action_std=[1.] * 9,
                     flow_mean=[[0.] * 3] * 24, flow_std=[[1.] * 3] * 24,
                     translation_mean=[[0.] * 3] * 24, translation_std=[[1.] * 3] * 24,
                     rotation_scale=[1.] * 24)
        super().__init__(stats, patch_size)
        self.motion_token = bool(motion_token)
        if self.motion_token:
            if motion_mean is None or motion_std is None:
                raise ValueError('motion token requires train-only mean/std')
            motion_mean = torch.as_tensor(motion_mean, dtype=torch.float32)
            motion_std = torch.as_tensor(motion_std, dtype=torch.float32)
            if motion_mean.shape != (MOTION_FEATURES,) or motion_std.shape != (MOTION_FEATURES,):
                raise ValueError('motion statistics must have shape (15,)')
            if not torch.isfinite(motion_mean).all() or not torch.isfinite(motion_std).all():
                raise ValueError('motion statistics must be finite')
            if (motion_std <= 0).any():
                raise ValueError('motion standard deviations must be positive')
            self.register_buffer('motion_mean', motion_mean)
            self.register_buffer('motion_std', motion_std)
            self.motion_proj = nn.Sequential(
                nn.Linear(MOTION_FEATURES, 128), nn.GELU(), nn.Linear(128, 128)
            )
            self.motion_interval = nn.Embedding(MOTION_INTERVALS, 128)
            self.motion_group = nn.Embedding(2, 128)
            self.motion_invalid = nn.Parameter(torch.zeros(128))
        # The native backbone/scene projection can be transferred later, but
        # rigid effects and observed-future-hand modules do not run in this arm.
        for name in ('action_proj', 'hand', 'keypoint', 'action_type',
                     'hand_gamma', 'hand_beta', 'object_proj'):
            delattr(self, name)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, inputs, use_motion=True):
        # Deliberately accept only the observed input contract.
        base_keys = {'xyz', 'features', 'point_valid'}
        motion_keys = {'motion_stats', 'motion_valid'}
        if self.motion_token:
            if set(inputs) not in (base_keys, base_keys | motion_keys):
                raise ValueError('video motion forward received an unknown input contract')
            if set(inputs) == base_keys:
                raise ValueError('motion-token model requires history motion inputs')
        elif set(inputs) != base_keys:
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
        context = local.new_zeros((len(xyz), 128))
        if self.motion_token and use_motion:
            motion = inputs['motion_stats'].to(local.device, dtype=local.dtype)
            motion_valid = inputs['motion_valid'].to(local.device)
            motion = (motion - self.motion_mean) / self.motion_std
            interval = torch.arange(MOTION_INTERVALS, device=local.device)[None, :, None]
            group = torch.arange(2, device=local.device)[None, None, :]
            tokens = (self.motion_proj(motion) + self.motion_interval(interval)
                      + self.motion_group(group))
            invalid = self.motion_invalid[None, None, None].expand_as(tokens)
            tokens = torch.where(motion_valid[..., None], tokens, invalid)
            context = tokens.mean(dim=(1, 2))
        flow = self.head(local[:, :, None] + self.time.weight[None, None] + context[:, None, None]).float() * .01
        return flow
