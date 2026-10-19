"""Serialized PTv3 surface fusion and joint21 readback; no nearest-neighbour op."""
import torch
from torch import nn
from . import paths
from ptv3_shared_wam_v2.geometry import JointGeometry
from ptv3_shared_wam.model import clean
from oakink_wm.model import mean_groups
from oakink_wm.pointworld import fixed_workspace_grid


class DenseJointGeometry(JointGeometry):
    def __init__(self, config, backbone=None):
        super().__init__(config, backbone=backbone)
        self.surface_encoder = nn.Sequential(nn.Linear(12, self.width), nn.GELU(), nn.Linear(self.width, self.width))
        self.surface_part = nn.Embedding(16, self.width)
        self.surface_side = nn.Embedding(2, self.width)
        scales = [config['position_scale_m']]*3 + [config['velocity_scale_mps']]*3
        scales += [config['acceleration_scale_mps2']]*3 + [config['position_scale_m']]*3
        self.register_buffer('surface_scale', torch.tensor(scales))

    def forward(self, batch, hand_encoding, hand_valid):
        sv = batch['point_valid'].bool()
        if not sv.any(1).all():
            raise ValueError('observed object geometry required in every sample')
        sx = clean(batch['xyz'], sv)
        sf = self.scene_proj(clean(batch['features'], sv)/self.feature_scale)
        jx = clean(batch['start_hand'], batch['start_hand_valid']).flatten(1, 2)
        dv = batch['dense_hand_valid'].bool()
        dx = clean(batch['dense_hand'], dv)
        values = torch.cat((dx, clean(batch['dense_history'], dv)), -1)
        part = torch.where(dv, batch['dense_hand_part'].long(), 0)
        df = self.surface_encoder(values/self.surface_scale) + self.surface_part(part)
        df = df + self.surface_side.weight[None, :, None]
        xyz = torch.cat((sx, jx, dx.flatten(1, 2)), 1)
        valid = torch.cat((sv, hand_valid, dv.flatten(1)), 1)
        features = torch.cat((sf, hand_encoding, df.flatten(1, 2)), 1)
        count, joints = sv.shape[1], hand_encoding.shape[1]
        is_hand = torch.zeros_like(valid)
        is_hand[:, count:] = True
        ids = torch.arange(len(xyz), device=xyz.device)[:, None].expand_as(valid)
        with torch.autocast('cuda' if xyz.is_cuda else 'cpu', enabled=False):
            grid = fixed_workspace_grid(xyz, valid, self.voxel_origin_m, self.voxel_size_m)
            unique, inverse = torch.unique(torch.cat((ids[valid][:, None], grid[valid]), -1),
                                           dim=0, return_inverse=True)
            hr = is_hand[valid]
            sm = mean_groups(features.float()[valid][~hr], inverse[~hr], len(unique))
            hm = mean_groups(features.float()[valid][hr], inverse[hr], len(unique))
            encoded = self.backbone(dict(coord=mean_groups(xyz[valid], inverse, len(unique)),
                feat=self.voxel_fusion(torch.cat((sm, hm), -1)), grid_coord=unique[:, 1:].int(),
                batch=unique[:, 0], grid_size=self.voxel_size_m))
            packed = encoded.feat.new_zeros((*valid.shape, self.width))
            packed[valid] = encoded.feat[inverse]
        scene, mask = self._sample_tokens(torch.cat((packed[:, :count], sx), -1), sv)
        return dict(tokens=scene[..., :self.width], xyz=scene[..., self.width:], mask=mask,
                    hand=packed[:, count:count+joints], voxel_count=len(unique))
