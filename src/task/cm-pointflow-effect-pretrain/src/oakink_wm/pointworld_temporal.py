"""Time-preserving PointWorld adapter; the live V1 adapter stays frozen.

Attention serializes by physical sample and space. Sparse CPE uses a virtual
batch per (sample, time), so equal spatial coordinates never create duplicate
spconv indices. All encoder pooling levels retain the integer time identity.
"""
import torch
import torch_scatter
import spconv.pytorch as spconv
from torch import nn
import torch.nn.functional as F

from .pointworld import (DEFAULT_VOXEL_ORIGIN_M, DEFAULT_VOXEL_SIZE_M,
                          PointWorldWM, capped_collate, fixed_workspace_grid, VENDOR)
from .model import mean_groups, rotation6d, rigid_points, geodesic
from ptv3.structure import Point
from ptv3.ptv3 import GridPooling
from ptv3.module import PointModule


class TemporalPoint(Point):
    def sparsify(self, pad=96):
        shape = self.get('sparse_shape', (self.grid_coord.max(0).values + pad).tolist())
        virtual_batch = self.batch * 25 + self.time_id
        self.sparse_shape = shape
        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat((virtual_batch[:, None], self.grid_coord), 1).int().contiguous(),
            spatial_shape=shape, batch_size=int(virtual_batch.max()) + 1)


class TemporalGridPooling(PointModule):
    """Upstream spatial pooling, with time included in the cluster identity."""
    def __init__(self, original):
        super().__init__()
        for attr in ('stride', 'reduce', 'shuffle_orders', 'traceable'):
            setattr(self, attr, getattr(original, attr))
        self.proj = original.proj
        self.norm, self.act = getattr(original, 'norm', None), getattr(original, 'act', None)

    def forward(self, point, skip_postprocess=False):
        grid = torch.div(point.grid_coord, self.stride, rounding_mode='trunc')
        keys = torch.cat((point.batch[:, None], point.time_id[:, None], grid), 1)
        unique, cluster, counts = torch.unique(keys, dim=0, sorted=True,
                                               return_inverse=True, return_counts=True)
        indices = torch.argsort(cluster)
        ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        child = TemporalPoint(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], ptr, reduce='mean'),
            grid_coord=unique[:, 2:].int(), batch=unique[:, 0], time_id=unique[:, 1])
        for attr in ('condition', 'context', 'name', 'split'):
            if attr in point:
                child[attr] = point[attr]
        for attr in ('origin_coord', 'color'):
            if attr in point:
                child[attr] = torch_scatter.segment_csr(point[attr][indices], ptr, reduce='mean')
        if 'grid_size' in point:
            child.grid_size = point.grid_size * self.stride
        if self.traceable:
            child.pooling_inverse, child.pooling_parent = cluster, point
        if skip_postprocess:
            return child
        if self.norm is not None:
            child = self.norm(child)
        if self.act is not None:
            child = self.act(child)
        child.serialization(order=point.order, shuffle_orders=self.shuffle_orders)
        child.sparsify()
        return child


class TemporalBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.core = backbone
        self._replace_pooling(backbone)

    @classmethod
    def _replace_pooling(cls, module):
        for name, child in list(module.named_children()):
            if isinstance(child, GridPooling):
                setattr(module, name, TemporalGridPooling(child))
            else:
                cls._replace_pooling(child)

    def forward(self, data):
        point = self.core.embedding(TemporalPoint(data))
        point.serialization(order=self.core.order, shuffle_orders=self.core.shuffle_orders)
        point.sparsify()
        point = self.core.enc(point)
        if not self.core.enc_mode:
            point = self.core.dec(point)
        return point


class TemporalPointWorldWM(PointWorldWM):
    def __init__(self, stats, patch_size=128, *, motion_weighting='cumulative_effect',
                 motion_tau_m=.002, rotation_tau_rad=.02, motion_temperature=5., motion_floor=.1,
                 voxel_origin_m=DEFAULT_VOXEL_ORIGIN_M,
                 voxel_size_m=DEFAULT_VOXEL_SIZE_M):
        if motion_weighting not in ('cumulative_effect', 'released_incremental'):
            raise ValueError('unknown motion weighting')
        if motion_tau_m <= 0 or rotation_tau_rad <= 0 or motion_temperature <= 0 or not 0 < motion_floor <= 1:
            raise ValueError('invalid motion selector parameters')
        super().__init__(stats, patch_size, voxel_origin_m=voxel_origin_m,
                         voxel_size_m=voxel_size_m)
        self.backbone = TemporalBackbone(self.backbone)
        self.motion_weighting = motion_weighting
        self.motion_tau_m, self.rotation_tau_rad = motion_tau_m, rotation_tau_rad
        self.motion_temperature, self.motion_floor = motion_temperature, motion_floor

    def forward(self, batch, arm='action'):
        xyz, valid = batch['xyz'], batch['point_valid']
        B, N = valid.shape
        M = batch['object_valid'].shape[1]
        scene = self.scene_proj((batch['features'] - self.scene_mean) / self.scene_std)
        times = torch.zeros_like(valid, dtype=torch.long)
        if arm != 'history':
            a = batch['action'].reshape(B, 24, 2, 11, 9)
            av = batch['action_valid'].reshape(B, 528)
            a = torch.where(av.reshape(B, 24, 2, 11, 1), a, torch.zeros_like(a))
            af = (self.action_proj((a-self.action_mean)/self.action_std) +
                  self.time.weight[None, :, None, None] + self.hand.weight[None, None, :, None] +
                  self.keypoint.weight[None, None, None] + self.action_type).reshape(B, 528, 128)
            coord = torch.cat((xyz, a[..., :3].reshape(B, 528, 3)), 1)
            feat = torch.cat((scene, af), 1)
            exists = torch.cat((valid, av), 1)
            action_times = torch.arange(1, 25, device=xyz.device).repeat_interleave(22)
            times = torch.cat((times, action_times[None].expand(B, -1)), 1)
        else:
            coord, feat, exists = xyz, scene, valid
        allb = torch.arange(B, device=xyz.device)[:, None].expand_as(exists)
        with torch.autocast('cuda', enabled=False):
            coords, inputs = coord[exists].float(), feat[exists].float()
            grid_all = fixed_workspace_grid(coord, exists, self.voxel_origin_m,
                                            self.voxel_size_m)
            grid = grid_all[exists]
            keys = torch.cat((allb[exists][:, None], times[exists][:, None], grid), -1)
            unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
            point = self.backbone(dict(coord=mean_groups(coords, inverse, len(unique)),
                                       feat=mean_groups(inputs, inverse, len(unique)),
                                       grid_coord=unique[:, 2:].int(), batch=unique[:, 0],
                                       time_id=unique[:, 1], grid_size=self.voxel_size_m))
        packed = point.feat.new_zeros((*exists.shape, 128))
        packed[exists] = point.feat[inverse]
        summary = packed.new_zeros((B, 24, 128))
        if arm != 'history':
            hand_valid = av.reshape(B, 24, 22)
            hand = packed[:, N:].reshape(B, 24, 22, 128)
            hand = hand.masked_fill(~hand_valid[..., None], -torch.inf).amax(2)
            summary = torch.where(hand_valid.any(2)[..., None], hand, torch.zeros_like(hand))
        local = packed[:, :N] + scene*self.skip_gamma + self.skip_beta
        obj = batch['scene_object']
        selected = valid & (obj >= 0)
        bids = torch.arange(B, device=xyz.device)[:, None].expand_as(valid)
        group = (bids*M + obj)[selected]
        pooled = mean_groups(local[selected], group, B*M).reshape(B, M, 128)
        query = pooled + self.object_proj(batch['object_features'])
        action_query = summary*self.hand_gamma + self.hand_beta
        raw = self.head(query[:, :, None] + self.time.weight[None, None] + action_query[:, None]).float()
        translation = raw[..., :3]*self.translation_std + self.translation_mean
        rotation = rotation6d(raw.new_tensor([1, 0, 0, 0, 1, 0]) + raw[..., 3:])
        return dict(translation=translation, rotation=rotation)

    def motion_weights(self, batch, actual=None):
        gt = batch['effect'].float()
        if actual is None:
            actual = rigid_points(gt[..., :3, :3], gt[..., :3, 3], batch['points'])
        if self.motion_weighting == 'released_incremental':
            delta = torch.diff(torch.cat((batch['points'][:, :, None], actual), 2), dim=2)
            weight = torch.sigmoid(1000*(torch.linalg.vector_norm(delta, dim=-1)-.005))
        else:
            displacement = torch.linalg.vector_norm(actual-batch['points'][:, :, None], dim=-1)
            identity = torch.eye(3, device=gt.device, dtype=gt.dtype).expand_as(gt[..., :3, :3])
            angle = geodesic(identity, gt[..., :3, :3])
            magnitude = torch.maximum(displacement/self.motion_tau_m,
                                      angle[..., None]/self.rotation_tau_rad)
            weight = self.motion_floor + (1-self.motion_floor)*torch.sigmoid(
                self.motion_temperature*(magnitude-1))
        return weight * batch['object_valid'][:, :, None, None]

    def loss(self, pred, batch):
        gt = batch['effect'].float()
        actual = rigid_points(gt[..., :3, :3], gt[..., :3, 3], batch['points'])
        predicted = rigid_points(pred['rotation'], pred['translation'], batch['points'])
        weight = self.motion_weights(batch, actual)
        target_flow = (actual-batch['points'][:, :, None]-self.flow_mean[None, None, :, None]) / self.flow_std[None, None, :, None]
        pred_flow = (predicted-batch['points'][:, :, None]-self.flow_mean[None, None, :, None]) / self.flow_std[None, None, :, None]
        floor = 1 if self.motion_weighting == 'released_incremental' else 1e-6
        point = (F.huber_loss(pred_flow, target_flow, reduction='none').mean(-1)*weight).sum()/weight.sum().clamp_min(floor)
        ow = weight.mean(-1)
        onorm = ow.sum().clamp_min(1e-6)
        trans = F.huber_loss((pred['translation']-self.translation_mean)/self.translation_std,
                             (gt[..., :3, 3]-self.translation_mean)/self.translation_std,
                             reduction='none').mean(-1)
        angle = geodesic(pred['rotation'], gt[..., :3, :3])/self.rotation_scale
        rot = F.huber_loss(angle, torch.zeros_like(angle), reduction='none')
        terms = dict(point=point, translation=(trans*ow).sum()/onorm, rotation=(rot*ow).sum()/onorm)
        return sum(terms.values()), terms


def model_from_config(stats, config):
    if config.get('schema') != 'pointworld-small-wm24.temporal.v1':
        raise ValueError('temporal entry requires a temporal configuration')
    if config.get('temporal_pooling') is not True or config.get('action_summary') != 'per_horizon':
        raise ValueError('temporal pooling and per-horizon summary must be enabled')
    return TemporalPointWorldWM(stats, config['patch_size'],
        motion_weighting=config['motion_weighting'], motion_tau_m=config['motion_tau_m'],
        rotation_tau_rad=config['rotation_tau_rad'], motion_temperature=config['motion_temperature'],
        motion_floor=config['motion_floor'],
        voxel_origin_m=config.get('voxel_origin_m', DEFAULT_VOXEL_ORIGIN_M),
        voxel_size_m=config.get('voxel_m', DEFAULT_VOXEL_SIZE_M))
