"""PointWorld's unified PTv3-small, adapted to masked hands and rigid effects."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from .data import collate
from .model import rotation6d, rigid_points, geodesic, mean_groups

VENDOR = Path(__file__).resolve().parents[5] / 'third_party/PointWorld'
sys.path.insert(0, str(VENDOR))
from ptv3.ptv3 import PointTransformerV3


# The anchor-frame corpus reaches slightly below -1m on validation windows.
# Keep a generous, fixed margin while retaining the explicit 16-bit PTv3
# workspace contract; never recenter this origin from a batch or action.
DEFAULT_VOXEL_ORIGIN_M = (-2.0, -2.0, -2.0)
DEFAULT_VOXEL_SIZE_M = 0.01
# The pinned PTv3 serializer encodes three coordinates with at most 16 bits
# each (48 bits total).  Keep this contract explicit instead of allowing an
# out-of-range coordinate to wrap in the Hilbert/Z-order implementation.
MAX_SERIALIZED_GRID_COORD = (1 << 16) - 1


def fixed_workspace_grid(coord, valid, origin=DEFAULT_VOXEL_ORIGIN_M,
                         voxel_size=DEFAULT_VOXEL_SIZE_M):
    """Map anchor-frame coordinates to a fixed, bounded integer workspace.

    The grid origin is deliberately independent of the batch and of future
    action points.  Invalid padding is ignored when checking the physical
    bounds, but valid non-finite or out-of-range points fail closed.
    """
    if coord.ndim != 3 or coord.shape[-1] != 3:
        raise ValueError('coord must have shape (batch, points, 3)')
    if valid.shape != coord.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('valid must be a boolean (batch, points) mask')
    if not np.isfinite(float(voxel_size)) or float(voxel_size) <= 0:
        raise ValueError('voxel_size must be positive and finite')
    origin = torch.as_tensor(origin, device=coord.device, dtype=torch.float32)
    if origin.shape != (3,) or not torch.isfinite(origin).all():
        raise ValueError('origin must be three finite coordinates')
    points = coord.float()
    selected = points[valid]
    if not selected.numel():
        raise ValueError('fixed workspace requires at least one valid point')
    if not torch.isfinite(selected).all():
        raise ValueError('valid point coordinates must be finite')
    grid = torch.floor((points - origin) / float(voxel_size)).long()
    used = grid[valid]
    bad = (used < 0).any(-1) | (used > MAX_SERIALIZED_GRID_COORD).any(-1)
    if bad.any():
        lo = used.min(0).values.tolist()
        hi = used.max(0).values.tolist()
        raise ValueError(
            'valid point outside fixed PTv3 workspace: '
            f'grid_min={lo}, grid_max={hi}, '
            f'allowed=[0,{MAX_SERIALIZED_GRID_COORD}]')
    return grid


def capped_collate(samples):
    """Cap encoder points without dropping any supervised object or its identity."""
    capped = []
    for original in samples:
        s = dict(original)
        ids = s['scene_object']
        hand = np.flatnonzero(ids < 0)
        objects = len(s['points'])
        quota = min(512, (4096 - len(hand)) // objects)
        if quota < 1:
            raise ValueError('scene has too many objects for point budget')
        take = [hand]
        for m in range(objects):
            pool = np.flatnonzero(ids == m)
            take.append(pool[np.linspace(0, len(pool)-1, quota, dtype=np.int64)])
        take = np.sort(np.concatenate(take))
        for key in ('xyz', 'features', 'scene_object'):
            s[key] = s[key][take]
        capped.append(s)
    out = collate(capped)
    out['scene_object'] = torch.full(out['point_valid'].shape, -1, dtype=torch.long)
    for i, s in enumerate(capped):
        out['scene_object'][i, :len(s['scene_object'])] = torch.from_numpy(s['scene_object'])
    return out


class PointWorldWM(nn.Module):
    def __init__(self, stats, patch_size=128, *,
                 voxel_origin_m=DEFAULT_VOXEL_ORIGIN_M,
                 voxel_size_m=DEFAULT_VOXEL_SIZE_M):
        super().__init__()
        self.voxel_size_m = float(voxel_size_m)
        # This is a fixed coordinate contract, not a learned/statistical
        # buffer.  Non-persistent keeps old checkpoints loadable for an
        # explicitly labelled pre/post-repair sensitivity probe.
        self.register_buffer('voxel_origin_m', torch.as_tensor(
            voxel_origin_m, dtype=torch.float32), persistent=False)
        d = 128
        cfg = yaml.safe_load((VENDOR / 'ptv3/ptv3_arch.yaml').read_text())['sizes']['small']
        channels = lambda key: tuple(d if x == 'channels' else x for x in cfg[key])
        self.backbone = PointTransformerV3(
            in_channels=d, order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
            stride=(2,)*4, enc_depths=tuple(cfg['enc_depths']),
            enc_channels=channels('enc_channels'), enc_num_head=tuple(cfg['enc_num_head']),
            enc_patch_size=(patch_size,)*5, dec_depths=tuple(cfg['dec_depths']),
            dec_channels=channels('dec_channels'), dec_num_head=tuple(cfg['dec_num_head']),
            dec_patch_size=(patch_size,)*4, drop_path=.3, shuffle_orders=True,
            enable_rpe=False, traceable=True, mask_token=False, enc_mode=False)
        self.scene_proj = nn.Sequential(nn.Linear(18, d), nn.GELU(), nn.Linear(d, d))
        self.action_proj = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.time = nn.Embedding(24, d)
        self.hand = nn.Embedding(2, d)
        self.keypoint = nn.Embedding(11, d)
        self.action_type = nn.Parameter(torch.zeros(d))
        self.skip_gamma = nn.Parameter(torch.ones(d))
        self.skip_beta = nn.Parameter(torch.zeros(d))
        self.hand_gamma = nn.Parameter(torch.ones(d))
        self.hand_beta = nn.Parameter(torch.zeros(d))
        self.object_proj = nn.Sequential(nn.Linear(15, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 9))
        nn.init.normal_(self.head[-1].weight, std=.01)
        nn.init.zeros_(self.head[-1].bias)
        for key in ('flow_mean', 'flow_std', 'translation_mean', 'translation_std',
                    'rotation_scale', 'scene_mean', 'scene_std', 'action_mean', 'action_std'):
            self.register_buffer(key, torch.tensor(stats[key], dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        # Fixed serialization orders at evaluation make action interventions
        # comparable without unrelated random order perturbations.
        for module in self.backbone.modules():
            if hasattr(module, 'shuffle_orders'):
                module.shuffle_orders = mode
        return self

    def forward(self, batch, arm='action'):
        xyz, valid = batch['xyz'], batch['point_valid']
        B, N = valid.shape
        M = batch['object_valid'].shape[1]
        scene = self.scene_proj((batch['features'] - self.scene_mean) / self.scene_std)
        bids = torch.arange(B, device=xyz.device)[:, None].expand_as(valid)
        if arm != 'history':
            a = batch['action'].reshape(B, 24, 2, 11, 9)
            av = batch['action_valid'].reshape(B, 528)
            # Zero invalid features BEFORE projection: arbitrary masked coordinates
            # cannot enter serialized geometry, the summary or normalization.
            a = torch.where(av.reshape(B, 24, 2, 11, 1), a, torch.zeros_like(a))
            af = (self.action_proj((a-self.action_mean)/self.action_std) +
                  self.time.weight[None, :, None, None] + self.hand.weight[None, None, :, None] +
                  self.keypoint.weight[None, None, None] + self.action_type).reshape(B, 528, 128)
            coord = torch.cat((xyz, a[..., :3].reshape(B, 528, 3)), 1)
            feat = torch.cat((scene, af), 1)
            exists = torch.cat((valid, av), 1)
        else:
            coord, feat, exists = xyz, scene, valid
        allb = torch.arange(B, device=xyz.device)[:, None].expand_as(exists)
        # PTv3 sparse convolutions require float32, as in the existing CUDA stem.
        with torch.autocast('cuda', enabled=False):
            coords, inputs = coord[exists].float(), feat[exists].float()
            grid_all = fixed_workspace_grid(coord, exists, self.voxel_origin_m,
                                            self.voxel_size_m)
            grid = grid_all[exists]
            keys = torch.cat((allb[exists][:, None], grid), -1)
            unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
            # SparseConv requires ONE row per (batch, voxel). Coincident scene
            # and future-hand points otherwise yield nondeterministic kernels.
            # Pool only for PTv3, then restore every original point's identity;
            # scene skip features and full rigid supervision stay unpooled.
            point = self.backbone(dict(coord=mean_groups(coords, inverse, len(unique)),
                                       feat=mean_groups(inputs, inverse, len(unique)),
                                       grid_coord=unique[:, 1:].int(), batch=unique[:, 0],
                                       grid_size=self.voxel_size_m))
        packed = point.feat.new_zeros((*exists.shape, 128))
        packed[exists] = point.feat[inverse]
        summary = packed.new_zeros((B, 128))
        if arm != 'history':
            hand = packed[:, N:].masked_fill(~av[..., None], -torch.inf).amax(1)
            summary = torch.where(av.any(1)[:, None], hand, torch.zeros_like(hand))
        local = packed[:, :N] + scene*self.skip_gamma + self.skip_beta
        local = local + (summary*self.hand_gamma + self.hand_beta)[:, None]
        obj = batch['scene_object']
        selected = valid & (obj >= 0)
        group = (bids*M + obj)[selected]
        pooled = mean_groups(local[selected], group, B*M).reshape(B, M, 128)
        query = pooled + self.object_proj(batch['object_features'])
        raw = self.head(query[:, :, None] + self.time.weight[None, None]).float()
        # Scaled outputs learn in normalized units; rotations remain rigid.
        translation = raw[..., :3]*self.translation_std + self.translation_mean
        identity6 = raw.new_tensor([1, 0, 0, 0, 1, 0])
        rotation = rotation6d(identity6 + raw[..., 3:])
        return dict(translation=translation, rotation=rotation)

    def loss(self, pred, batch):
        gt = batch['effect'].float()
        actual = rigid_points(gt[..., :3, :3], gt[..., :3, 3], batch['points'])
        predicted = rigid_points(pred['rotation'], pred['translation'], batch['points'])
        delta = torch.diff(torch.cat((batch['points'][:, :, None], actual), 2), dim=2)
        # Exact released PointWorld selector formula, tau=5mm/temp=5/gamma=1.
        weight = torch.sigmoid(5/.005*(torch.linalg.vector_norm(delta, dim=-1)-.005))
        weight = weight*batch['object_valid'][:, :, None, None]
        norm = weight.sum().clamp_min(1)
        target_flow = (actual-batch['points'][:, :, None]-self.flow_mean[None, None, :, None]) / self.flow_std[None, None, :, None]
        pred_flow = (predicted-batch['points'][:, :, None]-self.flow_mean[None, None, :, None]) / self.flow_std[None, None, :, None]
        point = (F.huber_loss(pred_flow, target_flow, reduction='none').mean(-1)*weight).sum()/norm
        ow = weight.mean(-1)
        onorm = ow.sum().clamp_min(1e-6)
        trans = F.huber_loss((pred['translation']-self.translation_mean)/self.translation_std,
                             (gt[..., :3, 3]-self.translation_mean)/self.translation_std,
                             reduction='none').mean(-1)
        angle = geodesic(pred['rotation'], gt[..., :3, :3])/self.rotation_scale
        rot = F.huber_loss(angle, torch.zeros_like(angle), reduction='none')
        terms = dict(point=point, translation=(trans*ow).sum()/onorm, rotation=(rot*ow).sum()/onorm)
        return sum(terms.values()), terms
