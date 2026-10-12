"""Shared WAM encoder and point loss for rigid sources and video tracks.

World predictions remain executed-action conditioned, serving Cm. Video labels
need point correspondence, not object poses. The native rigid head is optional.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .model import GoalPointWAM, HIDDEN, HORIZON, HANDS, KEYPOINTS
from oakink_wm.model import rigid_points


def masked_point_loss(predicted, target, valid, current, scale_m=.05,
                      motion_tau_m=.002, motion_floor=.1, confidence=None):
    """Normalize points within each valid timestep, then time within each window.

    Mask BEFORE arithmetic: invalid NaN teacher labels must not poison gradients.
    Motion weighting cannot select observations and is used only in the loss.
    """
    if predicted.shape != target.shape or valid.shape != target.shape[:-1]:
        raise ValueError('point trajectory/mask shape mismatch')
    if scale_m <= 0 or motion_tau_m <= 0 or not 0 < motion_floor <= 1:
        raise ValueError('invalid point loss scales')
    valid = valid.bool()
    if not torch.isfinite(target[valid]).all() or not torch.isfinite(predicted[valid]).all():
        raise ValueError('nonfinite valid point trajectory')
    safe_target = torch.where(valid[..., None], target.float(), 0.)
    safe_pred = torch.where(valid[..., None], predicted.float(), 0.)
    displacement = torch.linalg.vector_norm(safe_target - current.float()[:, None], dim=-1)
    motion = motion_floor + (1-motion_floor)*torch.sigmoid(5*(displacement/motion_tau_m-1))
    weight = motion * valid
    if confidence is not None:
        c = torch.where(valid, confidence.float(), 0.)
        if not torch.isfinite(c).all() or ((c < 0) | (c > 1)).any():
            raise ValueError('invalid point confidence')
        weight = weight * c
    error = F.huber_loss(safe_pred/scale_m, safe_target/scale_m, reduction='none').mean(-1)
    denominator = weight.sum(-1)
    per_time = (error*weight).sum(-1)/denominator.clamp_min(1e-8)
    time_valid = denominator > 0
    per_window = (per_time*time_valid).sum(-1)/time_valid.sum(-1).clamp_min(1)
    window_valid = time_valid.any(-1)
    if not window_valid.any():
        raise ValueError('batch has no supervised point trajectory')
    return per_window[window_valid].mean()


class PointTrajectoryDecoder(nn.Module):
    """Independent point queries attend to common scene/action memory.

    No N*24 query self-attention: chunking queries preserves point independence
    and limits attention memory. Time is decoded after spatial cross-attention.
    """
    def __init__(self, hidden=HIDDEN, chunk_size=256):
        super().__init__()
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError('positive query chunk required')
        self.position = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.query_norm = nn.LayerNorm(hidden)
        self.cross = nn.MultiheadAttention(hidden, 4, dropout=0, batch_first=True)
        self.context = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.time = nn.Embedding(HORIZON, hidden)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, query_xyz, query_features, query_valid, scene, scene_valid, action, action_valid):
        memory = torch.cat((scene, action), 1)
        memory_valid = torch.cat((scene_valid, action_valid), 1)
        outputs = []
        for lo in range(0, query_xyz.shape[1], self.chunk_size):
            xyz = query_xyz[:, lo:lo+self.chunk_size]
            q = query_features[:, lo:lo+self.chunk_size] + self.position(xyz.float()/.05)
            attended, _ = self.cross(self.query_norm(q), memory, memory,
                                      key_padding_mask=~memory_valid, need_weights=False)
            context = self.context(q + attended)
            time = context[:, None] + self.time.weight[None, :, None]
            # Each query retains its current point identity through all24 times.
            prediction = xyz.float()[:, None] + .05*self.head(time).float()
            mask = query_valid[:, None, lo:lo+self.chunk_size, None]
            outputs.append(torch.where(mask, prediction, 0.))
        return torch.cat(outputs, 2)


class PointSupervisedWAM(GoalPointWAM):
    """One actual backbone for every source; no video-specific backbone copy."""
    def __init__(self, stats, config):
        super().__init__(stats, config)
        if self.goal_type != 'object_points':
            raise ValueError('paired point goal required')
        self.rigid_native = bool(config.get('rigid_native_head', True))
        self.point_world = PointTrajectoryDecoder(chunk_size=config.get('point_query_chunk', 256))
        self.point_loss_scale = float(config.get('point_loss_scale_m', .05))

    def _native_queries(self, batch, scene):
        xyz = batch['points'].flatten(1, 2).float()
        valid = batch['object_valid'][:, :, None].expand_as(batch['points'][..., 0]).flatten(1, 2)
        # Canonical rigid queries are matched only to CURRENT scene coordinates.
        features = []
        for lo in range(0, xyz.shape[1], 256):
            distances = torch.cdist(xyz[:, lo:lo+256], scene['point_xyz'].float())
            distances = distances.masked_fill(~scene['point_mask'][:, None], float('inf'))
            index = distances.argmin(-1)
            features.append(torch.gather(scene['point_features'], 1, index[..., None].expand(-1, -1, HIDDEN)))
        return xyz, torch.cat(features, 1), valid

    def forward(self, batch):
        scene = self.scene(batch, return_points=True)
        goal, goal_mask = self.goal(batch['start_object_points'], batch['goal_object_points'], batch['object_goal_valid'])
        trajectory = self.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, goal_mask,
                                     self.observed_hand_condition(batch))
        action, action_mask = self.action(batch['action'].reshape(-1, HORIZON, HANDS, KEYPOINTS, 9),
                                         batch['action_valid'].reshape(-1, HORIZON, HANDS, KEYPOINTS))
        video = bool(batch.get('video_points', False))
        if not video and self.rigid_native:
            rigid = self.world(scene['scene_tokens'], scene['scene_mask'], scene['object_tokens'], scene['object_mask'],
                               action, action_mask)
            points = rigid_points(rigid['rotation'], rigid['translation'], batch['points'].float())
            return dict(trajectory=trajectory, world_points=points.permute(0, 2, 1, 3, 4).flatten(2, 3), **rigid)
        if video:
            # Loader appends measured hand points after these tracked scene points.
            n = batch['query_xyz'].shape[1]
            xyz, features, valid = batch['query_xyz'], scene['point_features'][:, :n], batch['query_valid']
        else:
            xyz, features, valid = self._native_queries(batch, scene)
        points = self.point_world(xyz, features, valid, scene['scene_tokens'], scene['scene_mask'], action, action_mask)
        return dict(trajectory=trajectory, world_points=points)

    def loss(self, prediction, batch):
        if bool(batch.get('video_points', False)):
            actual, valid, current = batch['point_target'], batch['point_target_valid'], batch['query_xyz']
            confidence = batch.get('point_confidence')
        else:
            gt = batch['effect'].float()
            actual = rigid_points(gt[..., :3, :3], gt[..., :3, 3], batch['points'].float())
            actual = actual.permute(0, 2, 1, 3, 4).flatten(2, 3)
            current = batch['points'].flatten(1, 2)
            valid = batch['object_valid'][:, None, :, None].expand(-1, HORIZON, -1, batch['points'].shape[2]).flatten(2, 3)
            confidence = None
        point = masked_point_loss(prediction['world_points'], actual, valid, current,
                                  self.point_loss_scale, self.motion_tau_m, self.motion_floor, confidence)
        # Native and video both supervise hand POINT trajectories with the same loss.
        if batch['trajectory_valid'].any():
            actor = self.trajectory_loss(prediction['trajectory'], batch)
        else:
            actor = prediction['trajectory'].sum()*0
        return actor + self.world_weight*point, dict(trajectory=actor.detach(), world_point=point.detach())

    def transfer_wam(self, state):
        """Explicit weight transfer into NEW model identity; never optimizer resume."""
        if state.get('checkpoint_kind') != 'goal-point-wam.single.v1' or state.get('artificial_labels'):
            raise ValueError('real WAM parent required')
        result = self.load_state_dict(state['model'], strict=False)
        if result.unexpected_keys or any(not k.startswith('point_world.') for k in result.missing_keys):
            raise ValueError('parent architecture mismatch: ' + str(result))
        return dict(missing_new_head_keys=result.missing_keys, optimizer_transferred=False)
