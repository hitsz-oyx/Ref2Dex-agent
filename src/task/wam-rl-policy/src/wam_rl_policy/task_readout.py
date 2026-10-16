"""Task queries over frozen interaction tokens, supervised by real transitions.

No action cloning or candidate-action dynamics is implemented here. Forecasts
are goal-conditioned priors, not counterfactual outcomes of the policy action.
"""
import numpy as np
import torch
from torch import nn


def observed_batch(samples, point_budget, device='cuda'):
    """Mask ragged CAD after normal filtering; never admit target fields."""
    keys = ('start_hand', 'start_hand_history', 'start_hand_valid', 'object_valid',
            'scene_points', 'start_object_points', 'goal_object_delta', 'object_goal_valid')
    batch = {key: torch.as_tensor(np.stack([s[key] for s in samples]), device=device) for key in keys}
    batch.update(xyz=torch.zeros(len(samples),point_budget,3,device=device),
        features=torch.zeros(len(samples),point_budget,18,device=device),
        point_valid=torch.zeros(len(samples),point_budget,dtype=torch.bool,device=device))
    for i,sample in enumerate(samples):
        n=len(sample['xyz'])
        if n>point_budget:raise ValueError('CAD point budget exceeded')
        batch['xyz'][i,:n]=torch.as_tensor(sample['xyz'],device=device)
        batch['features'][i,:n]=torch.as_tensor(sample['features'],device=device)
        batch['point_valid'][i,:n]=True
    return batch


def object_split(objects):
    if not objects or any(row['split'] != 'train' for row in objects):
        raise ValueError('Only existing TRAIN objects may enter this qualification')
    codes = sorted({row['object_code'] for row in objects})
    held = codes[::4]
    fit = np.array([row['object_code'] not in held for row in objects])
    if not fit.any() or fit.all():
        raise ValueError('Nonempty disjoint object groups required')
    return fit, held


def finite_returns(reward, reset, gamma=.99):
    if reset.shape != (len(reward) + 1, reward.shape[1]):
        raise ValueError('Reset flags must be state-aligned')
    if reset[1:-1].any() or not reset[-1].all():
        raise ValueError('Complete episodes without interior resets required')
    result = np.empty_like(reward, dtype=np.float32)
    running = np.zeros(reward.shape[1], np.float32)
    for t in reversed(range(len(reward))):
        running = reward[t] + gamma * (~reset[t + 1].astype(bool)) * running
        result[t] = running
    return result


def transition_targets(data, tick, rotations, horizons=(16, 32)):
    """Labels only: observed future displacement, goal distance, center gap."""
    if tick + max(horizons) >= len(data['object']):
        raise ValueError('Complete observed future required')
    if data['reset'][tick + 1:tick + max(horizons) + 1].any():
        raise ValueError('Future targets cannot cross a reset')
    now = data['object'][tick, :, :3]
    goal = data['goal'][tick]  # known decision goal, not future goal
    result = []
    for h in horizons:
        future = data['object'][tick + h, :, :3]
        delta = np.einsum('bi,bij->bj', future - now, rotations)
        distance = np.linalg.norm(future - goal, axis=-1, keepdims=True)
        gap = np.linalg.norm(data['hand'][tick + h] - future[:, None], axis=-1).min(-1, keepdims=True)
        result.append(np.concatenate((delta, distance, gap), -1))
    return np.concatenate(result, -1).astype(np.float32)


def raw_tokens(batch):
    """Same observed information as the WAM encoder, before pretraining.

    xyz3 + field21; scene uses all18 features, hand uses9 history fields,
    goal uses3 deltas; last3 entries identify scene/hand/goal, never source IDs.
    """
    scene = torch.cat((batch['xyz'], batch['features'],
                       batch['xyz'].new_zeros((*batch['xyz'].shape[:2], 3))), -1)
    scene[..., 21] = 1
    hand = scene.new_zeros((len(scene), 22, 24))
    hand[..., :3] = batch['start_hand'].flatten(1, 2)
    hand[..., 3:12] = batch['start_hand_history'].flatten(1, 2)
    hand[..., 12] = torch.arange(2, device=hand.device).repeat_interleave(11)
    hand[..., 13] = torch.arange(11, device=hand.device).repeat(2) / 10
    hand[..., 22] = 1
    goal = scene.new_zeros((len(scene), batch['start_object_points'].shape[1], 24))
    goal[..., :3] = batch['start_object_points']
    goal[..., 3:6] = batch['goal_object_delta']
    goal[..., 23] = 1
    valid = torch.cat((batch['point_valid'], batch['start_hand_valid'].flatten(1),
                       batch['object_goal_valid']), 1).bool()
    tokens = torch.cat((scene, hand, goal), 1)
    return torch.where(valid[..., None], tokens, 0), valid


def forecast_from_encoded(model, batch, shared, valid, scene_count):
    """Reuse the unchanged V3 decoder, avoiding a second encoder forward."""
    hand_z = model.hand_future(shared[:, scene_count:scene_count + 22]).reshape(len(shared), 24, 2, 11, 3)
    points = batch['scene_points']
    point_valid = batch['object_valid'][..., None].expand(*points.shape[:-1])
    clean_points = torch.where(point_valid[..., None], points, 0)
    query = model.scene_query(clean_points / model.position_scale).flatten(1, 2)
    context = model.scene_context(shared)
    update, _ = model.scene_readout(query, context, context, key_padding_mask=~valid, need_weights=False)
    scene_z = model.scene_future(model.scene_norm(query + update))
    scene_z = scene_z.reshape(len(points), 24, *points.shape[1:3], 3).transpose(1, 2)
    hand = torch.where(batch['start_hand_valid'][:, None, ..., None],
                       model.normalizer.denormalize('hand', hand_z), 0)
    scene = torch.where(point_valid[:, :, None, :, None],
                        model.normalizer.denormalize('scene', scene_z), 0)
    return hand, scene


class TaskReadout(nn.Module):
    """Small state/goal-conditioned readout suitable for actor/critic adapters."""
    def __init__(self, state_width, token_width, *, mean=False, forecast_width=0, width=128):
        super().__init__()
        self.mean = mean
        self.state = nn.Sequential(nn.Linear(state_width, width), nn.ELU(), nn.Linear(width, width))
        self.tokens = nn.Sequential(nn.Linear(token_width, width), nn.LayerNorm(width))
        if not mean:
            self.attention = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
        self.fusion = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.ELU())
        self.forecast = nn.Linear(forecast_width, width, bias=False) if forecast_width else None
        self.value = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, 1))
        self.physics = nn.Sequential(nn.Linear(width, width), nn.ELU(), nn.Linear(width, 10))

    def encode(self, state, tokens, valid=None, forecast=None):
        query = self.state(state)
        context = self.tokens(tokens)
        if self.mean:
            read = context
        else:
            if valid is None or not valid.any(-1).all():
                raise ValueError('At least one valid token per state required')
            read, _ = self.attention(query[:, None], context, context,
                                     key_padding_mask=~valid, need_weights=False)
            read = read[:, 0]
        z = self.fusion(torch.cat((query, read), -1))
        if self.forecast is not None:
            if forecast is None:
                raise ValueError('Declared forecast features are required')
            z = z + self.forecast(forecast)
        return z

    def forward(self, state, tokens, valid=None, forecast=None):
        z = self.encode(state, tokens, valid, forecast)
        return self.value(z).squeeze(-1), self.physics(z)
