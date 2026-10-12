"""Isolated geometric adaptations of ref1_2 methods; active DDP source stays frozen."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_three_domain import GoalPointWAM


def action_from_delta(delta, start, valid):
    """Dataset's 9D action is XYZ/current displacement/step displacement, NOT v/a."""
    previous = torch.cat((torch.zeros_like(delta[:, :1]), delta[:, :-1]), 1)
    action = torch.cat((start[:, None] + delta, delta, delta - previous), -1)
    # A missing preceding sample cannot provide a valid step displacement.
    previous_valid = torch.cat((valid[:, :1], valid[:, :-1]), 1)
    mask = valid & previous_valid
    return torch.where(mask[..., None], action, 0.), mask


def achieved_prefix(batch, horizons):
    """Achieved state at k labels only the demonstrated prefix through k."""
    out = dict(batch)
    rows = torch.arange(len(horizons), device=horizons.device)
    effect = batch['effect'][rows, batch['goal_object_slot'], horizons - 1]
    start = batch['start_object_points']
    out['goal_object_points'] = torch.einsum('bij,bnj->bni', effect[:, :3, :3], start) + effect[:, None, :3, 3]
    times = torch.arange(24, device=horizons.device)[None, :, None, None]
    out['trajectory_valid'] = batch['trajectory_valid'] & (times < horizons[:, None, None, None])
    out['goal_horizon'] = horizons
    return out


def goal_points_from_world(world, batch):
    rows = torch.arange(len(batch['goal_object_slot']), device=batch['goal_object_slot'].device)
    slots = batch['goal_object_slot']
    rotation = world['rotation'][rows, slots, -1]
    translation = world['translation'][rows, slots, -1]
    return torch.einsum('bij,bnj->bni', rotation, batch['start_object_points']) + translation[:, None]


def target_roi_weights(weight, slots, multiplier=4.):
    objects = torch.arange(weight.shape[1], device=weight.device)[None]
    boost = torch.where(objects == slots[:, None], multiplier, 1.)
    return weight * boost[:, :, None, None]


class Ref12Method(GoalPointWAM):
    def __init__(self, stats, config):
        super().__init__(stats, config)
        self.method = config['method']
        self.goal_weight = float(config.get('goal_consistency_weight', .1))
        self._supervised_batch = None
        if self.method == 'hindsight_prefix':
            self.horizon_condition = nn.Embedding(3, 128)
            nn.init.zeros_(self.horizon_condition.weight)
        if self.method == 'change_supervision':
            self.future_change = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 3))
        if self.method == 'intermediate_plan':
            layer = nn.TransformerDecoderLayer(128, 4, 512, dropout=0., activation='gelu', batch_first=True, norm_first=True)
            self.plan_decoder = nn.TransformerDecoder(layer, 1)
            self.plan_head = nn.Linear(128, 9)
            nn.init.zeros_(self.plan_head.weight); nn.init.zeros_(self.plan_head.bias)
        if self.method == 'world_roi':
            for module in (self.goal, self.trajectory, self.hand_query):
                if module is not None:
                    module.requires_grad_(False)
        if self.method == 'frozen_goal_consistency':
            for module in (self.scene, self.action, self.world):
                module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.method == 'frozen_goal_consistency':
            # Frozen world includes buffers, dropout and sparse serialization order.
            for module in (self.scene, self.action, self.world):
                module.eval()
        return self

    def observed_hand_condition(self, batch):
        condition = super().observed_hand_condition(batch)
        if self.method == 'hindsight_prefix':
            horizons = batch.get('goal_horizon', torch.full((len(condition),), 24, device=condition.device))
            indices = horizons // 8 - 1
            condition = condition + self.horizon_condition(indices)[:, None, None]
        return condition

    def motion_weights(self, batch, actual):
        weight = super().motion_weights(batch, actual)
        if self.method == 'world_roi':
            weight = target_roi_weights(weight, batch['goal_object_slot'])
        return weight

    def forward(self, batch):
        if self.training and self.method == 'hindsight_prefix':
            horizons = 8 * (torch.randint(3, (len(batch['start_hand']),), device=batch['start_hand'].device) + 1)
            batch = achieved_prefix(batch, horizons)
        self._supervised_batch = batch
        scene = self.scene(batch)
        action, mask = self.action(batch['action'].reshape(-1, 24, 2, 11, 9), batch['action_valid'].reshape(-1, 24, 2, 11))
        world = self.world(scene['scene_tokens'], scene['scene_mask'], scene['object_tokens'], scene['object_mask'], action, mask)
        out = dict(translation=world['translation'], rotation=world['rotation'])
        if self.method == 'world_roi':
            out['trajectory'] = torch.zeros_like(batch['trajectory_target'])
        else:
            out.update(self.actor_from_scene(scene, batch))
        if self.method == 'frozen_goal_consistency' and self.training:
            executed, valid = action_from_delta(out['trajectory'], batch['start_hand'], batch['trajectory_valid'])
            policy_action, policy_mask = self.action(executed, valid)
            # Frozen parameters still propagate gradients with respect to actor actions.
            induced = self.world(scene['scene_tokens'], scene['scene_mask'], scene['object_tokens'], scene['object_mask'], policy_action, policy_mask)
            out['policy_goal_points'] = goal_points_from_world(induced, batch)
        if self.method == 'change_supervision':
            token = scene['object_tokens'][:, :, None] + self.world.time.weight[None, None]
            out['future_point_change'] = self.future_change(token).float()
        return out

    def actor_from_scene(self, scene, batch):
        out = {}
        goal, goal_mask = self.goal(batch['start_object_points'], batch['goal_object_points'], batch['object_goal_valid'])
        actor_scene, actor_mask = scene['scene_tokens'], scene['scene_mask']
        if self.method == 'intermediate_plan':
            from oakink_wm.model import rotation6d
            rows = torch.arange(len(goal), device=goal.device)
            objects = scene['object_tokens'][rows, batch['goal_object_slot']]
            query = objects[:, None] + self.world.time.weight[None]
            memory = torch.cat((actor_scene, goal), 1)
            mask = torch.cat((actor_mask, goal_mask), 1)
            planned = self.plan_decoder(query, memory, memory_key_padding_mask=~mask)
            raw = self.plan_head(planned).float()
            identity = raw.new_tensor([1., 0., 0., 0., 1., 0.])
            rotation = rotation6d(identity + raw[..., 3:])
            out['planned_object_points'] = torch.einsum('btij,bnj->btni', rotation, batch['start_object_points']) + .05 * raw[..., None, :3]
            # Predicted features only. No GT intermediate pose enters the actor.
            actor_scene = torch.cat((actor_scene, planned), 1)
            actor_mask = torch.cat((actor_mask, torch.ones(planned.shape[:2], device=goal.device, dtype=torch.bool)), 1)
        out['trajectory'] = self.trajectory(actor_scene, actor_mask, goal, goal_mask, self.observed_hand_condition(batch))
        return out

    def loss(self, prediction, batch):
        batch = self._supervised_batch
        world, world_terms = self.world_loss(prediction, batch)
        terms = {'world': world.detach(), **{'world_' + key: value.detach() for key, value in world_terms.items()}}
        if self.method == 'world_roi':
            return world, terms
        trajectory = self.trajectory_loss(prediction['trajectory'], batch)
        terms['trajectory'] = trajectory.detach()
        if self.method == 'frozen_goal_consistency':
            errors = F.huber_loss(prediction['policy_goal_points'] / .05, batch['goal_object_points'] / .05, reduction='none').mean(-1)
            mask = batch['object_goal_valid']
            goal = (errors * mask).sum() / mask.sum().clamp_min(1)
            terms['goal_consistency'] = goal.detach()
            return trajectory + self.goal_weight * goal, terms
        total = trajectory + world
        if self.method == 'intermediate_plan':
            rows = torch.arange(len(batch['goal_object_slot']), device=batch['goal_object_slot'].device)
            effect = batch['effect'][rows, batch['goal_object_slot']]
            target = torch.einsum('btij,bnj->btni', effect[..., :3, :3], batch['start_object_points']) + effect[..., None, :3, 3]
            error = F.huber_loss(prediction['planned_object_points'] / .05, target / .05, reduction='none').mean(-1)
            mask = batch['object_goal_valid'][:, None].expand_as(error)
            plan = (error * mask).sum() / mask.sum().clamp_min(1)
            terms['intermediate_plan'] = plan.detach()
            total = total + .5 * plan
        if self.method == 'change_supervision':
            from oakink_wm.model import rigid_points
            effect = batch['effect']
            actual = rigid_points(effect[..., :3, :3], effect[..., :3, 3], batch['points'])
            centers = actual.mean(-2)
            previous = torch.cat((batch['points'].mean(-2)[:, :, None], centers[:, :, :-1]), 2)
            target = centers - previous
            errors = F.huber_loss(prediction['future_point_change'] / .01, target / .01, reduction='none').mean(-1)
            valid = batch['object_valid'][:, :, None].expand_as(errors)
            change = (errors * valid).sum() / valid.sum().clamp_min(1)
            terms['future_change'] = change.detach()
            total = total + .25 * change
        return total, terms
