"""Causal, held motion targets for independent native action learning."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class NoReference:
    """Compatibility slots only: no checkpoint, actor, features, or demonstration."""
    def __init__(self):
        self.model = nn.Module()
        self.invalid = set()

    def act(self, snapshot, observation, tick):
        return observation.new_zeros((len(observation), 24))


def bounded_delta(goal, center, limit=.05):
    delta = goal - center
    return delta * (limit / delta.norm(dim=-1, keepdim=True).clamp_min(limit))


def canonical_errors(target, current, yaw):
    """World targets stay fixed; express errors in the native action yaw frame."""
    error = target - current[:, None]
    c, s = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
    x, y, z = error.unbind(-1)
    return torch.stack((c*x+s*y, -s*x+c*y, z), -1)


class MotionReference:
    def __init__(self, kind, checkpoint, objects, cache):
        from shadow_finetune import paths
        from video_data_preparation.unidexgrasp_data import object_poses
        self.object_poses = object_poses
        self.kind = kind
        self.model = None
        if kind == 'motion-wam':
            from .wam_features import FeatureController
            controller = FeatureController(checkpoint, objects, cache)
            if controller.schema != 'ref2dex.ptv3-shared-wam.v3':
                raise ValueError('Motion decoder requires V3')
            self.model = controller.model.forecaster
            self.geometry = controller.geometry
        else:
            meta = json.loads((Path(cache) / 'manifest.json').read_text())
            lookup = {row['object_code']: row['geometry'] for row in meta['records']}
            self.geometry = []
            for row in objects:
                with np.load(lookup[row['object_code']]) as d:
                    length = np.linalg.norm(d['normals'], axis=-1)
                    keep = length > 1e-6
                    self.geometry.append(dict(surface=d['surface'][keep], normals=d['normals'][keep]/length[keep, None]))
        self.query = torch.as_tensor(np.stack([g['surface'][np.linspace(0, len(g['surface'])-1, 32, dtype=int)]
                                              for g in self.geometry]), device='cuda', dtype=torch.float32)
        self.history = []
        self.contract = dict(kind=kind, refresh_native_steps=16, forecast_native_steps=[16, 32],
                             goal_native_steps=48, goal_translation_cap_m=.05, goal_rotation='current',
                             short_goal='task-directed capped displacement, not certified reachable',
                             coordinates='fixed world targets; current native yaw-frame errors / .05m clipped to +/-10',
                             history='observed [-6,-4,-2,0]; initial missing samples repeat earliest',
                             reference_width=259, task_reward_only=True)

    @torch.inference_mode()
    def observe(self, snapshot, tick, yaw):
        if tick == 1:
            self.history = []
        self.history.append(snapshot)
        self.history = self.history[-7:]
        pose = torch.as_tensor(self.object_poses(snapshot['object']), device='cuda', dtype=torch.float32)
        rotation, center = pose[:, :3, :3], pose[:, :3, 3]
        hand = torch.as_tensor(snapshot['hand'], device='cuda', dtype=torch.float32)
        scene = self.query @ rotation.transpose(-1, -2) + center[:, None]
        if (tick-1) % 16 == 0:
            self.reference_tick = tick
            goal = torch.as_tensor(snapshot['goal'], device='cuda', dtype=torch.float32)
            self.delta = bounded_delta(goal, center)
            factor = torch.tensor([16/48, 32/48], device='cuda')[None, :, None, None]
            hand_delta = self.delta[:, None, None].expand(-1, 2, 11, -1) * factor
            scene_delta = self.delta[:, None, None].expand(-1, 2, 32, -1) * factor
            if self.model is not None:
                from .fast_bc import batched_inputs
                from .task_readout import observed_batch, forecast_from_encoded
                past = [max(len(self.history)-1+offset, 0) for offset in (-6, -4, -2, 0)]
                objects = np.stack([self.history[i]['object'] for i in past], 1)
                hands = np.stack([self.history[i]['hand'] for i in past], 1)
                samples = batched_inputs(objects, hands, (center+self.delta).cpu().numpy(), self.geometry)
                batch = observed_batch(samples, max(len(g['surface']) for g in self.geometry))
                shared, valid, geometry = self.model.encode(batch)
                hd, sd = forecast_from_encoded(self.model, batch, shared, valid, geometry['tokens'].shape[1])
                # Decoder outputs displacements in the refresh-time object frame.
                hand_delta = hd[:, [7, 15], 0] @ rotation[:, None].transpose(-1, -2)
                scene_delta = sd[:, 0, [7, 15]] @ rotation[:, None].transpose(-1, -2)
            self.hand_target = hand[:, None] + hand_delta
            self.scene_target = scene[:, None] + scene_delta
        h = canonical_errors(self.hand_target, hand, yaw).flatten(1)
        s = canonical_errors(self.scene_target, scene, yaw).flatten(1)
        errors = torch.cat((h, s), -1)
        if not torch.isfinite(errors).all():
            raise FloatingPointError('Nonfinite held motion reference')
        self.unclipped_max_m = float(errors.abs().max())
        self.clipped_fraction = float((errors.abs() > .5).float().mean())
        if self.kind == 'motion-none':
            errors = torch.zeros_like(errors)
        age = errors.new_full((len(errors), 1), (tick-self.reference_tick)/48)
        return torch.cat(((errors/.05).clamp(-10, 10), age), -1).detach()

    def audit(self, output, tick, observation):
        # Only first-episode refreshes are retained by the caller.
        torch.save(dict(tick=tick, reference_tick=self.reference_tick, delta_world=self.delta.cpu(),
                        hand_target_world=self.hand_target.cpu(), scene_target_world=self.scene_target.cpu(),
                        observation=observation.cpu(), unclipped_max_m=self.unclipped_max_m,
                        clipped_fraction=self.clipped_fraction, contract=self.contract),
                   Path(output) / ('motion-input-%03d.pt' % tick))
