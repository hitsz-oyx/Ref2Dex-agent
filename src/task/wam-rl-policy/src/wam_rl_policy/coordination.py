"""Bounded post-lift coordination bonus, measured against actual next-state geometry."""
import torch
from .motion_reference import MotionReference


def rigid_fit(source, target):
    """Row points: target = source @ rotation.T + translation, proper SE(3)."""
    a, b = source.mean(-2), target.mean(-2)
    x, y = source-a[:, None], target-b[:, None]
    u, singular, vh = torch.linalg.svd(x.transpose(-1, -2) @ y)
    correction = torch.ones_like(singular)
    correction[:, -1] = torch.linalg.det(u @ vh)
    rotation = (u @ torch.diag_embed(correction) @ vh).transpose(-1, -2)
    translation = b - (a[:, None] @ rotation.transpose(-1, -2)).squeeze(1)
    residual = ((source @ rotation.transpose(-1, -2) + translation[:, None])-target).norm(dim=-1).mean(-1)
    return rotation, translation, residual


def relative_hand(hand, rotation, translation):
    return (hand-translation[:, None]) @ rotation


def coordination_score(current, target, enabled):
    """A bonus in [0,1], never a prediction-as-success reward."""
    mse = (current-target).square().sum(-1).mean(-1)
    return torch.exp(-mse/(2*.03**2)) * enabled.float()


def lifted_nearby(hand, center, initial_z):
    return ((center[:, 2]-initial_z >= .03)
            & ((hand[:, 6:]-center[:, None]).norm(dim=-1).mean(-1) <= .15)
            & ((hand[:, 0]-center).norm(dim=-1) <= .15))


class CoordinationReference(MotionReference):
    def __init__(self, kind, checkpoint, objects, cache):
        super().__init__('motion-wam' if kind == 'wam' else 'motion-geometry', checkpoint, objects, cache)
        self.coordination_kind = kind
        self.contract.update(kind='coordination-'+kind, reference_width=35,
            coordinates='hand11 in actual current object frame minus held desired relative hand11 / .03m',
            target='16-native-step hand prediction in rigid-fit predicted object frame',
            eligibility='refresh and actual scored state: lift>=3cm, mean tip-center<=15cm, palm-center<=15cm',
            admission='rigid scene residual<=1cm; max relative hand target change<=8cm',
            score='eligible * exp(-mean hand point squared distance/(2*.03^2))', task_reward_only=False)

    def actual_relative(self, snapshot):
        pose = torch.as_tensor(self.object_poses(snapshot['object']), device='cuda', dtype=torch.float32)
        hand = torch.as_tensor(snapshot['hand'], device='cuda', dtype=torch.float32)
        return relative_hand(hand, pose[:, :3, :3], pose[:, :3, 3]), hand, pose

    @torch.inference_mode()
    def observe(self, snapshot, tick, yaw, initial_z):
        self.initial_z = initial_z
        super().observe(snapshot, tick, yaw)
        current, hand, pose = self.actual_relative(snapshot)
        if (tick-1) % 16 == 0:
            rotation, translation, residual = rigid_fit(self.query, self.scene_target[:, 0])
            target = relative_hand(self.hand_target[:, 0], rotation, translation)
            change = (target-current).norm(dim=-1).amax(-1)
            valid = torch.isfinite(target).all(dim=(-1, -2)) & (residual <= .01) & (change <= .08)
            center = pose[:, :3, 3]
            self.phase = lifted_nearby(hand, center, initial_z)
            self.admitted = valid
            self.enabled = self.phase & valid
            # Invalid targets supply neither bonus nor an extreme observation.
            self.relative_target = torch.where(valid[:, None, None], target, current)
            self.rigid_residual = residual
            self.target_change = change
        error = (self.relative_target-current).flatten(1)
        age = error.new_full((len(error), 1), (tick-self.reference_tick)/48)
        observed_enabled = self.enabled & lifted_nearby(hand, pose[:, :3, 3], initial_z)
        return torch.cat(((error/.03).clamp(-5, 5), age, observed_enabled[:, None].float()), -1).detach()

    @torch.inference_mode()
    def score(self, next_snapshot):
        current, hand, pose = self.actual_relative(next_snapshot)
        self.reward_enabled = self.enabled & lifted_nearby(hand, pose[:, :3, 3], self.initial_z)
        return coordination_score(current, self.relative_target, self.reward_enabled)

    def audit(self, output, tick, observation):
        from pathlib import Path
        torch.save(dict(tick=tick, reference_tick=self.reference_tick, relative_target=self.relative_target.cpu(),
                        enabled=self.enabled.cpu(), phase=self.phase.cpu(), admitted=self.admitted.cpu(),
                        rigid_residual=self.rigid_residual.cpu(), target_change=self.target_change.cpu(),
                        delta_world=self.delta.cpu(), observation=observation.cpu(), contract=self.contract),
                   Path(output)/('coordination-input-%03d.pt' % tick))
