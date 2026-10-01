"""Predetermined training curriculum and phase-only dense reward."""
from __future__ import annotations
import types
from collections import OrderedDict
import torch
from src.task.CmResidual.dexplore_grasp_reward import held_lift_reward


def canonical_model_state(state):
    result = OrderedDict()
    for key,value in state.items():
        name = key[len('_orig_mod.'):] if key.startswith('_orig_mod.') else key
        if name in result:
            raise ValueError('ambiguous compiled checkpoint key')
        result[name] = value
    return result


def plateau_probability(epoch):
    if epoch < 0:
        raise ValueError('negative epoch')
    if epoch <= 80:
        return .5
    if epoch >= 200:
        return 0.
    return .5 * (200 - epoch) / 120


def phase_lift_bonus(height, rest, contact, progress, start, stop):
    held = held_lift_reward(height, rest, contact[:,0].bool(), contact[:,1].bool())
    return 10 * held * ((progress >= start) & (progress <= stop)).float()


def install_training_reset(task, phase_start):
    """Change training resets only, before native target/root tensor reset."""
    original = task._reset_ref_state_init
    task.hold_reset_counts = torch.zeros((3,2), dtype=torch.long, device=task._dof_pos.device)
    task.hold_plateau_probability = .5

    def reset(self, env_ids):
        self._hybrid_init_prob = 1.
        original(env_ids)
        if not len(env_ids):
            return
        if self.start_times[env_ids].any():
            raise ValueError('non-curriculum reset must be frame0')
        selected = torch.rand(len(env_ids), device=self._dof_pos.device) < self.hold_plateau_probability
        motion = self.data_id[env_ids].long()
        self.hold_reset_counts.view(-1).index_add_(0,motion*2+selected.long(),torch.ones_like(motion))
        ids = env_ids[selected]
        if not len(ids):
            return
        times = phase_start[self.data_id[ids]]
        self.progress_buf[ids] = times
        self.start_times[ids] = times
        self._hist_obs[ids] = 0
        self.contact_reset[ids] = 0
        ref = self.hoi_refs[self.data_id[ids], self.ref_index[ids], times]
        self._set_env_state(env_ids=ids, dof_pos=ref[:,119:137], dof_vel=ref[:,137:155])
        if ref[:,113:119].any() or ref[:,137:155].any():
            raise ValueError('plateau training reset velocity nonzero')

    task._reset_ref_state_init = types.MethodType(reset, task)
