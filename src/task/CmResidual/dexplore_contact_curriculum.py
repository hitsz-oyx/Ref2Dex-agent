"""Optional near-contact reference resets shared by both scratch PPO arms.

The existing DExplore Start initializer samples frames *before* the first
20 cm approach.  This training curriculum samples around the first recorded
object-contact frame instead.  It changes only episode initialization, never
the policy weights, reference reward, or test-time starting state.
"""
from __future__ import annotations

import json

import torch


def annealed_curriculum_scale(epoch: int, start: int, end: int) -> float:
    if epoch < 0 or start < 0 or end <= start:
        raise ValueError("invalid curriculum annealing schedule")
    if epoch <= start:
        return 1.0
    if epoch >= end:
        return 0.0
    return float(end - epoch) / float(end - start)


def install_contact_reset_curriculum(task_class, *, before: int, after: int,
                                     fraction: float = 1.0, lift_fraction: float = 0.0,
                                     lift_threshold_m: float = 0.03) -> None:
    """Patch a process-local DExplore task class before task construction."""
    if (before < 0 or after < 0 or fraction < 0 or lift_fraction < 0 or
            fraction + lift_fraction <= 0 or fraction + lift_fraction > 1.0 or
            lift_threshold_m <= 0):
        raise ValueError("contact curriculum windows must be nonnegative")
    if getattr(task_class, "_ref2dex_contact_curriculum_installed", False):
        raise RuntimeError("contact reset curriculum is already installed")
    original = task_class._reset_ref_state_init

    def reset_near_contact(self, env_ids):
        original(self, env_ids)
        if not len(env_ids):
            return
        if not hasattr(self, "_ref2dex_first_contact_frames"):
            anchors, lift_anchors = [], []
            for motion in self.hoi_data_dict:
                frames = (motion["contact"].reshape(-1) > 0.5).nonzero(as_tuple=True)[0]
                if frames.numel() == 0:
                    raise ValueError("contact curriculum requires recorded object contact in every motion")
                contact_anchor = int(frames[0].item())
                anchors.append(contact_anchor)
                z = motion["obj_pos"][:, 2]
                post_contact = z[contact_anchor:]
                lifted = (post_contact - post_contact.cummin(0).values >= lift_threshold_m).nonzero(
                    as_tuple=True)[0]
                if lift_fraction and lifted.numel() == 0:
                    raise ValueError("lift curriculum requires a reference lift after contact")
                lift_anchors.append(contact_anchor + int(lifted[0].item()) if lifted.numel()
                                    else contact_anchor)
            self._ref2dex_first_contact_frames = torch.tensor(
                anchors, device=self.device, dtype=torch.long)
            self._ref2dex_first_lift_frames = torch.tensor(
                lift_anchors, device=self.device, dtype=torch.long)
            print("REF2DEX_CONTACT_CURRICULUM " + json.dumps({
                "first_contact_frames": anchors, "window_before": before,
                "window_after": after, "fraction": fraction,
                "first_lift_frames": lift_anchors, "lift_fraction": lift_fraction,
                "lift_threshold_m": lift_threshold_m,
                "mode": "mixed_start_and_contact_training_reset",
            }, sort_keys=True), flush=True)

        scale = float(getattr(self, "_ref2dex_contact_curriculum_scale", 1.0))
        if not 0 <= scale <= 1:
            raise ValueError("contact curriculum scale must be in [0,1]")
        effective_contact = fraction * scale
        effective_lift = lift_fraction * scale
        draw = torch.rand(len(env_ids), device=self.device)
        contact_mask = draw < effective_contact
        lift_mask = ((draw >= effective_contact) &
                     (draw < effective_contact + effective_lift))
        selected_mask = contact_mask | lift_mask
        selected_envs = env_ids[selected_mask]
        if not len(selected_envs):
            return
        motion = self.data_id[selected_envs].long()
        anchors = self._ref2dex_first_contact_frames[motion].clone()
        selected_lift = lift_mask[selected_mask]
        anchors[selected_lift] = self._ref2dex_first_lift_frames[motion[selected_lift]]
        offset = torch.randint(-before, after + 1, (len(selected_envs),), device=self.device)
        last = self.max_episode_length[motion].to(self.device) - 2
        times = torch.minimum((anchors + offset).clamp_min(0), last)
        ref = self.hoi_refs[motion, self.ref_index[selected_envs], times]
        self.progress_buf[selected_envs] = times
        self.start_times[selected_envs] = times
        self._hist_obs[selected_envs] = 0
        self.contact_reset[selected_envs] = 0
        self._set_env_state(env_ids=selected_envs,
                            dof_pos=ref[:, 119:119 + self.num_dof],
                            dof_vel=ref[:, 119 + self.num_dof:119 + 2 * self.num_dof])

    task_class._reset_ref_state_init = reset_near_contact
    task_class._ref2dex_contact_curriculum_installed = True
