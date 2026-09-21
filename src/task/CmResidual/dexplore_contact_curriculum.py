"""Optional near-contact reference resets shared by both scratch PPO arms.

The existing DExplore Start initializer samples frames *before* the first
20 cm approach.  This training curriculum samples around the first recorded
object-contact frame instead.  It changes only episode initialization, never
the policy weights, reference reward, or test-time starting state.
"""
from __future__ import annotations

import json

import torch


def install_contact_reset_curriculum(task_class, *, before: int, after: int,
                                     fraction: float = 1.0) -> None:
    """Patch a process-local DExplore task class before task construction."""
    if before < 0 or after < 0 or not (0.0 < fraction <= 1.0):
        raise ValueError("contact curriculum windows must be nonnegative")
    if getattr(task_class, "_ref2dex_contact_curriculum_installed", False):
        raise RuntimeError("contact reset curriculum is already installed")
    original = task_class._reset_ref_state_init

    def reset_near_contact(self, env_ids):
        original(self, env_ids)
        if not len(env_ids):
            return
        if not hasattr(self, "_ref2dex_first_contact_frames"):
            anchors = []
            for motion in self.hoi_data_dict:
                frames = (motion["contact"].reshape(-1) > 0.5).nonzero(as_tuple=True)[0]
                if frames.numel() == 0:
                    raise ValueError("contact curriculum requires recorded object contact in every motion")
                anchors.append(int(frames[0].item()))
            self._ref2dex_first_contact_frames = torch.tensor(
                anchors, device=self.device, dtype=torch.long)
            print("REF2DEX_CONTACT_CURRICULUM " + json.dumps({
                "first_contact_frames": anchors, "window_before": before,
                "window_after": after, "fraction": fraction,
                "mode": "mixed_start_and_contact_training_reset",
            }, sort_keys=True), flush=True)

        selected_envs = env_ids
        if fraction < 1.0:
            selected_envs = env_ids[torch.rand(len(env_ids), device=self.device) < fraction]
        if not len(selected_envs):
            return
        motion = self.data_id[selected_envs].long()
        offset = torch.randint(-before, after + 1, (len(selected_envs),), device=self.device)
        last = self.max_episode_length[motion].to(self.device) - 2
        times = torch.minimum((self._ref2dex_first_contact_frames[motion] + offset).clamp_min(0), last)
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
