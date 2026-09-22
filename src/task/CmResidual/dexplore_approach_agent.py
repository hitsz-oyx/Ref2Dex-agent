"""Random-initialized DExplore agent with shared approach shaping for both arms."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from learning.dexplore_agent import DexploreAgent

from src.task.CmResidual.dexplore_approach import (
    ApproachConfig, potential_approach_reward, sampled_surface_gap,
)
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge
from src.task.CmResidual.dexplore_contact_curriculum import annealed_curriculum_scale
from src.task.CmResidual.dexplore_contact_curriculum import reverse_curriculum_progress
from src.task.CmResidual.dexplore_grasp_reward import held_lift_reward


class DExploreApproachAgent(DexploreAgent):
    """Apply identical geometry shaping to a Cm-off or Cm-on scratch actor."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.approach_reward_coef = float(os.environ.get("REF2DEX_APPROACH_REWARD_COEF", "0"))
        self.held_lift_reward_coef = float(os.environ.get("REF2DEX_HELD_LIFT_REWARD_COEF", "0"))
        if self.approach_reward_coef < 0 or not torch.isfinite(torch.tensor(self.approach_reward_coef)):
            raise ValueError("REF2DEX_APPROACH_REWARD_COEF must be finite and nonnegative")
        if self.held_lift_reward_coef < 0 or not torch.isfinite(torch.tensor(self.held_lift_reward_coef)):
            raise ValueError("REF2DEX_HELD_LIFT_REWARD_COEF must be finite and nonnegative")
        self.approach_config = ApproachConfig()
        self.approach_bridge = None
        if self.approach_reward_coef:
            asset_root = Path(__file__).resolve().parents[3] / "third_party/DExplore/dexplore/data/assets"
            self.approach_bridge = DExploreCmv2GeometryBridge(
                hand_urdf=asset_root / "inspire_hand_new/inspire_hand_right.urdf",
                object_urdf=asset_root / "mjcf/airplane.urdf", device=self.ppo_device, seed=42)
        self._approach_calls = self._approach_samples = self._approach_contacts = 0
        self._approach_object_contacts = self._approach_positive = 0
        self._approach_sum = self._approach_abs_sum = self._approach_gap_before = self._approach_gap_after = 0.0
        self._approach_positive_object_dz = 0.0
        self._held_lift_sum = self._held_lift_positive = 0.0
        self._approach_min_gap = float("inf")
        self.curriculum_anneal_start = int(os.environ.get(
            "REF2DEX_CURRICULUM_ANNEAL_START", "-1"))
        self.curriculum_anneal_end = int(os.environ.get(
            "REF2DEX_CURRICULUM_ANNEAL_END", "-1"))
        self.curriculum_backtrack_start = int(os.environ.get(
            "REF2DEX_CURRICULUM_BACKTRACK_START", "-1"))
        self.curriculum_backtrack_end = int(os.environ.get(
            "REF2DEX_CURRICULUM_BACKTRACK_END", "-1"))
        if ((self.curriculum_anneal_start < 0) != (self.curriculum_anneal_end < 0) or
                (self.curriculum_anneal_start >= 0 and
                 self.curriculum_anneal_end <= self.curriculum_anneal_start)):
            raise ValueError("invalid curriculum annealing environment")
        if ((self.curriculum_backtrack_start < 0) != (self.curriculum_backtrack_end < 0) or
                (self.curriculum_backtrack_start >= 0 and
                 self.curriculum_backtrack_end <= self.curriculum_backtrack_start)):
            raise ValueError("invalid curriculum backtrack environment")

    def train_epoch(self):
        if self.curriculum_backtrack_start >= 0:
            progress = reverse_curriculum_progress(
                self.epoch_num, self.curriculum_backtrack_start,
                self.curriculum_backtrack_end)
            self._cm_task()._ref2dex_contact_backtrack_progress = progress
            if (self.epoch_num in (self.curriculum_backtrack_start,
                                   self.curriculum_backtrack_end) or
                    self.epoch_num % 10 == 0):
                print("REF2DEX_CURRICULUM_BACKTRACK " + json.dumps({
                    "epoch": self.epoch_num, "progress": progress,
                    "start": self.curriculum_backtrack_start,
                    "end": self.curriculum_backtrack_end,
                }, sort_keys=True), flush=True)
        if self.curriculum_anneal_start >= 0:
            scale = annealed_curriculum_scale(
                self.epoch_num, self.curriculum_anneal_start,
                self.curriculum_anneal_end)
            self._cm_task()._ref2dex_contact_curriculum_scale = scale
            if (self.epoch_num in (self.curriculum_anneal_start,
                                   self.curriculum_anneal_end) or
                    self.epoch_num % 10 == 0):
                print("REF2DEX_CURRICULUM_ANNEAL " + json.dumps({
                    "epoch": self.epoch_num, "scale": scale,
                    "start": self.curriculum_anneal_start,
                    "end": self.curriculum_anneal_end,
                }, sort_keys=True), flush=True)
        return super().train_epoch()

    def _cm_task(self):
        required = ("_dof_pos", "_target_states", "hoi_data", "data_id", "progress_buf",
                    "dof_limits_lower", "dof_limits_upper")
        pending = [self.vec_env]
        visited, chain = set(), []
        while pending and len(visited) < 8:
            candidate = pending.pop(0)
            if candidate is None or id(candidate) in visited:
                continue
            visited.add(id(candidate))
            chain.append(type(candidate).__name__)
            if all(hasattr(candidate, name) for name in required):
                return candidate
            pending.extend(getattr(candidate, name, None) for name in ("env", "task"))
        raise RuntimeError(f"Approach/Cm reward cannot locate DExplore task through {chain}")

    @torch.inference_mode()
    def _approach_gap(self, task):
        geometry = self.approach_bridge.current(task._dof_pos, task._target_states)
        return sampled_surface_gap(geometry.hand_points, geometry.object_points, self.approach_config)

    def env_step(self, actions):
        if not self.approach_reward_coef and not self.held_lift_reward_coef:
            return super().env_step(actions)
        task = self._cm_task()
        gap_before = getattr(self, "_ref2dex_cached_gap_before", None)
        if hasattr(self, "_ref2dex_cached_gap_before"):
            del self._ref2dex_cached_gap_before
        if self.approach_reward_coef and gap_before is None:
            gap_before = self._approach_gap(task)
        object_z_before = task._target_states[:, 2].clone()
        obs, rewards, dones, infos = super().env_step(actions)
        contact = (task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > 0.1).any(dim=-1)
        object_contact = task._tar_contact_forces.norm(dim=-1) > 0.1
        if self.approach_reward_coef:
            gap_after = self._approach_gap(task)
            shaped = potential_approach_reward(
                gap_before, gap_after, dones.bool(), gamma=self.gamma, config=self.approach_config)
            bonus = self.approach_reward_coef * shaped
            rewards = rewards + bonus.view(-1, 1) if rewards.ndim == 2 else rewards + bonus
        else:
            gap_after = torch.zeros_like(object_z_before)
            shaped = torch.zeros_like(object_z_before)
        rest_z = task.hoi_refs[task.data_id, task.ref_index, 0, 108]
        held_lift = held_lift_reward(task._target_states[:, 2], rest_z, contact, object_contact)
        held_bonus = self.held_lift_reward_coef * held_lift
        rewards = rewards + held_bonus.view(-1, 1) if rewards.ndim == 2 else rewards + held_bonus
        self._approach_calls += 1
        self._approach_samples += shaped.numel()
        self._approach_positive += int((shaped > 0).sum())
        self._approach_sum += float(shaped.sum())
        self._approach_abs_sum += float(shaped.abs().sum())
        self._approach_gap_before += float(gap_before.sum())
        self._approach_gap_after += float(gap_after.sum())
        self._approach_min_gap = min(self._approach_min_gap, float(gap_after.min()))
        self._approach_positive_object_dz += float((task._target_states[:, 2] - object_z_before).clamp_min(0).sum())
        self._held_lift_sum += float(held_lift.sum())
        self._held_lift_positive += int((held_lift > 0).sum())
        self._approach_contacts += int(contact.sum())
        self._approach_object_contacts += int((contact & object_contact).sum())
        if self._approach_calls % self.horizon_length == 0:
            n = self._approach_samples
            print("REF2DEX_APPROACH_REWARD " + json.dumps({
                "rank": int(getattr(self, "rank", 0)), "call": self._approach_calls,
                "coef": self.approach_reward_coef, "samples": n,
                "mean": self._approach_sum / n, "abs_mean": self._approach_abs_sum / n,
                "positive_fraction": self._approach_positive / n,
                "gap_before_mean_m": self._approach_gap_before / n,
                "gap_after_mean_m": self._approach_gap_after / n,
                "min_gap_m": self._approach_min_gap,
                "hand_contact_fraction": self._approach_contacts / n,
                "hand_and_object_contact_fraction": self._approach_object_contacts / n,
                "positive_object_dz_mean_m": self._approach_positive_object_dz / n,
                "held_lift_reward_coef": self.held_lift_reward_coef,
                "held_lift_mean": self._held_lift_sum / n,
                "held_lift_positive_fraction": self._held_lift_positive / n,
            }, sort_keys=True), flush=True)
            self._approach_samples = self._approach_contacts = self._approach_object_contacts = 0
            self._approach_positive = 0
            self._approach_sum = self._approach_abs_sum = self._approach_gap_before = self._approach_gap_after = 0.0
            self._approach_positive_object_dz = 0.0
            self._held_lift_sum = self._held_lift_positive = 0.0
            self._approach_min_gap = float("inf")
        return obs, rewards, dones, infos
