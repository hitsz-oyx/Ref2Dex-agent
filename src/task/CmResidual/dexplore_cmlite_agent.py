"""DExplore PPO agent using the low-latency frozen CmLite reward."""
from __future__ import annotations

import json
import os

import torch

from src.task.CmResidual.cmlite import (
    FrozenCmLite, contact_gate, goal_reward, proximity_trust,
)
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent


class DExploreCmLiteAgent(DExploreApproachAgent):
    """Shape scratch-policy rewards with a 49D-input one-step model."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.cmlite_reward_coef = float(os.environ["REF2DEX_CMLITE_REWARD_COEF"])
        if self.cmlite_reward_coef <= 0:
            raise ValueError("REF2DEX_CMLITE_REWARD_COEF must be positive")
        self.cmlite = FrozenCmLite(
            os.environ["REF2DEX_CMLITE_CHECKPOINT"], self.ppo_device,
            os.environ["REF2DEX_CMLITE_SHA256"])
        self.cmlite_positive_only = os.environ.get(
            "REF2DEX_CMLITE_REWARD_POSITIVE_ONLY", "1") == "1"
        # A hard contact gate makes the world-model reward disappear before the
        # policy has learned to touch the object.  Keep the old behavior as the
        # default, but allow scratch training to use the model's predicted
        # contact probability as a dense exploration signal.
        self.cmlite_use_predicted_contact = os.environ.get(
            "REF2DEX_CMLITE_USE_PREDICTED_CONTACT", "0") == "1"
        self.cmlite_max_gap_m = float(os.environ.get(
            "REF2DEX_CMLITE_MAX_GAP_M", "inf"))
        if self.cmlite_max_gap_m <= 0:
            raise ValueError("REF2DEX_CMLITE_MAX_GAP_M must be positive")
        self._cmlite_calls = self._cmlite_samples = self._cmlite_positive = 0
        self._cmlite_actual_contact = self._cmlite_trusted = 0
        self._cmlite_sum = self._cmlite_contact_sum = self._cmlite_delta_sum = 0.0

    @torch.inference_mode()
    def _predict_reward(self, actions):
        task = self._cm_task()
        prediction = self.cmlite.predict(task._dof_pos, actions.detach(), task._target_states)
        hand_contact = (task._contact_forces[:, task._contact_body_ids].norm(
            dim=-1) > 0.1).any(dim=-1)
        object_contact = task._tar_contact_forces.norm(dim=-1) > 0.1
        actual_contact = hand_contact & object_contact
        progress = (task.progress_buf + 1).clamp_max(task.hoi_data.shape[1] - 1)
        goal_position = task.hoi_data[task.data_id, progress, 106:109]
        reward_gate = contact_gate(
            prediction["contact_probability"], actual_contact,
            use_predicted_contact=self.cmlite_use_predicted_contact)
        trusted = torch.ones_like(reward_gate)
        if self.cmlite_use_predicted_contact and self.cmlite_max_gap_m < float("inf"):
            if self.approach_bridge is None:
                raise ValueError("proximity-trusted CmLite requires approach geometry")
            gap = self._approach_gap(task)
            self._ref2dex_cached_gap_before = gap
            trusted = proximity_trust(gap, self.cmlite_max_gap_m)
            reward_gate = reward_gate * trusted
        reward = goal_reward(
            task._target_states[:, :3], goal_position, prediction["delta_world"],
            reward_gate,
            positive_only=self.cmlite_positive_only)
        return reward, prediction, actual_contact, trusted

    def env_step(self, actions):
        reward, prediction, actual_contact, trusted = self._predict_reward(actions)
        obs, rewards, dones, infos = super().env_step(actions)
        bonus = self.cmlite_reward_coef * reward
        rewards = rewards + bonus.unsqueeze(-1) if rewards.ndim == 2 else rewards + bonus
        self._cmlite_calls += 1
        self._cmlite_samples += reward.numel()
        self._cmlite_positive += int((reward > 0).sum())
        self._cmlite_actual_contact += int(actual_contact.sum())
        self._cmlite_trusted += int(trusted.sum())
        self._cmlite_sum += float(reward.sum())
        self._cmlite_contact_sum += float(prediction["contact_probability"].sum())
        self._cmlite_delta_sum += float(prediction["delta_world"].norm(dim=-1).sum())
        if self._cmlite_calls % self.horizon_length == 0:
            count = self._cmlite_samples
            print("REF2DEX_CMLITE_REWARD " + json.dumps({
                "rank": int(getattr(self, "rank", 0)), "call": self._cmlite_calls,
                "coef": self.cmlite_reward_coef, "samples": count,
                "mean": self._cmlite_sum / count,
                "positive_fraction": self._cmlite_positive / count,
                "actual_contact_gate_fraction": self._cmlite_actual_contact / count,
                "geometry_trust_fraction": self._cmlite_trusted / count,
                "max_trusted_gap_m": self.cmlite_max_gap_m,
                "predicted_contact_mean": self._cmlite_contact_sum / count,
                "predicted_delta_norm_mean_m": self._cmlite_delta_sum / count,
            }, sort_keys=True), flush=True)
            self._cmlite_samples = self._cmlite_positive = self._cmlite_actual_contact = 0
            self._cmlite_trusted = 0
            self._cmlite_sum = self._cmlite_contact_sum = self._cmlite_delta_sum = 0.0
        return obs, rewards, dones, infos
