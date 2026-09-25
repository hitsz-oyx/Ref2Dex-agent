"""DExplore PPO agent with history-conditioned Cm credit."""
from __future__ import annotations

import json
import os

import torch

from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent
from src.task.CmResidual.temporal_cm import (
    FrozenTemporalHistoryCm,
    TemporalHistoryBuffer,
    temporal_step_features,
)


class DExploreTemporalCmAgent(DExploreApproachAgent):
    """Use predicted multi-step supported lift as a dense credit signal.

    The reward is deliberately based on the temporal supported-lift head and
    does not multiply by predicted contact probability.  This keeps the probe
    distinct from the stopped predicted-contact gate route.
    """

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.temporal_reward_coef = float(os.environ["REF2DEX_TEMPORAL_CM_REWARD_COEF"])
        self.temporal_reward_scale_mm = float(os.environ.get(
            "REF2DEX_TEMPORAL_CM_REWARD_SCALE_MM", "20"))
        self.temporal_positive_only = os.environ.get(
            "REF2DEX_TEMPORAL_CM_POSITIVE_ONLY", "1") == "1"
        if not torch.isfinite(torch.tensor(self.temporal_reward_coef)) or self.temporal_reward_coef <= 0:
            raise ValueError("REF2DEX_TEMPORAL_CM_REWARD_COEF must be finite and positive")
        if not torch.isfinite(torch.tensor(self.temporal_reward_scale_mm)) or self.temporal_reward_scale_mm <= 0:
            raise ValueError("REF2DEX_TEMPORAL_CM_REWARD_SCALE_MM must be finite and positive")
        self.temporal_cm = FrozenTemporalHistoryCm(
            os.environ["REF2DEX_TEMPORAL_CM_CHECKPOINT"], self.ppo_device,
            os.environ["REF2DEX_TEMPORAL_CM_SHA256"])
        self.temporal_history = None
        self._temporal_calls = 0
        self._temporal_samples = 0
        self._temporal_positive = 0
        self._temporal_reward_sum = 0.0
        self._temporal_lift_sum = 0.0
        self._temporal_contact_sum = 0.0

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
        if self.temporal_history is not None:
            self.temporal_history.reset(env_ids)
        return obs

    def _ensure_history(self, count: int, device: torch.device) -> None:
        if self.temporal_history is None:
            self.temporal_history = TemporalHistoryBuffer(
                count, self.temporal_cm.history_len, device)
        elif len(self.temporal_history.history) != count:
            raise RuntimeError("temporal Cm environment count changed during training")

    @torch.inference_mode()
    def _predict_reward(self, actions: torch.Tensor):
        task = self._cm_task()
        self._ensure_history(actions.shape[0], actions.device)
        step = temporal_step_features(
            task._dof_pos, task._dof_vel, task._target_states, actions.detach())
        history, history_mask = self.temporal_history.append(step)
        prediction = self.temporal_cm.predict(history, history_mask)
        lift_mm = prediction["supported_lift_mm"]
        reward = torch.tanh(lift_mm / self.temporal_reward_scale_mm)
        if self.temporal_positive_only:
            reward = reward.clamp_min(0)
        # The offline model was fit on complete short histories.  Fade in its
        # credit while the online buffer is being populated instead of treating
        # zero-padded startup state as an equally reliable prediction.
        reward = reward * history_mask.mean(dim=1)
        return reward, prediction, history_mask

    def env_step(self, actions):
        reward, prediction, history_mask = self._predict_reward(actions)
        obs, rewards, dones, infos = super().env_step(actions)
        bonus = self.temporal_reward_coef * reward
        rewards = bonus.unsqueeze(-1) + rewards if rewards.ndim == 2 else bonus + rewards
        self._temporal_calls += 1
        self._temporal_samples += int(reward.numel())
        self._temporal_positive += int((reward > 0).sum())
        self._temporal_reward_sum += float(reward.sum())
        self._temporal_lift_sum += float(prediction["supported_lift_mm"].sum())
        self._temporal_contact_sum += float(prediction["final_contact_probability"].sum())
        if self._temporal_calls % self.horizon_length == 0:
            count = self._temporal_samples
            print("REF2DEX_TEMPORAL_CM_REWARD " + json.dumps({
                "rank": int(getattr(self, "rank", 0)),
                "call": self._temporal_calls,
                "coef": self.temporal_reward_coef,
                "scale_mm": self.temporal_reward_scale_mm,
                "samples": count,
                "mean_reward": self._temporal_reward_sum / count,
                "positive_fraction": self._temporal_positive / count,
                "supported_lift_mean_mm": self._temporal_lift_sum / count,
                "predicted_final_contact_mean": self._temporal_contact_sum / count,
                "history_valid_fraction": float(history_mask[:, :-1].mean()),
            }, sort_keys=True), flush=True)
            self._temporal_samples = self._temporal_positive = 0
            self._temporal_reward_sum = self._temporal_lift_sum = self._temporal_contact_sum = 0.0
        return obs, rewards, dones, infos
