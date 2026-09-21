"""DExplore PPO agent with frozen-Cmv2 reward shaping and random policy init."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from src.task.CmResidual.cm_v2_adapter import FrozenCmv2Adapter
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent
from src.task.CmResidual.dexplore_cm_geometry import (
    DExploreCmv2GeometryBridge, pose_xyzw_to_matrix,
)
from src.task.CmResidual.dexplore_cm_reward import CmRewardConfig, predict_cm_reward


class DExploreCmRewardAgent(DExploreApproachAgent):
    """Add a detached model-based reward; never load or imitate an expert policy."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.cm_reward_coef = float(os.environ["REF2DEX_CM_REWARD_COEF"])
        if self.cm_reward_coef <= 0:
            raise ValueError("REF2DEX_CM_REWARD_COEF must be positive")
        checkpoint = Path(os.environ["REF2DEX_CMV2_CHECKPOINT"])
        self.cm_adapter = FrozenCmv2Adapter(
            checkpoint, os.environ["REF2DEX_CMV2_SHA256"], self.ppo_device)
        asset_root = Path(__file__).resolve().parents[3] / "third_party/DExplore/dexplore/data/assets"
        self.cm_bridge = self.approach_bridge or DExploreCmv2GeometryBridge(
            hand_urdf=asset_root / "inspire_hand_new/inspire_hand_right.urdf",
            object_urdf=asset_root / "mjcf/airplane.urdf", device=self.ppo_device, seed=42)
        self.cm_reward_config = CmRewardConfig(
            effect_translation_scale=float(os.environ.get("REF2DEX_CMV2_EFFECT_TRANSLATION_SCALE", "1.0")),
            effect_translation_cap_m=float(os.environ.get("REF2DEX_CMV2_EFFECT_TRANSLATION_CAP_M", "0.02")),
            compile_swept_topk=os.environ.get("REF2DEX_CMV2_COMPILE", "1") == "1",
        )
        self._cm_reward_calls = 0
        self._cm_reward_sum = 0.0
        self._cm_reward_abs_sum = 0.0
        self._cm_reward_valid = 0
        self._cm_reward_samples = 0
        self._cm_prefilter_candidates = 0

    @torch.inference_mode()
    def _predict_reward(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        task = self._cm_task()
        progress = (task.progress_buf + 1).clamp_max(task.hoi_data.shape[1] - 1)
        reference = task.hoi_data[task.data_id, progress, 106:113]
        return predict_cm_reward(
            adapter=self.cm_adapter, bridge=self.cm_bridge,
            action=self.preprocess_actions(actions).detach(),
            current_native=task._dof_pos, object_root_state=task._target_states,
            goal_pose=pose_xyzw_to_matrix(reference), lower=task.dof_limits_lower,
            upper=task.dof_limits_upper, config=self.cm_reward_config)

    def env_step(self, actions):
        cm = self._predict_reward(actions)
        obs, rewards, dones, infos = super().env_step(actions)
        bonus = self.cm_reward_coef * cm["reward"]
        if rewards.ndim == 2:
            bonus = bonus.unsqueeze(-1)
        rewards = rewards + bonus
        self._cm_reward_calls += 1
        self._cm_reward_sum += float(cm["reward"].sum())
        self._cm_reward_abs_sum += float(cm["reward"].abs().sum())
        self._cm_reward_valid += int(cm["valid"].sum())
        self._cm_prefilter_candidates += int(cm["prefilter_candidate"].sum())
        self._cm_reward_samples += int(cm["reward"].numel())
        if self._cm_reward_calls % self.horizon_length == 0:
            payload = {
                "event": "cm_reward", "call": self._cm_reward_calls,
                "rank": int(getattr(self, "rank", 0)),
                "coef": self.cm_reward_coef,
                "mean": self._cm_reward_sum / self._cm_reward_samples,
                "abs_mean": self._cm_reward_abs_sum / self._cm_reward_samples,
                "valid_fraction": self._cm_reward_valid / self._cm_reward_samples,
                "prefilter_candidate_fraction": self._cm_prefilter_candidates / self._cm_reward_samples,
                "prefilter_skipped_fraction": 1 - self._cm_prefilter_candidates / self._cm_reward_samples,
                "samples": self._cm_reward_samples,
            }
            print("REF2DEX_CM_REWARD " + json.dumps(payload, sort_keys=True), flush=True)
            self._cm_reward_sum = self._cm_reward_abs_sum = 0.0
            self._cm_reward_valid = self._cm_reward_samples = 0
            self._cm_prefilter_candidates = 0
        return obs, rewards, dones, infos
