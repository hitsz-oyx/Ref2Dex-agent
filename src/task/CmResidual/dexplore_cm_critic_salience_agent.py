"""Train-only frozen Cm state salience for PPO critic, leaving actor loss intact."""
from __future__ import annotations

import json
import os

import torch

from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.cm_critic_salience import (
    critic_salience_weight, normalized_critic_loss,
)
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent


class DExploreCmCriticSalienceAgent(DExploreApproachAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.cm_critic_salience_coefficient = float(
            os.environ["REF2DEX_CM_CRITIC_SALIENCE_COEF"])
        self.cmlite = FrozenCmLite(
            os.environ["REF2DEX_CMLITE_CHECKPOINT"], self.ppo_device,
            os.environ["REF2DEX_CMLITE_SHA256"])
        self._cm_critic_weight_sum = 0.0
        self._cm_critic_weight_count = 0
        self._active_cm_critic_weight = None

    def init_tensors(self):
        super().init_tensors()
        shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["cm_critic_weight"] = torch.ones(
            shape, dtype=torch.float32, device=self.ppo_device)
        self.update_list += ["cm_critic_weight"]
        self.tensor_list += ["cm_critic_weight"]

    @torch.inference_mode()
    def get_action_values(self, obs_dict, rand_action_probs):
        result = super().get_action_values(obs_dict, rand_action_probs)
        task = self._cm_task()
        # Policy mean is deterministic for a state; exploratory sampled actions
        # must not change the critic's state-level target distribution.
        prediction = self.cmlite.predict(
            task._dof_pos, result["mus"].clamp(-1, 1), task._target_states)
        weight = critic_salience_weight(
            prediction["contact_probability"], prediction["delta_world"],
            coefficient=self.cm_critic_salience_coefficient)
        result["cm_critic_weight"] = weight
        self._cm_critic_weight_sum += float(weight.sum())
        self._cm_critic_weight_count += weight.numel()
        if self._cm_critic_weight_count >= self.horizon_length * weight.numel():
            print("REF2DEX_CM_CRITIC_SALIENCE " + json.dumps({
                "epoch": self.epoch_num, "samples": self._cm_critic_weight_count,
                "mean_raw_weight": self._cm_critic_weight_sum / self._cm_critic_weight_count,
            }, sort_keys=True), flush=True)
            self._cm_critic_weight_sum = 0.0
            self._cm_critic_weight_count = 0
        return result

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["cm_critic_weight"] = batch_dict["cm_critic_weight"]

    def calc_gradients(self, input_dict):
        if self._active_cm_critic_weight is not None:
            raise RuntimeError("nested Cm critic gradient call")
        self._active_cm_critic_weight = input_dict["cm_critic_weight"]
        try:
            return super().calc_gradients(input_dict)
        finally:
            self._active_cm_critic_weight = None

    def _critic_loss(self, value_preds_batch, values, curr_e_clip,
                     return_batch, clip_value):
        info = super()._critic_loss(value_preds_batch, values, curr_e_clip,
                                    return_batch, clip_value)
        if self._active_cm_critic_weight is None:
            raise RuntimeError("Cm critic weight missing during loss")
        info["critic_loss"] = normalized_critic_loss(
            info["critic_loss"], self._active_cm_critic_weight)
        return info
