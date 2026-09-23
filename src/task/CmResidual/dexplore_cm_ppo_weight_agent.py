"""Use frozen one-step Cm predictions to weight PPO actor samples, not rewards."""
from __future__ import annotations

import json
import os

import torch

from src.task.CmResidual.cmlite import FrozenCmLite
from src.task.CmResidual.cm_ppo_weight import effect_actor_weight
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent


class DExploreCmPpoWeightAgent(DExploreApproachAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.cm_actor_weight_coefficient = float(os.environ["REF2DEX_CM_ACTOR_WEIGHT_COEF"])
        if not 0 < self.cm_actor_weight_coefficient <= 2:
            raise ValueError("Cm PPO actor weight coefficient must be in (0,2]")
        self.cmlite = FrozenCmLite(
            os.environ["REF2DEX_CMLITE_CHECKPOINT"], self.ppo_device,
            os.environ["REF2DEX_CMLITE_SHA256"])
        self._cm_weight_calls = 0
        self._cm_weight_sum = self._cm_probability_sum = self._cm_effect_sum = 0.0
        self._cm_weight_count = 0

    def init_tensors(self):
        super().init_tensors()
        shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["cm_actor_weight"] = torch.ones(
            shape, dtype=torch.float32, device=self.ppo_device)
        self.update_list += ["cm_actor_weight"]
        self.tensor_list += ["cm_actor_weight"]

    @torch.inference_mode()
    def get_action_values(self, obs_dict, rand_action_probs):
        result = super().get_action_values(obs_dict, rand_action_probs)
        task = self._cm_task()
        prediction = self.cmlite.predict(
            task._dof_pos, result["actions"], task._target_states)
        weights = effect_actor_weight(
            prediction["contact_probability"], prediction["delta_world"],
            coefficient=self.cm_actor_weight_coefficient)
        result["cm_actor_weight"] = weights
        self._cm_weight_calls += 1
        self._cm_weight_count += weights.numel()
        self._cm_weight_sum += float(weights.sum())
        self._cm_probability_sum += float(prediction["contact_probability"].sum())
        self._cm_effect_sum += float(prediction["delta_world"][:, 2].abs().sum())
        if self._cm_weight_calls % self.horizon_length == 0:
            print("REF2DEX_CM_PPO_WEIGHT " + json.dumps({
                "epoch": self.epoch_num, "coefficient": self.cm_actor_weight_coefficient,
                "mean_weight": self._cm_weight_sum / self._cm_weight_count,
                "mean_contact_probability": self._cm_probability_sum / self._cm_weight_count,
                "mean_abs_predicted_dz_m": self._cm_effect_sum / self._cm_weight_count,
                "samples": self._cm_weight_count,
            }, sort_keys=True), flush=True)
            self._cm_weight_count = 0
            self._cm_weight_sum = self._cm_probability_sum = self._cm_effect_sum = 0.0
        return result

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["cm_actor_weight"] = batch_dict["cm_actor_weight"]

    def calc_gradients(self, input_dict):
        weighted = dict(input_dict)
        weights = input_dict["cm_actor_weight"]
        if weights.shape != input_dict["rand_action_mask"].shape:
            raise ValueError("Cm PPO weight and exploration mask shape mismatch")
        weighted["rand_action_mask"] = input_dict["rand_action_mask"] * weights
        return super().calc_gradients(weighted)
