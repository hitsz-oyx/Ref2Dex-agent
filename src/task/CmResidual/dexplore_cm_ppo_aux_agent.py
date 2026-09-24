"""Matched PPO arms with frozen Cm targets for actor-feature auxiliary learning."""
from __future__ import annotations

import json
import os

import torch
from torch import nn
from rl_games.algos_torch import torch_ext

import learning.common_agent as common_agent
from src.task.CmResidual.cm_ppo_auxiliary import (
    cm_candidate_targets, masked_auxiliary_loss,
)
from src.task.CmResidual.contact_aware_cm import RawContactAwareCm
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent


class DExploreCmPpoAuxAgent(DExploreApproachAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.cm_aux_coefficient = float(os.environ["REF2DEX_CM_AUX_COEF"])
        if self.cm_aux_coefficient not in (0.0, .002):
            raise ValueError("Cm auxiliary Probe only permits coefficients 0 or .002")
        device = self.ppo_device
        self.cm_teacher = RawContactAwareCm(torch.zeros(67), torch.ones(67)).to(device).eval()
        payload = torch.load(os.environ["REF2DEX_CONTACT_CM_CHECKPOINT"],
                             map_location=device, weights_only=False)
        if payload.get("schema") != "ref2dex.contact_aware_cm.v1" or (
                payload.get("name") != "raw_action"):
            raise ValueError("contact Cm teacher schema/name mismatch")
        self.cm_teacher.load_state_dict(payload["model"], strict=True)
        self.cm_teacher.requires_grad_(False)
        self.cm_aux_head = None
        self._cm_aux_hook = None
        self._active_cm_aux = None
        self._captured_actor_hidden = None
        self._aux_loss_sum = self._aux_mask_sum = 0.0
        self._aux_minibatches = 0
        self._aux_gradient_checked = False

    def _install_aux_head(self):
        if self.cm_aux_head is not None:
            return
        width = self.model.a2c_network.mu.in_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(240936)
            head = nn.Linear(width, 3)
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)
        self.cm_aux_head = head.to(self.ppo_device)
        self.optimizer.add_param_group({"params": self.cm_aux_head.parameters(),
                                        "lr": self.last_lr})

        def capture(_module, _inputs, hidden):
            if self._active_cm_aux is not None:
                if self._captured_actor_hidden is not None:
                    raise RuntimeError("multiple actor hidden captures per PPO minibatch")
                self._captured_actor_hidden = hidden

        self._cm_aux_hook = self.model.a2c_network.actor_mlp.register_forward_hook(capture)

    def restore(self, filename):
        checkpoint = torch_ext.load_checkpoint(filename)
        aux_state = checkpoint.get("cm_aux_head")
        if aux_state is not None:
            self._install_aux_head()  # optimizer group must exist before restoring it
        super().restore(filename)
        if aux_state is None:
            self._install_aux_head()
        else:
            self.cm_aux_head.load_state_dict(aux_state, strict=True)

    def train(self):
        # Runner.run_train restores the pinned --checkpoint before calling
        # train(); DExplore's legacy resume_from remains "None" in this path.
        # Never silently fall back to random weights if Runner skipped restore.
        if not os.environ.get("REF2DEX_SCRATCH_RESUME_CHECKPOINT"):
            raise ValueError("Cm auxiliary Probe requires pinned self-trained resume")
        if self.cm_aux_head is None:
            raise RuntimeError("Cm auxiliary Probe checkpoint was not restored")
        return common_agent.CommonAgent.train(self)

    def get_stats_weights(self):
        weights = super().get_stats_weights()
        if self.cm_aux_head is not None:
            weights["cm_aux_head"] = self.cm_aux_head.state_dict()
            weights["cm_aux_coefficient"] = self.cm_aux_coefficient
        return weights

    def init_tensors(self):
        super().init_tensors()
        shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["cm_aux_target"] = torch.zeros(
            (*shape, 3), dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict["cm_aux_mask"] = torch.zeros(
            (*shape, 1), dtype=torch.float32, device=self.ppo_device)
        self.update_list += ["cm_aux_target", "cm_aux_mask"]
        self.tensor_list += ["cm_aux_target", "cm_aux_mask"]

    @torch.inference_mode()
    def get_action_values(self, obs_dict, rand_action_probs):
        result = super().get_action_values(obs_dict, rand_action_probs)
        task = self._cm_task()
        contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                   (task._tar_contact_forces.norm(dim=-1) > .1) &
                   (task.reset_buf.reshape(-1) == 0))
        target = cm_candidate_targets(
            self.cm_teacher, task._dof_pos, task._dof_vel,
            task._target_states, result["mus"], contact)
        result["cm_aux_target"] = target
        result["cm_aux_mask"] = contact.float()[:, None]
        return result

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["cm_aux_target"] = batch_dict["cm_aux_target"]
        self.dataset.values_dict["cm_aux_mask"] = batch_dict["cm_aux_mask"]

    def calc_gradients(self, input_dict):
        if self._active_cm_aux is not None or self.cm_aux_head is None:
            raise RuntimeError("Cm auxiliary gradient state is invalid")
        self.cm_aux_head.zero_grad(set_to_none=True)
        self._active_cm_aux = (input_dict["cm_aux_target"], input_dict["cm_aux_mask"])
        self._captured_actor_hidden = None
        try:
            result = super().calc_gradients(input_dict)
            if self.cm_aux_coefficient and not self._aux_gradient_checked and (
                    self._active_cm_aux[1].sum() > 0):
                grad = self.cm_aux_head.weight.grad
                if grad is None or not torch.isfinite(grad).all() or grad.norm() <= 0:
                    raise RuntimeError("Cm-on auxiliary head has no finite gradient")
                print("REF2DEX_CM_AUX_GRAD " + json.dumps({
                    "epoch": self.epoch_num, "head_grad_norm": float(grad.norm())},
                    sort_keys=True), flush=True)
                self._aux_gradient_checked = True
            return result
        finally:
            self._active_cm_aux = None
            self._captured_actor_hidden = None

    def _critic_loss(self, value_preds_batch, values, curr_e_clip,
                     return_batch, clip_value):
        info = super()._critic_loss(value_preds_batch, values, curr_e_clip,
                                    return_batch, clip_value)
        if self._active_cm_aux is None or self._captured_actor_hidden is None:
            raise RuntimeError("Cm auxiliary target or shared actor hidden missing")
        target, mask = self._active_cm_aux
        prediction = self.cm_aux_head(self._captured_actor_hidden.float())
        rows = masked_auxiliary_loss(prediction, target.float(), mask.float())
        if rows.shape != info["critic_loss"].shape:
            raise ValueError("Cm auxiliary and PPO critic loss shape mismatch")
        if self.critic_coef <= 0:
            raise ValueError("PPO critic coefficient must be positive")
        info["critic_loss"] = info["critic_loss"] + (
            self.cm_aux_coefficient / self.critic_coef) * rows
        self._aux_loss_sum += float(rows.mean().detach())
        self._aux_mask_sum += float(mask.mean().detach())
        self._aux_minibatches += 1
        return info

    def train_epoch(self):
        result = super().train_epoch()
        count = max(1, self._aux_minibatches)
        print("REF2DEX_CM_AUX_EPOCH " + json.dumps({
            "epoch": self.epoch_num, "coefficient": self.cm_aux_coefficient,
            "mean_auxiliary_loss": self._aux_loss_sum / count,
            "mean_contact_mask": self._aux_mask_sum / count,
            "minibatches": self._aux_minibatches}, sort_keys=True), flush=True)
        self._aux_loss_sum = self._aux_mask_sum = 0.0
        self._aux_minibatches = 0
        return result
