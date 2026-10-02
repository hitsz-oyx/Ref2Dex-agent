"""PPO with common sustained-grasp reward and a separate action-value teacher."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import torch
from torch import nn
import learning.common_agent as common_agent
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent
from src.task.CmResidual.physical_value_contract import HoldTracker, HistoryBuffer, actor_supervision
from src.task.CmResidual.physical_value_live import snapshot, context, contacts
from src.task.CmResidual.physical_value_models import Features, OutcomeNetwork, Teacher, dynamics_output, dynamics_loss
from src.task.CmResidual.task_representation import load_frozen_cm_residual


class DExplorePhysicalValueAgent(DExploreApproachAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self.pv_arm = os.environ["REF2DEX_PHYSICAL_VALUE_ARM"]
        if self.pv_arm not in ("plain_off", "direct_q", "cm_value", "cm_representation",
                               "cm_task_aux", "cm_task_aux_off"):
            raise ValueError("invalid physical-value arm")
        task = self._cm_task()
        payload = torch.load(os.environ["REF2DEX_PHYSICAL_VALUE_CHECKPOINT"], map_location="cpu", weights_only=False)
        if payload["schema"] != "ref2dex.physical_value_models.v1" or payload["gamma"] != self.gamma:
            raise ValueError("physical checkpoint schema/discount mismatch")
        device = task._dof_pos.device
        self.pv_features = Features(**{k: v.to(device) for k, v in payload["stats"].items()}, device=device).to(device)
        self.pv_value = OutcomeNetwork(payload["context_dim"], 0, 1, 9283).to(device)
        self.pv_q = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
        self.pv_dynamics = nn.ModuleList([OutcomeNetwork(payload["context_dim"], 18, 53, 9300 + i).to(device) for i in range(3)])
        self.pv_value.load_state_dict(payload["value"])
        self.pv_q.load_state_dict(payload["direct_q"])
        for network, state in zip(self.pv_dynamics, payload["dynamics"]):
            network.load_state_dict(state)
        self.pv_value_optimizer = torch.optim.Adam(list(self.pv_value.parameters()) + list(self.pv_q.parameters()), lr=.001)
        self.pv_model_optimizer = torch.optim.Adam(self.pv_dynamics.parameters(), lr=.0001)
        self.pv_return_scale = payload["return_scale"].to(device)
        self.pv_reward_scale = payload["reward_scale"].to(device)
        self.pv_task_representation = None
        self.pv_teacher_q = self.pv_q
        self.pv_representation_sha = None
        if self.pv_arm in ("cm_representation", "cm_task_aux", "cm_task_aux_off"):
            representation_path = os.environ["REF2DEX_CM_REPRESENTATION_CHECKPOINT"]
            self.pv_task_representation = load_frozen_cm_residual(representation_path, device)
            self.pv_representation_sha = hashlib.sha256(Path(representation_path).read_bytes()).hexdigest()
            self.pv_teacher_q = OutcomeNetwork(payload["context_dim"], 18, 1, 9284).to(device)
            self.pv_teacher_q.load_state_dict(payload["direct_q"])
            self.pv_teacher_q.eval()
            for parameter in self.pv_teacher_q.parameters():
                parameter.requires_grad_(False)
        self.pv_teacher = Teacher(self.pv_features, self.pv_dynamics, self.pv_value, self.pv_teacher_q,
                                  self.gamma, 20260930286, self.pv_reward_scale,
                                  task_representation=self.pv_task_representation)
        self.pv_tracker = HoldTracker(task.num_envs, device)
        self.pv_history = HistoryBuffer(task.num_envs, device)
        self.pv_previous_action = torch.zeros(task.num_envs, 18, device=device)
        self._pv_active_batch = None
        self._pv_actor_mean = None
        self._pv_aux_active_batch = None
        self._pv_aux_hidden = None
        self._pv_task_aux_gradient_verified = False
        self.pv_aux_head = None
        self.pv_aux_coefficient = .002 if self.pv_arm == "cm_task_aux" else 0.0
        if self.pv_arm in ("cm_task_aux", "cm_task_aux_off"):
            self._pv_aux_target_scale = self.pv_task_representation[4]
            self._pv_aux_target_mean = self.pv_task_representation[3]
        self._pv_rollout_cursor = 0
        self._pv_updates = 0
        self.pv_integrity_verified = False
        self.pv_supervision_gradient_verified = False
        self.model.a2c_network.mu.register_forward_hook(self._capture_mu)

    def _capture_mu(self, module, inputs, output):
        if self._pv_active_batch is not None:
            self._pv_actor_mean = output

    def _install_aux_head(self):
        if self.pv_aux_head is not None:
            return
        width = self.model.a2c_network.mu.in_features
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(240937)
            self.pv_aux_head = nn.Linear(width, 1).to(self.ppo_device)
            nn.init.normal_(self.pv_aux_head.weight, std=.01)
            nn.init.zeros_(self.pv_aux_head.bias)
        self.optimizer.add_param_group({"params": self.pv_aux_head.parameters(),
                                         "lr": self.last_lr})

        def capture(_module, _inputs, hidden):
            if self._pv_aux_active_batch is not None:
                if self._pv_aux_hidden is not None:
                    raise RuntimeError("multiple task-value actor hidden captures")
                self._pv_aux_hidden = hidden

        self._pv_aux_hook = self.model.a2c_network.actor_mlp.register_forward_hook(capture)

    def restore(self, filename):
        super().restore(filename)
        payload = torch.load(filename, map_location="cpu", weights_only=False)
        if "physical_value" in payload:
            raise ValueError("this Probe starts only from source, not a partial research checkpoint")
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        self.pv_initial_model_sha = digest.hexdigest()
        if self.pv_arm in ("cm_task_aux", "cm_task_aux_off"):
            self._install_aux_head()
        print("REF2DEX_PV_INIT " + json.dumps(dict(arm=self.pv_arm, model_sha256=self.pv_initial_model_sha)), flush=True)

    def train(self):
        if not hasattr(self, "pv_initial_model_sha"):
            raise RuntimeError("source restoration did not occur")
        return common_agent.CommonAgent.train(self)

    def init_tensors(self):
        super().init_tensors()
        base = self.experience_buffer.obs_base_shape
        cdim = len(self.pv_features.context_mean)
        pre = dict(pv_history_state=(16, 55), pv_history_action=(16, 18), pv_history_mask=(16, 1),
                   pv_context=(cdim,), pv_next_context=(cdim,), pv_label=(18,), pv_active=(1,))
        post = dict(pv_next_state=(55,), pv_actual_reward=(1,))
        if self.pv_arm in ("cm_task_aux", "cm_task_aux_off"):
            pre.update(pv_aux_target=(1,), pv_aux_mask=(1,))
        for name, shape in {**pre, **post}.items():
            self.experience_buffer.tensor_dict[name] = torch.zeros((*base, *shape), device=self.ppo_device)
        self.update_list += list(pre)
        self.tensor_list += list(pre) + list(post)

    def env_reset(self, env_ids=None):
        result = super().env_reset(env_ids)
        if hasattr(self, "pv_tracker"):
            task = self._cm_task()
            ids = (torch.arange(task.num_envs, device=task._dof_pos.device) if env_ids is None
                   else torch.as_tensor(env_ids, device=task._dof_pos.device, dtype=torch.long).reshape(-1))
            if len(ids):
                self.pv_tracker.reset(ids, task._target_states[ids, 2])
                self.pv_history.reset(ids)
                self.pv_previous_action[ids] = 0
        return result

    @torch.no_grad()
    def get_action_values(self, obs_dict, rand_action_probs):
        result = super().get_action_values(obs_dict, rand_action_probs)
        task = self._cm_task()
        self.pv_history.append(snapshot(task, self.pv_tracker), self.pv_previous_action)
        states, actions, mask = self.pv_history.tensors()
        ctx = context(task, self.pv_tracker)
        next_ctx = context(task, self.pv_tracker, delta=1)
        if self.pv_arm == "plain_off":
            label = result["mus"].clone()
            active = torch.zeros(len(label), dtype=torch.bool, device=label.device)
        elif self.pv_arm in ("cm_task_aux", "cm_task_aux_off"):
            label = result["mus"].clone()
            active = torch.zeros(len(label), dtype=torch.bool, device=label.device)
            residual = self.pv_teacher.task_residual(
                states, actions, mask, ctx, result["mus"].detach()
            )
            target = ((residual - self._pv_aux_target_mean) /
                      self._pv_aux_target_scale).clamp(-8, 8)
            pair = contacts(task).bool().all(-1) & ~task.reset_buf.bool()
            result["pv_aux_target"] = target[:, None]
            result["pv_aux_mask"] = pair.float()[:, None]
        else:
            before = {k: result[k].clone() for k in ("actions", "neglogpacs", "mus", "sigmas") if k in result}
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(result["mus"].device)
            label, active, scores = self.pv_teacher.labels(self.pv_arm, result["mus"], states, actions, mask, ctx, next_ctx)
            if not torch.isfinite(label).all():
                raise FloatingPointError("nonfinite teacher labels")
            if any(not torch.equal(v, result[k]) for k, v in before.items()):
                raise ValueError("teacher changed PPO behavior tensors")
            if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(cuda_rng, torch.cuda.get_rng_state(result["mus"].device)):
                raise ValueError("teacher changed global RNG stream")
            self.pv_integrity_verified = True
        result.update(pv_history_state=states, pv_history_action=actions, pv_history_mask=mask,
                      pv_context=ctx, pv_next_context=next_ctx, pv_label=label, pv_active=active.float()[:, None])
        return result

    def env_step(self, actions):
        result, reward, done, info = super().env_step(actions)
        task = self._cm_task()
        stable, drop = self.pv_tracker.step(task._target_states[:, 2], contacts(task).bool().all(-1))
        reward = reward + stable.reshape_as(reward)
        next_state = snapshot(task, self.pv_tracker)
        if not torch.isfinite(next_state).all() or not torch.isfinite(reward).all():
            raise FloatingPointError("nonfinite physical transition")
        # Natural reference horizon is terminal for this finite task. Keep
        # termination reasons separate in the collector, but no bootstrap
        # beyond an episode's task horizon in the PPO value target.
        info["terminate"] = done.clone()
        self.experience_buffer.update_data("pv_next_state", self._pv_rollout_cursor, next_state)
        self.experience_buffer.update_data("pv_actual_reward", self._pv_rollout_cursor, reward.reshape(-1, 1))
        self._pv_rollout_cursor += 1
        self.pv_previous_action = actions.clamp(-1, 1).detach().clone()
        return result, reward, done, info

    def play_steps(self):
        self._pv_rollout_cursor = 0
        batch = super().play_steps()
        if self._pv_rollout_cursor != self.horizon_length:
            raise ValueError("physical rollout buffer length mismatch")
        return batch

    def prepare_dataset(self, batch):
        super().prepare_dataset(batch)
        for name in self.tensor_list:
            if name.startswith("pv_"):
                self.dataset.values_dict[name] = batch[name]
        self.dataset.values_dict["pv_done"] = batch["dones"]

    def calc_gradients(self, inputs):
        hs, ha, hm = (inputs[k] for k in ("pv_history_state", "pv_history_action", "pv_history_mask"))
        ctx = inputs["pv_context"]
        physical_action = inputs["actions"].clamp(-1, 1)
        hidden = self.pv_features.history(hs, ha, hm)
        context_values = self.pv_features.context(ctx)
        target = inputs["returns"].reshape(-1).detach()
        value = self.pv_value(hidden, context_values).squeeze(-1)
        q = self.pv_q(hidden, context_values, physical_action).squeeze(-1)
        loss = (((value - target) / self.pv_return_scale).square().mean() +
                ((q - target) / self.pv_return_scale).square().mean())
        self.pv_value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.pv_value.parameters()) + list(self.pv_q.parameters()), 5)
        self.pv_value_optimizer.step()
        if self.pv_arm == "cm_value":
            model_loss = 0
            for model in self.pv_dynamics:
                out, contact, r, term = dynamics_output(model, self.pv_features, hs, ha, hm, ctx, physical_action)
                model_loss += dynamics_loss(out, contact, r, term, inputs["pv_next_state"],
                                            inputs["pv_actual_reward"].reshape(-1), inputs["pv_done"].reshape(-1),
                                            self.pv_features, self.pv_reward_scale) / 3
            self.pv_model_optimizer.zero_grad(set_to_none=True)
            model_loss.backward()
            nn.utils.clip_grad_norm_(self.pv_dynamics.parameters(), 5)
            self.pv_model_optimizer.step()
        self._pv_active_batch = inputs
        self._pv_actor_mean = None
        if self.pv_arm in ("cm_task_aux", "cm_task_aux_off"):
            self._pv_aux_active_batch = inputs
            self._pv_aux_hidden = None
        try:
            super().calc_gradients(inputs)
            if (self.pv_arm == "cm_task_aux" and
                    not self._pv_task_aux_gradient_verified):
                gradient = self.pv_aux_head.weight.grad
                if (gradient is None or not torch.isfinite(gradient).all() or
                        gradient.norm() <= 0):
                    raise RuntimeError("task-value auxiliary head has no finite gradient")
                self._pv_task_aux_gradient_verified = True
                print("REF2DEX_PV_TASK_AUX " + json.dumps(dict(
                    arm=self.pv_arm, gradient_norm=float(gradient.norm()))), flush=True)
        finally:
            self._pv_active_batch = None
            self._pv_actor_mean = None
            self._pv_aux_active_batch = None
            self._pv_aux_hidden = None
        self._pv_updates += 1
        if self._pv_updates % 48 == 0:
            print("REF2DEX_PV_UPDATE " + json.dumps(dict(arm=self.pv_arm, epoch=self.epoch_num,
                  value_loss=float(loss.detach()), active_fraction=float(inputs["pv_active"].mean()))), flush=True)

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        result = super()._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, clip_value)
        if self._pv_active_batch is None or self._pv_actor_mean is None:
            raise RuntimeError("actor supervision hook missing")
        b = self._pv_active_batch
        rows = actor_supervision(self._pv_actor_mean, b["sigma"], b["pv_label"], b["pv_active"].reshape(-1))[:, None]
        if self.pv_arm != "plain_off" and not self.pv_supervision_gradient_verified and b["pv_active"].sum() > 0:
            grad = torch.autograd.grad(rows.mean(), self._pv_actor_mean, retain_graph=True)[0]
            if not torch.isfinite(grad).all() or grad.norm() <= 0:
                raise ValueError("teacher produces no finite actor-mean gradient")
            self.pv_supervision_gradient_verified = True
            print("REF2DEX_PV_SUPERVISION " + json.dumps(dict(arm=self.pv_arm, gradient_norm=float(grad.norm()))), flush=True)
        if rows.shape != result["critic_loss"].shape:
            raise ValueError("PPO and teacher loss shapes differ")
        result["critic_loss"] += .01 / self.critic_coef * rows
        if self.pv_arm in ("cm_task_aux", "cm_task_aux_off"):
            if self._pv_aux_hidden is None or self.pv_aux_head is None:
                raise RuntimeError("task-value auxiliary hidden state missing")
            target = self._pv_aux_active_batch["pv_aux_target"].float()
            mask = self._pv_aux_active_batch["pv_aux_mask"].float()
            prediction = self.pv_aux_head(self._pv_aux_hidden.float())
            aux_rows = (prediction - target).square() * mask / mask.mean().clamp_min(.1)
            result["critic_loss"] += self.pv_aux_coefficient / self.critic_coef * aux_rows
        return result

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if hasattr(self, "pv_value"):
            state["physical_value"] = dict(arm=self.pv_arm, initial_model_sha256=self.pv_initial_model_sha,
                behavior_integrity_verified=self.pv_integrity_verified,
                actor_supervision_gradient_verified=self.pv_supervision_gradient_verified,
                representation_sha256=self.pv_representation_sha,
                task_aux_head=(self.pv_aux_head.state_dict() if self.pv_aux_head is not None else None),
                task_aux_coefficient=self.pv_aux_coefficient,
                task_aux_gradient_verified=self._pv_task_aux_gradient_verified,
                value=self.pv_value.state_dict(), direct_q=self.pv_q.state_dict(),
                dynamics=self.pv_dynamics.state_dict(), value_optimizer=self.pv_value_optimizer.state_dict(),
                model_optimizer=self.pv_model_optimizer.state_dict())
        return state
