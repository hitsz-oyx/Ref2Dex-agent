"""GT future supervision of the actor's existing latent during native PPO."""
from __future__ import annotations
import json
import os
import torch
from rl_games.common import a2c_common
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent
from src.task.CmResidual.physical_value_contract import private_initialization
from src.task.CmResidual.paired_evaluation import fingerprint
from oracle_y_utility import align_native_reference_tables
from learning.dexplore_agent import DexploreAgent
from src.task.CmResidual.dexplore_approach import sampled_surface_gap, potential_approach_reward
from src.task.CmResidual.dexplore_grasp_reward import held_lift_reward, contact_lift_progress_reward
from gt_interaction_aux import (InteractionDecoder, physical_state, rollout_targets,
                                auxiliary_rows, normalized_executed_action, shuffle_valid_chunks)
import learning.common_agent as common_agent


class DExploreGtAuxAgent(DExploreApproachAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        align_native_reference_tables(self._cm_task())
        self.gt_arm = os.environ['REF2DEX_GT_AUX_ARM']
        self.gt_coef = 0.0 if self.gt_arm == 'plain' else .05
        self.gt_decoder = None
        self._active_aux = None
        self._hidden = None
        self._capture_rollout = False
        self._rollout_records = []
        self._epoch_aux = []
        self._gradient_checked = False
        self._initial_physics_logged = False

    def restore(self, filename):
        # Restore source optimizer before adding the identically initialized head.
        super().restore(filename)
        before_rng = fingerprint(torch.get_rng_state())
        cuda_rng = fingerprint(torch.cuda.get_rng_state())
        with private_initialization(15042):
            self.gt_decoder = InteractionDecoder(self.model.a2c_network.mu.in_features)
        self.gt_decoder = self.gt_decoder.to(self.ppo_device)
        if before_rng != fingerprint(torch.get_rng_state()) or cuda_rng != fingerprint(torch.cuda.get_rng_state()):
            raise RuntimeError('decoder initialization changed rollout RNG')
        self.optimizer.add_param_group({'params': self.gt_decoder.parameters(), 'lr': self.last_lr})
        def capture(_module, _inputs, hidden):
            if self._active_aux is not None:
                if self._hidden is not None:
                    raise RuntimeError('duplicate actor latent capture')
                self._hidden = hidden
        self.model.a2c_network.actor_mlp.register_forward_hook(capture)
        task = self._cm_task()
        print('REF2DEX_GT_INIT '+json.dumps(dict(arm=self.gt_arm,
            source_model=fingerprint(self.model.state_dict()),
            rms=fingerprint(self.running_mean_std.state_dict()),
            decoder=fingerprint(self.gt_decoder.state_dict()),rng_unchanged=True,
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))), flush=True)

    def train(self):
        if self.gt_decoder is None:
            raise RuntimeError('pinned source checkpoint not restored')
        return common_agent.CommonAgent.train(self)

    def get_stats_weights(self):
        weights = super().get_stats_weights()
        if self.gt_decoder is not None:
            weights['gt_interaction_decoder'] = self.gt_decoder.state_dict()
            weights['gt_interaction_arm'] = self.gt_arm
        return weights

    @torch.no_grad()
    def get_action_values(self,obs_dict,rand_action_probs):
        if not hasattr(self,'_first_action_logged'):
            task=self._cm_task()
            initial=dict(obs=obs_dict['obs'],root=task._root_states,dof=task._dof_state,
                history=task._hist_obs,reference=task.hoi_data,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            print('REF2DEX_GT_FIRST_ACTION_INPUT '+json.dumps(dict(hash=fingerprint(initial))),flush=True)
            result=super().get_action_values(obs_dict,rand_action_probs)
            print('REF2DEX_GT_FIRST_ACTION '+json.dumps(dict(hash=fingerprint(result['actions']))),flush=True)
            self._first_action_logged=True
            return result
        return super().get_action_values(obs_dict,rand_action_probs)

    @torch.no_grad()
    def env_step(self, actions):
        task = self._cm_task()
        before = physical_state(task).to(self.ppo_device) if self._capture_rollout else None
        # Native GPU PhysX with reproducible CPU tensors; keep model/geometry GPU.
        q=task._dof_pos.to(self.ppo_device)
        obj=task._target_states.to(self.ppo_device)
        geo=self.approach_bridge.current(q,obj)
        gap0=sampled_surface_gap(geo.hand_points,geo.object_points,self.approach_config)
        z0=obj[:,2].clone()
        result = DexploreAgent.env_step(self,actions)
        obs,reward,done,info=result
        info=dict(info)
        if "terminate" in info:info["terminate"]=info["terminate"].to(self.ppo_device)
        q=task._dof_pos.to(self.ppo_device)
        obj=task._target_states.to(self.ppo_device)
        geo=self.approach_bridge.current(q,obj)
        gap1=sampled_surface_gap(geo.hand_points,geo.object_points,self.approach_config)
        pair=((task._contact_forces[:,task._contact_body_ids].norm(dim=-1)>.1).any(-1)&
              (task._tar_contact_forces.norm(dim=-1)>.1)).to(self.ppo_device)
        rest=task.hoi_refs[task.data_id,task.ref_index,0,108].to(self.ppo_device)
        reward=reward+2*potential_approach_reward(gap0,gap1,done.bool(),gamma=self.gamma,
                    config=self.approach_config)[:,None]
        reward=reward+torch.zeros_like(reward)
        reward=reward+10*held_lift_reward(obj[:,2],rest,pair,torch.ones_like(pair))[:,None]
        reward=reward+5*contact_lift_progress_reward(z0,obj[:,2],pair,torch.ones_like(pair),
                                                   done.bool())[:,None]
        result=(obs,reward,done,info)
        if self._capture_rollout:
            after = physical_state(task).to(self.ppo_device)
            executed = normalized_executed_action(task.actions).to(self.ppo_device)
            if not torch.allclose(executed, actions.to(self.ppo_device).clamp(-1, 1),atol=2e-7,rtol=0):
                raise RuntimeError('native executed action differs from normalized request')
            if not self._initial_physics_logged:
                print('REF2DEX_GT_INITIAL_PHYSICS '+json.dumps(dict(
                    physical=fingerprint(before),
                    motion=fingerprint(task.data_id), start=fingerprint(task.start_times))),flush=True)
                self._initial_physics_logged=True
            self._rollout_records.append((before, after, executed,
                                           result[2].to(self.ppo_device).clone()))
        return result

    def play_steps(self):
        self._rollout_records = []
        self._capture_rollout = True
        try:
            batch = super().play_steps()
        finally:
            self._capture_rollout = False
        if len(self._rollout_records) != self.horizon_length:
            raise RuntimeError('incomplete actual PPO rollout')
        before, after, actions, dones = [torch.stack([r[k] for r in self._rollout_records])
                                         for k in range(4)]
        target, chunks, mask = rollout_targets(before, after, actions, dones)
        if not hasattr(self,'_first_rollout_logged'):
            print('REF2DEX_GT_FIRST_ROLLOUT '+json.dumps(dict(
                states=fingerprint((before,after)),actions=fingerprint(actions),
                dones=fingerprint(dones),mask=fingerprint(mask))),flush=True)
            self._first_rollout_logged=True
        if self.gt_arm == 'shuffle':
            # Whole chunks rotated between environments at the same rollout time;
            # no additional RNG draws; H and target remain aligned.
            chunks = shuffle_valid_chunks(chunks,mask)
        for key, value in (('gt_target',target),('gt_chunk',chunks),('gt_mask',mask)):
            batch[key] = a2c_common.swap_and_flatten01(value)
        self._rollout_records = []
        return batch

    def prepare_dataset(self, batch):
        super().prepare_dataset(batch)
        for key in ('gt_target','gt_chunk','gt_mask'):
            self.dataset.values_dict[key] = batch[key]

    def calc_gradients(self, inputs):
        self.gt_decoder.zero_grad(set_to_none=True)
        self._active_aux = inputs
        self._hidden = None
        try:
            return super().calc_gradients(inputs)
        finally:
            self._active_aux = None
            self._hidden = None

    def _critic_loss(self, old_values, values, clip, returns, clip_value):
        result = super()._critic_loss(old_values, values, clip, returns, clip_value)
        if self._hidden is None or self._active_aux is None:
            raise RuntimeError('actor latent/GT minibatch missing')
        inp = self._active_aux
        prediction = self.gt_decoder(self._hidden.float(), inp['gt_chunk'].float(),
                                      detach=self.gt_arm == 'stopgrad')
        rows = auxiliary_rows(prediction, inp['gt_target'].float(), inp['gt_mask'].float())
        if not self._gradient_checked and inp['gt_mask'].sum() > 0:
            grad = torch.autograd.grad(rows.mean(), self._hidden, retain_graph=True,
                                       allow_unused=True)[0]
            norm = 0.0 if grad is None else float(grad.norm())
            actor_parameter=next(self.model.a2c_network.actor_mlp.parameters())
            critic_parameter=next(self.model.a2c_network.critic_mlp.parameters())
            ag,cg=torch.autograd.grad(rows.mean(),(actor_parameter,critic_parameter),
                retain_graph=True,allow_unused=True)
            actor_norm=0.0 if ag is None else float(ag.norm())
            critic_norm=0.0 if cg is None else float(cg.norm())
            if critic_norm != 0 or ((self.gt_arm != 'stopgrad') != (actor_norm>0)):
                raise RuntimeError('auxiliary enters wrong PPO trunk')
            should_reach = self.gt_arm != 'stopgrad'
            if should_reach != (norm > 0):
                raise RuntimeError('auxiliary encoder gradient contract mismatch')
            print('REF2DEX_GT_GRAD '+json.dumps(dict(arm=self.gt_arm,
                raw_latent_grad_norm=norm, raw_actor_weight_grad_norm=actor_norm,
                auxiliary_critic_grad_norm=critic_norm, applied_coef=self.gt_coef,
                applied_latent_grad_norm=norm*self.gt_coef)), flush=True)
            self._gradient_checked = True
        result['critic_loss'] = result['critic_loss'] + (self.gt_coef/self.critic_coef)*rows
        self._epoch_aux.append((float(rows.mean().detach()),float(inp['gt_mask'].mean())))
        return result

    def train_epoch(self):
        self._epoch_aux=[]
        result = super().train_epoch()
        if not self._epoch_aux:
            raise RuntimeError('no GT auxiliary optimization')
        count=len(self._epoch_aux)
        print('REF2DEX_GT_EPOCH '+json.dumps(dict(epoch=self.epoch_num, arm=self.gt_arm,
            aux_loss=sum(v[0] for v in self._epoch_aux)/count,
            valid_fraction=sum(v[1] for v in self._epoch_aux)/count,
            minibatches=count)),flush=True)
        return result
