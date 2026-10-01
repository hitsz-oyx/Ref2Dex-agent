"""One explicitly task-specific Cm-off PPO continuation, with fresh optimizer."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import time
import torch
from src.task.CmResidual.dexplore_approach_agent import DExploreApproachAgent
from src.task.CmResidual.physical_value_live import contacts
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.hold_plateau_training import plateau_probability, phase_lift_bonus, install_training_reset, canonical_model_state


class DExploreHoldPlateauAgent(DExploreApproachAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        if any((self.approach_reward_coef,self.held_lift_reward_coef,self.lift_progress_reward_coef,self.grasp_link_reward_coef,self.min_grasp_links)):
            raise ValueError('legacy extra rewards must be zero in this phase-specific Probe')
        task = self._cm_task()
        if task.num_envs != 96 or self.horizon_length != 16 or abs(task.dt-1/30)>1e-8:
            raise ValueError('frozen training shape/dt drift')
        task._enable_early_termination = False
        task._adaptive_kappa_enabled = False
        task._hybrid_init_prob = 1.
        generation = json.loads(Path(os.environ['REF2DEX_HOLD_REFERENCES_MANIFEST']).read_text())
        by_name = {r['name']:r for r in generation['references']}
        starts,stops = [],[]
        for motion,filename in enumerate(task.motion_file):
            r = by_name[Path(filename).name]
            if Path(filename).resolve() != Path(r['generated']).parent.resolve():
                raise ValueError('training loaded reference path drift')
            start,stop = r['plateau_reference_frames_inclusive']
            starts.append(start);stops.append(stop)
            if int(task.max_episode_length[motion]) != r['generated_frames']:
                raise ValueError('loaded reference length drift')
            for key in ('obj_pos_vel','obj_rot_vel','right_hand_pos_vel','right_dof_pos_vel','robot_dof_pos_vel','key_body_pos_vel','human_rot_vel'):
                if task.hoi_data_dict[motion][key][start:stop+1].any():
                    raise ValueError('training plateau velocity drift')
        self.hold_starts = torch.tensor(starts, device=task._dof_pos.device)
        self.hold_stops = torch.tensor(stops, device=task._dof_pos.device)
        install_training_reset(task,self.hold_starts)
        self.hold_begin = time.monotonic()
        self.hold_samples = self.hold_phase_samples = self.hold_positive_samples = 0
        self.hold_bonus_sum = 0.

    def restore(self, filename):
        # Copy the fixed self-trained network/RMS only. Optimizer was created
        # afresh in CommonAgent; do not reuse source epoch/moments/frame counters.
        p = torch.load(filename,map_location='cpu',weights_only=False)
        self.model.load_state_dict(canonical_model_state(p['model']))
        if self.normalize_input:
            self.running_mean_std.load_state_dict(p['running_mean_std'])
        if self._normalize_input:
            self._input_mean_std.load_state_dict(p['amp_input_mean_std'])
        if self.optimizer.state:
            raise ValueError('fresh PPO optimizer required')
        self.epoch_num = 0
        self.frame = 0
        self.hold_initial_fingerprints = dict(model=fingerprint(self.model.state_dict()),
            observation_rms=fingerprint(self.running_mean_std.state_dict()),
            amp_rms=fingerprint(self._input_mean_std.state_dict()))
        # Translation is an additive meter-scale target. The inherited .055m
        # standard deviation is large relative to the .01m physical probes.
        # Freeze .005m wrist exploration before collection; all other entries
        # and deterministic means retain their source initialization.
        with torch.no_grad():
            self.model.a2c_network.sigma[:3].fill_(math.log(.005))
        self.hold_exploration_std = self.model.a2c_network.sigma.detach().exp().cpu().tolist()
        print('HOLD_INIT '+json.dumps(dict(source=str(filename),fresh_optimizer=True,
            source_fingerprints=self.hold_initial_fingerprints,
            training_exploration_std=self.hold_exploration_std)),flush=True)

    def train_epoch(self):
        if time.monotonic()-self.hold_begin>1750:
            raise TimeoutError('training wall budget')
        task=self._cm_task()
        task.hold_plateau_probability=plateau_probability(self.epoch_num)
        result=super().train_epoch()
        if self.epoch_num%20==0:
            print('HOLD_TRAIN '+json.dumps(dict(epoch=self.epoch_num,frame=self.frame,
                plateau_start_probability=task.hold_plateau_probability,
                reset_counts_start_plateau=task.hold_reset_counts.cpu().tolist(),
                phase_samples=self.hold_phase_samples,positive_bonus_samples=self.hold_positive_samples,
                phase_bonus_sum=self.hold_bonus_sum)),flush=True)
        return result

    def env_step(self,actions):
        obs,reward,done,info=super().env_step(actions)
        task=self._cm_task();motion=task.data_id
        progress=task.progress_buf
        bonus=phase_lift_bonus(task._target_states[:,2],task.hoi_refs[motion,task.ref_index,0,108],
            contacts(task),progress,self.hold_starts[motion],self.hold_stops[motion])
        reward=reward+bonus.reshape_as(reward)
        if not torch.isfinite(obs['obs']).all() or not torch.isfinite(reward).all():
            raise ValueError('nonfinite training state/reward')
        self.hold_samples+=len(bonus)
        self.hold_phase_samples+=int(((progress>=self.hold_starts[motion])&(progress<=self.hold_stops[motion])).sum())
        self.hold_positive_samples+=int((bonus>0).sum());self.hold_bonus_sum+=float(bonus.sum())
        info['terminate']=done.clone()  # finite reference horizon is terminal.
        return obs,reward,done,info

    def get_stats_weights(self):
        result=super().get_stats_weights()
        if hasattr(self,'hold_initial_fingerprints'):
            result['hold_plateau']=dict(no_cm=True,initial_fingerprints=self.hold_initial_fingerprints,
                fixed_exploration_std=self.hold_exploration_std,
                new_optimizer=True,reset_counts_start_plateau=self._cm_task().hold_reset_counts.cpu(),
                phase_starts=self.hold_starts.cpu(),phase_stops=self.hold_stops.cpu(),
                sampled_interactions=self.hold_samples,phase_samples=self.hold_phase_samples,
                positive_bonus_samples=self.hold_positive_samples,phase_bonus_sum=self.hold_bonus_sum)
        return result
