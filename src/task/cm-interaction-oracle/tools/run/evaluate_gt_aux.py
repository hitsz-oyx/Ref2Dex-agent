#!/usr/bin/env python3
"""First full episodes, actor-only; archive sparse observations and exact GT/MC."""
from pathlib import Path
import sys
import json
import time
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/DExplore/dexplore'),
              str(ROOT/'src/task/cm-interaction-oracle/src')]
from isaacgym import gymtorch
import torch
import evaluate as original
from oracle_y_utility import align_native_reference_tables
from gt_interaction_aux import physical_state, normalized_executed_action, rollout_targets
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.physical_value_contract import HoldTracker
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge
from src.task.CmResidual.dexplore_approach import ApproachConfig, sampled_surface_gap, potential_approach_reward
from src.task.CmResidual.dexplore_grasp_reward import held_lift_reward, contact_lift_progress_reward


class GtAuxPlayer(original.EvalPlayer):
    @torch.no_grad()
    def run(self):
        start=time.monotonic();torch.set_num_threads(2)
        task=self.env.task;device=task.device
        align_native_reference_tables(task)
        task._enable_early_termination=False;task._adaptive_kappa_enabled=False
        task._hybrid_init_prob=1.0
        n=task.num_envs;ids=torch.arange(n,device=device);empty=ids[:0]
        obs=self.env_reset(ids);self.get_batch_size(obs['obs'],1)
        if self.is_rnn:raise ValueError('feedforward actor required')
        if task.start_times.any() or len(task.motion_file)!=3 or n!=96:
            raise ValueError('evaluation must have 96 frame0 episodes / three motions')
        motion=task.data_id.clone();counts=torch.bincount(motion,minlength=3)
        if not torch.equal(counts,torch.full_like(counts,32)):
            raise ValueError('unbalanced motion assignment')
        initial=dict(object=task._target_states.clone(),dof=task._dof_state.clone(),
                     root=task._humanoid_root_states.clone(),motion=motion,
                     start=task.start_times.clone(),obs=obs['obs'].clone())
        initial_hash=fingerprint(initial)
        self.model.eval();model_hash=fingerprint(self.model.state_dict())
        rms_hash=fingerprint(self.running_mean_std.state_dict())
        assets=ROOT/'third_party/DExplore/dexplore/data/assets'
        bridge=DExploreCmv2GeometryBridge(hand_urdf=assets/'inspire_hand_new/inspire_hand_right.urdf',
                object_urdf=assets/'mjcf/airplane.urdf',device=self.device,seed=42)
        config=ApproachConfig()
        def gap():
            geo=bridge.current(task._dof_pos.to(self.device),task._target_states.to(self.device))
            return sampled_surface_gap(geo.hand_points,geo.object_points,config).to(device)
        gap_before=gap();rest=task.hoi_refs[motion,task.ref_index,0,108].clone()
        tracker=HoldTracker(n,device);tracker.reset(ids,task._target_states[:,2])
        completed=torch.zeros(n,dtype=torch.bool,device=device)
        before_states=[];after_states=[];actions=[];dones=[];active_rows=[];rewards=[]
        components=[];gaps=[]
        observations=[];sample_ticks=[];lengths=torch.zeros(n,dtype=torch.long,device=device)
        max_run=torch.zeros_like(lengths);legacy=torch.zeros_like(completed)
        for tick in range(int(task.max_episode_length.max())+2):
            if time.monotonic()-start>360:raise TimeoutError('episode evaluation cap')
            obs=self.env_reset(empty)
            before=physical_state(task);active=~completed
            if tick%8==0:
                observations.append(obs['obs'].cpu().clone());sample_ticks.append(tick)
            action=self.get_action(obs,True).to(device).clone()
            action[completed]=0
            obs,native,done,info=self.env_step(self.env,action)
            after=physical_state(task)
            executed=normalized_executed_action(task.actions)
            if not torch.allclose(executed,action.clamp(-1,1),atol=2e-7,rtol=0):
                raise ValueError('native action capture mismatch')
            gap_after=gap();done=done.bool().flatten()
            pair=after[:,10:15].bool().any(-1)
            base=native.to(device).flatten()
            approach=2*potential_approach_reward(gap_before,gap_after,done,gamma=.99,config=config)
            held=10*held_lift_reward(after[:,2],rest,pair,torch.ones_like(pair))
            progress=5*contact_lift_progress_reward(before[:,2],after[:,2],pair,torch.ones_like(pair),done)
            reward=base+approach
            reward=reward+torch.zeros_like(reward) # original no-link reward addition
            reward=reward+held
            reward=reward+progress
            components.append(torch.stack((base,approach,held,progress),-1).cpu())
            gaps.append(torch.stack((gap_before,gap_after),-1).cpu())
            gap_before=gap_after
            prev_events=tracker.events[completed].clone()
            prev_run=tracker.run_steps[completed].clone()
            prev_max=tracker.max_run[completed].clone()
            prev_stable=tracker.stable[completed].clone()
            prev_drop=tracker.drop_after_success[completed].clone()
            tracker.step(after[:,2],pair)
            tracker.events[completed]=prev_events;tracker.run_steps[completed]=prev_run
            tracker.max_run[completed]=prev_max;tracker.stable[completed]=prev_stable
            tracker.drop_after_success[completed]=prev_drop
            lengths+=active.long();max_run=torch.maximum(max_run,tracker.run_steps)
            legacy|=(tracker.run_steps>=5)&active
            for store,value in ((before_states,before),(after_states,after),(actions,executed),
                 (dones,done|completed),(active_rows,active),(rewards,reward)):
                store.append(value.cpu().clone())
            completed|=done
            if completed.all():break
        if not completed.all():raise ValueError('incomplete episodes')
        before,after,actions,dones,active,reward=[torch.stack(v) for v in
                 (before_states,after_states,actions,dones,active_rows,rewards)]
        gt,chunks,mask=rollout_targets(before,after,actions,dones)
        returns=torch.zeros_like(reward);running=torch.zeros(n)
        for t in reversed(range(len(reward))):
            running=reward[t]+.99*running*(~dones[t]).float()
            returns[t]=running*active[t].float()
        ticks=torch.tensor(sample_ticks)
        payload=dict(schema='ref2dex.gt_aux_evaluation.v1',observations=torch.stack(observations),
            sample_ticks=ticks,gt_target=gt[ticks],action_chunks=chunks[ticks],
            target_mask=mask[ticks],active=active[ticks],mc_return=returns[ticks],
            rewards=reward,reward_components=torch.stack(components),gaps=torch.stack(gaps),rest=rest.cpu(),
            dones=dones,active_full=active,before=before,after=after,
            actions=actions,motion=motion.cpu(),lengths=lengths.cpu(),
            initial= {k:v.cpu() for k,v in initial.items()},initial_fingerprint=initial_hash,
            gamma=.99,model_fingerprint=model_hash,rms_fingerprint=rms_hash)
        sustained=tracker.stable & ~tracker.drop_after_success
        result=dict(episodes=n,seed=293,initial_fingerprint=initial_hash,
            motion_counts=counts.cpu().tolist(),stable45=int(tracker.stable.sum()),
            drop_after_success=int(tracker.drop_after_success.sum()),
            sustained_success=int(sustained.sum()),legacy5=int(legacy.sum()),
            inference='actor-only, no GT or decoder',elapsed_seconds=time.monotonic()-start,
            per_episode=[dict(env_id=i,motion=int(motion[i]),start=0,steps=int(lengths[i]),
                 stable45=bool(tracker.stable[i]),drop=bool(tracker.drop_after_success[i]),
                 sustained=bool(sustained[i]),legacy5=bool(legacy[i]),max_run=int(tracker.max_run[i]*30))
                 for i in range(n)])
        if fingerprint(self.model.state_dict())!=model_hash or fingerprint(self.running_mean_std.state_dict())!=rms_hash:
            raise ValueError('actor/RMS changed during evaluation')
        folder=Path(self.output_file).parent
        torch.save(payload,folder/'pool.pt')
        Path(self.output_file).write_text(json.dumps(result,indent=2)+'\n')
        print('REF2DEX_GT_EVAL '+json.dumps({k:v for k,v in result.items() if k!='per_episode'}),flush=True)


if __name__=='__main__':
    original.EvalPlayer=GtAuxPlayer
    original.main()
