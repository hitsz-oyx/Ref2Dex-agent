import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import json

import torch

from src.task.CmResidual.contact_consequence import local_outcomes, opportunity_gate, prefix_errors


class ContactConsequenceTests(unittest.TestCase):
    @staticmethod
    def fake_player_type():
        class Gym:
            def __init__(self): self.frame=0
            def get_frame_count(self,sim): return self.frame
            def get_actor_count(self,env): return 0
            def set_actor_root_state_tensor(self,sim,tensor): pass
            def set_dof_state_tensor(self,sim,tensor): pass

        class Player:
            def __init__(self):
                self.device='cpu';self.states=None;self.is_rnn=False
                self.batch_initialized=False
                self.inference_calls=0
                self.model=torch.nn.Linear(1,1,bias=False)
                self.model.weight.data.zero_()
                self.running_mean_std=torch.nn.Identity()
                task=SimpleNamespace(gym=Gym(),sim=object(),envs=list(range(96)),num_envs=96,dt=1/30,
                    device='cpu',dr_randomizations={},projtype='None',_motion_sampler=None,
                    _reset_default_env_ids=[],_reset_ref_env_ids=[])
                task._root_states=torch.zeros(288,13)
                task._target_states=task._root_states.view(96,3,13)[:,2]
                task._target_states[:,2]=.9;task._target_states[:,6]=1
                task._dof_state=torch.zeros(1728,2)
                task._dof_pos=task._dof_state.view(96,18,2)[:,:,0]
                task._dof_vel=task._dof_state.view(96,18,2)[:,:,1]
                task._rigid_body_state=torch.zeros(2208,13)
                task._contact_forces=torch.ones(96,23,3)
                task._tar_contact_forces=torch.ones(96,3)
                task._contact_body_ids=torch.arange(5)
                task.obs_buf=torch.zeros(96,1442)
                task.progress_buf=torch.zeros(96,dtype=torch.long)
                task.data_id=torch.arange(96)%3
                task.start_times=torch.zeros(96,dtype=torch.long)
                task.max_episode_length=torch.full((3,),100,dtype=torch.long)
                task.ref_index=torch.zeros(96,dtype=torch.long)
                task.hoi_refs=torch.zeros(3,1,101,150);task.hoi_refs[:,:,:,108]=.9
                for name in ('_curr_obs','_hist_obs','contact_reset','_terminate_buf'):
                    setattr(task,name,torch.zeros(96,4) if name!='_terminate_buf' else torch.zeros(96,dtype=torch.bool))
                task._refresh_sim_tensors=lambda:None
                self.env=SimpleNamespace(task=task)
                self.frozen_experts=[]
                for arm in range(6):
                    model=torch.nn.Linear(1,1,bias=False)
                    model.weight.data.fill_(0 if arm==4 else .2)
                    self.frozen_experts.append((model,torch.nn.Identity()))
            def env_reset(self,ids): return {'obs':self.env.task.obs_buf.clone()}
            def restore(self,filename): pass
            def get_batch_size(self,obs,count):
                self.batch_initialized=True
                return len(obs)
            def get_action(self,obs,deterministic):
                if not self.batch_initialized: raise ValueError('missing native batch initialization')
                self.inference_calls+=1
                result=torch.zeros(96,18);result[:,0]=float(self.model.weight[0,0]);return result
            def env_step(self,env,action):
                task=env.task;task.gym.frame+=1;task.progress_buf+=1
                task._target_states[:,2]+=action[:,0]*.01
                task.obs_buf[:,0]=task._target_states[:,2]
                return task.obs_buf,torch.zeros(96),torch.zeros(96,dtype=torch.bool),{}
        return Player

    def test_native_loop_replays_prefix_and_executes_candidate_then_own_base(self):
        from scripts.collect_contact_consequences import build_player
        Player=self.fake_player_type()

        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'first';branch=Path(directory)/'branch';first.mkdir();branch.mkdir()
            args=SimpleNamespace(reference=None,output=first,arm=4,repeat=0,seed=330,
                                 max_states=32,max_steps=100,wall_seconds=60)
            original=SimpleNamespace(EvalPlayer=Player)
            wrapper=SimpleNamespace(unwrap_tensor=lambda x:x)
            # Exercise actual expert restoration under the simulator's Python
            # interpreter, including the compiled-checkpoint key wrapper.
            pool=build_player(original,args,torch,wrapper)()
            route=json.loads(Path('src/task/CmResidual/configs/hf02_temporal_canonical_route.json').read_text())
            hashes={str((Path.cwd()/spec['checkpoint']).resolve()):spec['sha256'] for spec in route['experts'].values()}
            with patch('scripts.collect_contact_consequences.sha',side_effect=lambda p:hashes[str(p)]), \
                 patch.object(torch,'load',return_value={'model':{'_orig_mod.weight':torch.full((1,1),.25)},'running_mean_std':{}}):
                pool.restore('unused')
            self.assertEqual(len(pool.frozen_experts),6)
            self.assertTrue(all(float(m.weight[0,0])==.25 and not m.weight.requires_grad for m,_ in pool.frozen_experts))
            first_player=build_player(original,args,torch,wrapper)()
            first_player.run()
            args=SimpleNamespace(**dict(vars(args),reference=first/'records.pt',output=branch,arm=0))
            branch_player=build_player(original,args,torch,wrapper)()
            branch_player.run()
            data=torch.load(branch/'records.pt',weights_only=False)
            ref=torch.load(first/'records.pt',weights_only=False)
            selected=data['selected']
            self.assertEqual(int(selected.sum()),32)
            self.assertTrue(data['paired_valid'].all())
            self.assertEqual(data['initial_fingerprint'],ref['initial_fingerprint'])
            self.assertTrue((data['future_action'][selected,:2,0]==.2).all())
            self.assertTrue((data['future_action'][selected,2:]==0).all())
            self.assertTrue((data['outcome']['supported_lift_mm']>3.7).all())
            self.assertTrue((data['window_steps'][selected]==10).all())
            self.assertFalse(data['future_done'][selected].any())
            self.assertEqual(first_player.inference_calls,branch_player.inference_calls)
            self.assertEqual(first_player.inference_calls,len(ref['action'])*6)

    def test_randomized_native_windows_use_actual_states_known_propensity_and_complete_followup(self):
        from scripts.collect_randomized_contact_consequences import randomized_player
        Player=self.fake_player_type()
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output=Path(directory),seed=331,assignment_seed=7331,
                                 windows_per_episode=8,max_steps=100,wall_seconds=60)
            player=randomized_player(SimpleNamespace(EvalPlayer=Player),args,torch,
                                     SimpleNamespace(unwrap_tensor=lambda x:x))()
            player.run()
            data=torch.load(Path(directory)/'records.pt',weights_only=False)
            self.assertTrue(data['assignment_after_observation'])
            self.assertFalse(data['teacher_or_optimizer'])
            self.assertFalse(data['future_done'].any())
            self.assertGreaterEqual(len(data['assignment']),96)
            self.assertTrue((torch.bincount(data['assignment'],minlength=6)>0).all())
            self.assertTrue(torch.allclose(data['propensity'],torch.full_like(data['propensity'],1/6)))
            for row in range(len(data['assignment'])):
                selected=data['candidate_actions'][row,data['assignment'][row]]
                self.assertTrue(torch.equal(data['actual_action'][row,:2],selected[None].expand(2,-1)))
            self.assertTrue((data['actual_action'][:,2:]==0).all())
            self.assertTrue(torch.equal(data['history'][:,-1,:49],data['state']))
            self.assertEqual(len(set(data['episode_id'])),96)

    def test_targeted_allocation_executes_proposal_or_base_and_reobserves(self):
        from scripts.collect_randomized_contact_consequences import randomized_player
        class Selector:
            def __init__(self,*args):self.models=[torch.nn.Linear(1,1)];self.calls=0
            def predict(self,history,candidate,rest,motion,start,trigger):
                self.calls+=1
                arm=torch.where(torch.arange(len(history))%2==0,0,4)
                return dict(proposed_arm=arm,predicted_gain_mm=torch.where(arm==0,5.,0.),
                            lower_gain_mm=torch.where(arm==0,4.,0.),predicted_contact=torch.ones(len(history)),
                            predicted_drop=torch.zeros(len(history)))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'frozen.pt';checkpoint.write_bytes(b'frozen')
            args=SimpleNamespace(output=root,seed=341,assignment_seed=7341,ranker=checkpoint,
                                 windows_per_episode=8,max_steps=100,wall_seconds=60)
            with patch('src.task.CmResidual.contact_selector.FrozenContactSelector',Selector),patch.object(torch.cuda,'synchronize'):
                player=randomized_player(SimpleNamespace(EvalPlayer=self.fake_player_type()),args,torch,
                                         SimpleNamespace(unwrap_tensor=lambda x:x))()
                player.run()
            data=torch.load(root/'records.pt',weights_only=False);trace=data['policy_trace']
            active=trace['proposed_arm'].long()!=4;treat=trace['treatment'].bool()
            self.assertTrue(torch.equal(data['propensity'],torch.where(active,.5,1.)))
            self.assertGreater(int((active&treat).sum()),50)
            self.assertGreater(int((active&~treat).sum()),50)
            self.assertFalse(treat[~active].any())
            self.assertTrue(torch.equal(data['assignment'],torch.where(treat,trace['proposed_arm'].long(),4)))
            self.assertTrue((data['actual_action'][treat,:2,0]==.2).all())
            self.assertTrue((data['actual_action'][~treat,:2,0]==0).all())
            self.assertTrue((data['actual_action'][:,2:]==0).all())
            self.assertTrue(torch.equal(data['history'][:,-1,:49],data['state']))
            ids=data['env_id']==0
            self.assertGreater(int(ids.sum()),2)
            self.assertGreater(float(data['state'][ids,38].max()-data['state'][ids,38].min()),0)
            self.assertFalse(data['future_done'].any())

    def test_labels_distinguish_supported_progress_contact_loss_and_existing_lift_drop(self):
        future=torch.zeros(3,10,49)
        future[:,:,38]=torch.tensor([1.04,1.04,1.01])[:,None]
        contact=torch.ones(3,10,dtype=torch.bool)
        contact[1,4:]=False
        result=local_outcomes(future,contact,torch.ones(3),torch.tensor([.95,.95,1.]))
        self.assertAlmostEqual(float(result['supported_lift_mm'][0]),40,places=3)
        self.assertAlmostEqual(float(result['supported_lift_mm'][1]),16,places=3)
        self.assertEqual(result['drop'].tolist(),[False,True,False])
        self.assertEqual(result['drop_eligible'].tolist(),[True,True,False])
        self.assertAlmostEqual(float(result['contact_loss'][1]),.6,places=5)

    def test_native_option_learning_updates_actual_choice_probabilities_on_complete_episodes(self):
        from scripts.train_recovery_options import recovery_player
        from src.task.CmResidual.trajectory_selector import effective_commands
        class Physics:
            def __init__(self,*args):self.models=[torch.nn.Linear(1,1)]
            def inputs(self,history,candidate,*args):
                n=len(history)
                return dict(history=history,context=torch.zeros(n,22),actions=effective_commands(candidate),
                            physics=torch.zeros(n,6,22),recommended=torch.zeros(n,dtype=torch.long))
        Player=self.fake_player_type()
        class CompletePlayer(Player):
            def __init__(self):
                super().__init__();del self.batch_initialized
            def env_step(self,env,action):
                obs,reward,_,info=super().env_step(env,action)
                return obs,reward,env.task.progress_buf>=40,info
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'frozen.pt';checkpoint.write_bytes(b'frozen')
            args=SimpleNamespace(output=root,trajectory=checkpoint,policy=None,cm_on=True,assignment_seed=7381,
                                 rollouts=1,epochs=2,evaluate=False,smoke=True,wall_seconds=60,learnable_guide=True,physical_reward=True)
            with patch('src.task.CmResidual.recovery_option_policy.FrozenRecoveryPhysics',Physics),patch.object(torch.cuda,'synchronize'):
                recovery_player(SimpleNamespace(EvalPlayer=CompletePlayer),args,torch,SimpleNamespace(unwrap_tensor=lambda x:x))().run()
            result=json.loads((root/'results.json').read_text());data=torch.load(root/'decisions.pt',weights_only=False)[0]
            self.assertEqual(result['native_env_steps'],96*40)
            self.assertEqual(result['scientific_label'],'ENGINEERING_SMOKE')
            self.assertEqual(result['optimizer_updates'],4)
            self.assertTrue(result['policy_parameters_changed'] and result['frozen_cm_experts'])
            self.assertLess(result['rollouts'][0]['first_actual_option_ratio_error'],1e-6)
            self.assertTrue(torch.allclose(data['old_logprob'].exp(),torch.where(data['selected']==0,.5,.1),atol=1e-6))
            trace=torch.load(root/'episode_traces.pt',weights_only=False)[0]
            cold=torch.load(root/'cold_states.pt',weights_only=False)[0]
            self.assertTrue((trace['native_reward']==0).all())
            self.assertTrue((trace['training_reward']>0).any() and (trace['training_reward']<=1).all())
            expected=((trace['object_state'][:,:,2]-cold['rest'])/.03).clamp(0,1)
            self.assertTrue(torch.allclose(trace['training_reward'],expected,atol=1e-5))
            self.assertEqual(result['reward_mode'],'supported_height_fraction')

    def test_five_policy_pool_merges_physical_aliases_and_records_native_targets(self):
        from scripts.collect_randomized_contact_consequences import randomized_player
        class Selector:
            policy_names=('cm','state_only','action_shuffled','always_base','best_fixed')
            def __init__(self,*args):self.models=[torch.nn.Linear(1,1)]
            def predict(self,history,candidate,*args):
                n=len(history);pool=torch.tensor([0,1,2,4,0]).expand(n,-1)
                return dict(proposed_arm=pool[:,0],policy_proposals=pool,predicted_gain_mm=torch.ones(n),
                            lower_gain_mm=torch.ones(n),predicted_contact=torch.ones(n),predicted_release=torch.zeros(n))
        Player=self.fake_player_type()
        class NativePlayer(Player):
            def __init__(self):
                super().__init__()
                def targets(a):
                    a[:,6:]=(a[:,6:]+1)/2
                    a[:,:6]+=self.env.task._dof_pos[:,:6]
                    for dst,src,scale in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
                        a[:,dst]=a[:,src]*scale
                    return a
                self.env.task._action_to_pd_targets=targets
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'frozen.pt';checkpoint.write_bytes(b'frozen')
            args=SimpleNamespace(output=root,seed=351,assignment_seed=7351,trajectory=checkpoint,
                                 windows_per_episode=8,max_steps=100,wall_seconds=60)
            with patch('src.task.CmResidual.trajectory_selector.FrozenTrajectorySelectors',Selector),patch.object(torch.cuda,'synchronize'):
                randomized_player(SimpleNamespace(EvalPlayer=NativePlayer),args,torch,SimpleNamespace(unwrap_tensor=lambda x:x))().run()
            data=torch.load(root/'records.pt',weights_only=False);trace=data['policy_trace'];arm=data['assignment']
            self.assertEqual(data['schema'],'ref2dex.selector_pool_contact_consequence.v1')
            self.assertTrue((torch.bincount(trace['selected_policy'].long(),minlength=5)>0).all())
            self.assertTrue(torch.equal(data['propensity'],torch.where(arm==4,.2,.8)))
            selected=data['candidate_pd_targets'][torch.arange(len(arm)),arm]
            self.assertTrue(torch.equal(data['actual_pd_targets'][:,0],selected))
            self.assertTrue((data['actual_action'][:,2:]==0).all())
            self.assertTrue(torch.equal(data['history'][:,-1,:49],data['state']))
            self.assertGreater(int((data['env_id']==0).sum()),2)

    def test_opportunity_requires_independent_repeat_gain_not_first_repeat_winners(self):
        lift=torch.zeros(32,6,2);contact=torch.ones_like(lift);drop=torch.zeros_like(lift,dtype=torch.bool)
        lift[:,0,0]=20
        gate=opportunity_gate(lift,contact,drop,paired_valid=True)
        self.assertFalse(gate['passed'])
        self.assertEqual(gate['mean_confirmed_uplift_mm'],0)
        lift[:,0,1]=5
        self.assertTrue(opportunity_gate(lift,contact,drop,paired_valid=True)['passed'])
        self.assertFalse(opportunity_gate(lift,contact,drop,paired_valid=False)['passed'])
        self.assertEqual(opportunity_gate(lift,contact,drop,paired_valid=False)['label'],'UNCLEAR')
        contact[:,0,1]=0
        self.assertFalse(opportunity_gate(lift,contact,drop,paired_valid=True)['passed'])

    def test_gate_rejects_opportunity_smaller_than_base_repeat_noise(self):
        lift=torch.zeros(32,6,2);lift[:,4,1]=5;lift[:,0,:]=8
        gate=opportunity_gate(lift,torch.ones_like(lift),torch.zeros_like(lift,dtype=torch.bool),paired_valid=True)
        self.assertEqual(gate['required_mean_uplift_mm'],10)
        self.assertFalse(gate['passed'])

    def test_pairing_checks_pose_joint_and_velocity_and_quaternion_sign(self):
        state=torch.zeros(4,49);state[:,42]=1
        actual=state.clone();actual[0,42]=-1;actual[1,36]=.002;actual[2,3]=.02;actual[3,18]=.1
        valid,errors=prefix_errors(actual,state)
        self.assertEqual(valid.tolist(),[True,False,False,False])
        self.assertEqual(float(errors['object_rotation_rad'][0]),0)


if __name__=='__main__':unittest.main()
