import random
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.task.CmResidual.paired_evaluation import (
    capture_initial, capture_rng, fingerprint, repeatability_gate, restore_initial, restore_rng,
    physical_property_value,
)


def episodes(n=384):
    return [dict(pair_id=str(i), motion_id=i%3, start_frame=0, steps=560,
                 initial_object_height=.9, control_dt=1/30, stable_success=False,
                 drop_after_success=False) for i in range(n)]


class PairedEvaluationContracts(unittest.TestCase):
    def test_physical_property_dtype_metadata_is_finite_and_struct_padding_is_ignored(self):
        vector=SimpleNamespace(x=1.,y=2.,z=3.,dtype=np.dtype([('x','f4'),('y','f4'),('z','f4')]))
        result=physical_property_value(SimpleNamespace(com=vector,mass=2.))
        self.assertEqual(result['com']['z'],3.)
        self.assertIsInstance(result['com']['dtype'],str)
        dtype=np.dtype({'names':['hasLimits','lower'], 'formats':['bool','f4'],
                        'offsets':[0,4], 'itemsize':8})
        a=np.zeros(3,dtype=dtype); b=np.zeros(3,dtype=dtype)
        a['lower']=.25; b['lower']=.25
        b.view('u1').reshape(3,8)[:,1:4]=255
        self.assertEqual(fingerprint(a),fingerprint(b))
        b['lower'][0]=.5
        self.assertNotEqual(fingerprint(a),fingerprint(b))

    def test_gate_rejects_cancelling_episode_changes(self):
        first=episodes(); second=episodes()
        for i in range(18): first[i]['stable_success']=True
        for i in range(18,36): second[i]['stable_success']=True
        result=repeatability_gate(first,second,trace_contract_valid=True,closed_loop_equivalent=True)
        self.assertEqual(result['absolute_rate_difference'],0)
        self.assertFalse(result['passed'])
        self.assertEqual(result['success_label_disagreements'],36)

    def test_gate_requires_trace_and_closed_loop_and_post_success_drop(self):
        a=episodes(); b=episodes()
        self.assertTrue(repeatability_gate(a,b,trace_contract_valid=True,closed_loop_equivalent=True)['passed'])
        self.assertFalse(repeatability_gate(a,b,trace_contract_valid=False,closed_loop_equivalent=True)['passed'])
        self.assertFalse(repeatability_gate(a,b,trace_contract_valid=True,closed_loop_equivalent=False)['passed'])
        for i in range(25): b[i]['drop_after_success']=True
        self.assertFalse(repeatability_gate(a,b,trace_contract_valid=True,closed_loop_equivalent=True)['passed'])
        b[0]['initial_object_height'] += .001
        with self.assertRaisesRegex(ValueError,'pairing mismatch'):
            repeatability_gate(a,b,trace_contract_valid=True,closed_loop_equivalent=True)

    def test_rng_replay_covers_python_numpy_and_torch(self):
        saved=capture_rng()
        first=(random.random(),np.random.randn(5),torch.randn(5))
        restore_rng(saved)
        second=(random.random(),np.random.randn(5),torch.randn(5))
        self.assertEqual(fingerprint(first),fingerprint(second))

    def test_restore_includes_root_velocity_histories_and_rnn_and_rejects_warm_state(self):
        class Gym:
            frame=0
            def get_frame_count(self,sim): return self.frame
            def set_actor_root_state_tensor(self,sim,tensor): pass
            def set_dof_state_tensor(self,sim,tensor): pass
        task=SimpleNamespace(gym=Gym(),sim=object(),dr_randomizations={},projtype='None',
            _motion_sampler=None,device='cpu',_reset_default_env_ids=[],_reset_ref_env_ids=[])
        for name in ('_root_states','_dof_state','_rigid_body_state','_contact_forces',
                     '_tar_contact_forces','obs_buf','progress_buf','data_id','start_times',
                     '_curr_obs','_hist_obs','contact_reset','_terminate_buf'):
            setattr(task,name,torch.randn(3,13))
        task._refresh_sim_tensors=lambda:None
        player=SimpleNamespace(states=[torch.randn(1,3,5)],is_rnn=True,device='cpu')
        obs={'obs':task.obs_buf.clone()}
        saved=capture_initial(task,player,obs,{'mass':2.0})
        digest=fingerprint(saved)
        task._root_states[:,7:]+=100
        task._hist_obs.zero_(); player.states[0].zero_()
        actual=restore_initial(task,player,saved,lambda x:x,{'mass':2.0})
        self.assertEqual(fingerprint(capture_initial(task,player,actual,{'mass':2.0})),digest)
        self.assertEqual(fingerprint(saved),digest)
        task.gym.frame=1
        with self.assertRaisesRegex(ValueError,'warm simulator'):
            restore_initial(task,player,saved,lambda x:x,{'mass':2.0})

    def test_native_player_paths_share_setter_order_and_keep_success_followup(self):
        from scripts.run_paired_physical_value_environment import make_player
        class Gym:
            def __init__(self): self.frame=0; self.calls=[]
            def get_frame_count(self,sim): return self.frame
            def get_actor_count(self,env): return 0
            def set_actor_root_state_tensor(self,sim,tensor): self.calls.append('root')
            def set_dof_state_tensor(self,sim,tensor): self.calls.append('dof')
        class Player:
            def __init__(self):
                self.device='cpu'; self.states=None; self.is_rnn=False; self.normalize_input=False
                self.model=torch.nn.Linear(1,1)
                task=SimpleNamespace(gym=Gym(),sim=object(),envs=list(range(96)),num_envs=96,dt=1/30,
                    device='cpu',dr_randomizations={},projtype='None',_motion_sampler=None,
                    _reset_default_env_ids=[],_reset_ref_env_ids=[])
                task._root_states=torch.zeros(288,13)
                task._target_states=task._root_states.view(96,3,13)[:,2]
                task._target_states[:,2]=.9
                task._dof_state=torch.zeros(1728,2)
                task._dof_pos=task._dof_state.view(96,18,2)[:,:,0]
                task._dof_vel=task._dof_state.view(96,18,2)[:,:,1]
                task._rigid_body_state=torch.zeros(2208,13)
                task._contact_forces=torch.zeros(96,23,3)
                task._tar_contact_forces=torch.zeros(96,3)
                task._contact_body_ids=torch.arange(5)
                task.obs_buf=torch.zeros(96,1442)
                task.progress_buf=torch.zeros(96,dtype=torch.long)
                task.data_id=torch.arange(96)%3
                task.start_times=torch.zeros(96,dtype=torch.long)
                for name in ('_curr_obs','_hist_obs','contact_reset','_terminate_buf'):
                    setattr(task,name,torch.zeros(96,4) if name!='_terminate_buf' else torch.zeros(96,dtype=torch.bool))
                task._refresh_sim_tensors=lambda:None
                self.env=SimpleNamespace(task=task)
            def env_reset(self,ids): return {'obs':self.env.task.obs_buf.clone()}
            def get_batch_size(self,obs,batch): return 96
            def get_action(self,obs,deterministic): return torch.zeros(96,18)
            def env_step(self,env,action):
                task=env.task; task.gym.frame+=1; task.progress_buf+=1
                task.obs_buf[:,0]=task.gym.frame
                task._target_states[:,2]=1.0 if task.gym.frame<49 else .9
                task._contact_forces[:]=1. if task.gym.frame<49 else 0.
                task._tar_contact_forces[:]=1. if task.gym.frame<49 else 0.
                done=torch.full((96,),task.gym.frame==55)
                return task.obs_buf,torch.zeros(96),done,{'terminate':torch.zeros(96,dtype=torch.bool)}
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'first'; repeat=Path(directory)/'repeat'
            first.mkdir();repeat.mkdir()
            args=SimpleNamespace(initial=None,trace=None,replay_actions=False,run_dir=first,
                wall_seconds=60,arm='plain_off',training_seed=286,eval_seed=288,checkpoint_sha256='a')
            baseline=make_player(SimpleNamespace(EvalPlayer=Player),args,torch,SimpleNamespace(unwrap_tensor=lambda x:x))()
            baseline.run()
            args=SimpleNamespace(**dict(vars(args),initial=first/'initial_state.pt',trace=first/'trace.pt',
                replay_actions=True,run_dir=repeat))
            second=make_player(SimpleNamespace(EvalPlayer=Player),args,torch,SimpleNamespace(unwrap_tensor=lambda x:x))()
            second.run()
            self.assertEqual(baseline.env.task.gym.calls,['root','dof'])
            self.assertEqual(second.env.task.gym.calls,baseline.env.task.gym.calls)
            result=json.loads((repeat/'results.json').read_text())
            self.assertTrue(result['closed_loop_equivalent'])
            self.assertEqual(result['stable_success_count'],96)
            self.assertEqual(result['drop_after_success_count'],96)
            self.assertEqual(result['per_episode'][0]['first_success_step'],45)
            self.assertEqual(result['per_episode'][0]['first_drop_step'],49)
            self.assertEqual(result['per_episode'][0]['followup_after_success_steps'],10)


if __name__=='__main__': unittest.main()
