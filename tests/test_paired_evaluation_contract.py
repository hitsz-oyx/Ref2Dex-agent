import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.task.CmResidual.paired_evaluation import (
    capture_initial, capture_rng, fingerprint, repeatability_gate, restore_initial, restore_rng,
)


def episodes(n=384):
    return [dict(pair_id=str(i), motion_id=i%3, start_frame=0, steps=560,
                 initial_object_height=.9, control_dt=1/30, stable_success=False,
                 drop_after_success=False) for i in range(n)]


class PairedEvaluationContracts(unittest.TestCase):
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


if __name__=='__main__': unittest.main()
