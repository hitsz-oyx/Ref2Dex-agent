import math,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from src.task.CmResidual.executable_contact_options import hold_target,hold_action,TableClearance
import tests.test_contact_consequence as fixtures


def native_targets(a,q,offset,scale):
    a=a.clone();a[:,6:]=(a[:,6:]+1)/2
    t=offset+scale*a;t[:,:6]+=q[:,:6]
    t[:,7]=t[:,6]*1.05;t[:,9]=t[:,8]*1.05;t[:,11]=t[:,10]*1.05;t[:,13]=t[:,12]*1.05;t[:,16]=t[:,15]*.6;t[:,17]=t[:,15]*.8
    return t


def test_hold_inverse_tracks_fixed_wrist_and_native_finger_couplings_after_motion():
    scale=torch.ones(18);scale[3:6]=math.pi;offset=torch.zeros(18)
    q=torch.rand(5,18)*.4;target=hold_target(q,offset,scale)
    later=q.clone();later[:,:6]+=.12
    action=hold_action(target,later,offset,scale)
    assert (action[:,:6]<0).all()
    assert torch.allclose(native_targets(action,later,offset,scale),target,atol=1e-6)
    beyond=q.clone();beyond[:,6:]=10
    bounded=hold_target(beyond,offset,scale)
    assert (bounded[:,[6,8,10,12,14,15]]==1).all()
    assert torch.allclose(native_targets(hold_action(bounded,q,offset,scale),q,offset,scale),bounded,atol=1e-6)


def test_full_vertex_clearance_respects_thin_y_table_axis_and_object_rotation():
    table=torch.tensor([[-1.,-.005,-1.],[1.,.005,1.]])
    vertices=torch.tensor([[-.1,-.02,-.03],[.1,.02,.03]])
    geometry=TableClearance(vertices,table)
    tp=torch.tensor([[0.,0.,.8,math.sqrt(.5),0.,0.,math.sqrt(.5)]])
    op=torch.tensor([[0.,0.,1.,0.,0.,0.,1.]])
    assert geometry.axis==1
    assert abs(float(geometry.clearance(op,tp))-.165)<1e-6
    op[:,3:7]=torch.tensor([[math.sqrt(.5),0.,0.,math.sqrt(.5)]])
    assert abs(float(geometry.clearance(op,tp))-.175)<1e-6


def test_actual_loop_executes_full_feedback_window_and_fixed_hold_target():
    from scripts.collect_executable_contact_options import executable_player
    Player=fixtures.ContactConsequenceTests.fake_player_type()
    class FeedbackPlayer(Player):
        def __init__(self):
            super().__init__();t=self.env.task;t._pd_action_offset=torch.zeros(18);t._pd_action_scale=torch.ones(18);t._pd_action_scale[3:6]=math.pi
            t._target_states[:64,2]=.94
            t._table_states=t._root_states.view(96,3,13)[:,1];t._table_states[:,2]=.88;t._table_states[:,3]=math.sqrt(.5);t._table_states[:,6]=math.sqrt(.5)
            t._action_to_pd_targets=lambda a:native_targets(a,t._dof_pos,t._pd_action_offset,t._pd_action_scale)
        def get_action(self,obs,deterministic):
            a=super().get_action(obs,deterministic)
            if float(self.model.weight[0,0])!=0:a[:,0]+=self.env.task.progress_buf*.001
            return a
        def env_step(self,env,action):
            q=env.task._action_to_pd_targets(action.clone());out=super().env_step(env,action)
            env.task._dof_pos[:]=q;env.task._dof_pos[:,0]+=.001
            return out
    def vertices(path,device):
        if 'table' in str(path):return torch.tensor([[-1.,-.005,-1.],[1.,.005,1.]])
        return torch.tensor([[-.01,-.01,-.01],[.01,.01,.01]])
    with tempfile.TemporaryDirectory() as d:
        args=SimpleNamespace(output=Path(d),seed=420,assignment_seed=7420,windows_per_stratum=2,max_steps=100,wall_seconds=60)
        with patch('src.task.CmResidual.executable_contact_options.obj_vertices',side_effect=vertices):
            executable_player(SimpleNamespace(EvalPlayer=FeedbackPlayer),args,torch,SimpleNamespace())().run()
        data=torch.load(Path(d)/'records.pt',weights_only=False)
        assert not data['future_done'].any() and data['frozen_experts']
        rare=data['outcome']['initially_clear']
        expected=torch.where(data['assignment']==4,.25,.125)
        expected[rare]=torch.where((data['assignment'][rare]==4)|(data['assignment'][rare]==6),.4,.04)
        assert (data['propensity']==expected).all()
        assert torch.allclose(data['allocation_probabilities'].sum(-1),torch.ones(len(rare)))
        assert rare[data['sampling_cohort']==0].all()
        hold=data['assignment']==6;assert hold.any()
        assert torch.allclose(data['actual_pd_targets'][hold],data['hold_target'][hold,None].expand(-1,10,-1),atol=1e-6)
        expert=(data['assignment']<6)&(data['assignment']!=4)
        assert expert.any() and (data['actual_action'][expert,-1,0]>data['actual_action'][expert,0,0]).all()
        assert torch.equal(data['history'][:,-1,:49],data['state'])
