"""Producer reset restoration and mechanism-only pilot quotas."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'tools/run/collect_rigid_pilot.py'
spec=importlib.util.spec_from_file_location('rigid_pilot',SOURCE)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)


def coverage():
    return [dict(source_id=4,phase=p,geometry_class=g,windows=(5 if p=='close' and g=='elongated' else
                22 if p=='hold' and g=='elongated' else 0)) for p in pilot.PHASES for g in pilot.CLASSES]


def test_quota_waterfills_source4_mechanisms_without_identity():
    rows=coverage();allocation=pilot.quotas(rows)
    assert sum(r['rollouts'] for r in allocation)==96
    assert next(r for r in allocation if r['phase']=='hold' and r['geometry_class']=='elongated')['rollouts']==0
    levels=[r['prior_windows']+r['rollouts'] for r in allocation if r['rollouts']]
    assert max(levels)-min(levels)<=1
    for row in rows:row['object_name']='an irrelevant identity'
    assert pilot.quotas(rows)==allocation
    with pytest.raises(ValueError):pilot.quotas(rows[:-1])


def test_reset_restores_frozen_goal_even_when_live_state_and_init_corrupted():
    import torch
    class Task:
        def reset_target_pose(self,ids,apply_reset=False):
            self.goal_states[ids]=-5
            self.goal_init_state[ids]=-7
    frozen=torch.arange(26,dtype=torch.float32).reshape(2,13)
    task=Task();task.goal_states=torch.zeros(2,13);task.goal_init_state=torch.zeros(2,13)
    task.root_state_tensor=torch.zeros(4,13);task.goal_object_indices=torch.tensor([1,3]);task.reset_goal_buf=torch.ones(2)
    evidence=[]
    pilot.install_bound_reset(Task,frozen,lambda t,ids,stamp:evidence.append(stamp))
    task.reset_target_pose(torch.tensor([0,1]))
    assert torch.equal(task.goal_states,frozen)
    assert torch.equal(task.root_state_tensor[[1,3]],frozen)
    task.goal_states[:]=99;task.goal_init_state[:]=88
    task.reset_target_pose(torch.tensor([1]))
    assert torch.equal(task.goal_states[1],frozen[1])
    assert torch.equal(task.goal_init_state[1],frozen[1])
    assert len(evidence)==2 and evidence[0]<=evidence[1]
    assert (task.reset_goal_buf==0).all()


def test_registry_samples_class_and_normals_from_actual_mesh():
    import trimesh
    mesh=trimesh.creation.box(extents=[.2,.1,.01])
    geo=pilot.sample_mesh(mesh,np.random.default_rng(198))
    assert geo['surface'].shape==(2048,3) and geo['query'].shape==(32,3)
    assert np.allclose(np.linalg.norm(geo['normals'],axis=-1),1.)
    assert np.array_equal(geo['query'],geo['surface'][::64])
    assert pilot.geometry_class(pilot.pca_shape(geo['surface']))=='flat'


def test_fresh_consumer_goal_is_independent_of_own_future_labels():
    from ptv3_shared_wam_v6.fresh_rigid import reset_bound_sample
    from test_rigid_dataset import bundle
    seq,geo,contact,_,_=bundle()
    reset=dict(execution_id='fresh',reset_id='0',clock_id='test',at_ns=10)
    def source(payload,role,name):
        return pilot.bind_source(payload,source_role=role,source_kind='declared_task',source_id=name,
                                 reset=reset,observed_at_ns=1,available_at_ns=2)
    goal=np.eye(4);goal[2,3]=.9
    instruction=pilot.instruction_from_displacement(source(goal,'task_instruction','goal'),
        source(dict(rule='fixed_world_translation',translation_m=[0,0,0],target_stage='task_completion'),'task_instruction','rule'),
        source(geo['query'],'reset_geometry','query'),source(geo['surface'],'reset_geometry','surface'),
        reset=reset,object_slot=0,hand_slot=0,goal_type='se3_pose')
    before=reset_bound_sample(seq,geo,contact,3,0,4,[0,0,3],instruction,reset)
    seq['object_poses'][4:,0,0,3]=5
    seq['joints'][4:]=7
    after=reset_bound_sample(seq,geo,contact,3,0,4,[0,0,3],instruction,reset)
    assert np.array_equal(before['goal_object_delta'],after['goal_object_delta'])
    assert not np.array_equal(before['trajectory_target'],after['trajectory_target'])
    assert not np.array_equal(before['scene_target'],after['scene_target'])
    with pytest.raises(ValueError):
        reset_bound_sample(seq,geo,contact,3,0,4,[0,0,3],instruction,dict(reset,reset_id='other'))
