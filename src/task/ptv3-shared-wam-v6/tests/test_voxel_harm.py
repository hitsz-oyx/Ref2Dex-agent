"""CPU regression contracts for first-point diagnosis and accounting bounds."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[4];TASK=ROOT/'src/task/ptv3-shared-wam-v6'
sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/audit'),str(ROOT/'src/task/wam-rl-policy-v3/src')]


def sample():
    return dict(xyz=np.array([[0.,0.,0.],[.02,0.,0.]],np.float32),features=np.zeros((2,18),np.float32),
        point_valid=np.ones(2,bool),start_hand=np.zeros((2,11,3),np.float32),start_hand_valid=np.zeros((2,11),bool))


def test_first_point_names_slot_joint_and_geometric_relations():
    from ptv3_shared_wam_v6.negative_result_invariants import voxel_first_failure
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    s=sample();s['start_hand_valid'][1,[0,7]]=True;s['start_hand'][1,7]=[.1,-2.01,.3]
    config=dict(voxel_origin_m=[-2.]*3,voxel_m=.01,hand_in_ptv3=True)
    pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[1,2,3]
    rt=object.__new__(V4Runtime);rt.config=config
    assert not rt.workspace_valid([s])[0]
    d=voxel_first_failure(s,config,pose)
    assert d['point']==('hand',1,7) and d['axis']=='y' and d['violated_bound']=='lower'
    assert d['hand_identity']['slot_name']=='left' and d['hand_identity']['canonical_joint_name']=='index_tip'
    assert d['point_index_in_gate']==3 and d['invalid_point_count']==1
    np.testing.assert_allclose(d['world_xyz'],[3.01,2.1,3.3],atol=1e-6)
    g=d['geometry_relation'];np.testing.assert_allclose(g['object_center_world_xyz'],[1,2,3])
    np.testing.assert_allclose(g['same_slot_wrist_world_xyz'],[1,2,3])
    assert g['object_anchor_distance_m']==pytest.approx(np.linalg.norm([.1,-2.01,.3]),abs=1e-6)
    assert g['nearest_included_scene_surface_m']==pytest.approx(np.linalg.norm([.08,-2.01,.3]),abs=1e-6)
    assert g['nearest_valid_same_slot_fingertip_m']==0.
    # Mutant: treating canonical7 as native7 changes the semantic identity.
    assert d['hand_identity']['canonical_joint_name']!='ring_base'


def test_invalid_hand_mask_and_hand_exclusion_are_not_false_scene_failures():
    from ptv3_shared_wam_v6.negative_result_invariants import voxel_first_failure
    s=sample();s['start_hand'][0,0]=[-3,0,0]
    config=dict(voxel_origin_m=[-2.]*3,voxel_m=.01,hand_in_ptv3=True)
    assert voxel_first_failure(s,config,np.eye(4)) is None
    s['start_hand_valid'][0,0]=True
    assert voxel_first_failure(s,config,np.eye(4))['hand_identity']['canonical_joint_name']=='wrist'
    assert voxel_first_failure(s,dict(config,hand_in_ptv3=False),np.eye(4)) is None
    s['xyz'][0]=[-3,0,0]
    d=voxel_first_failure(s,config,np.eye(4));assert d['point']==('scene',0) and d['hand_identity'] is None


def test_float32_voxel_index65535_inclusive_but65536_rejected():
    from ptv3_shared_wam_v6.negative_result_invariants import voxel_first_failure
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    s=sample();config=dict(voxel_origin_m=[0.]*3,voxel_m=1.,hand_in_ptv3=True)
    rt=object.__new__(V4Runtime);rt.config=config;s['xyz'][0]=[65535.5,0,0]
    assert rt.workspace_valid([s])[0] and voxel_first_failure(s,config,np.eye(4)) is None
    s['xyz'][0,0]=65536.
    d=voxel_first_failure(s,config,np.eye(4))
    assert not rt.workspace_valid([s])[0] and d['voxel_index'][0]==65536 and d['violated_bound']=='upper'


def test_admission_vs_every_tick_preserves_safe_prefix_and_detects_gate_deletion():
    from audit_voxel_failures_cpu import compare_gate_protocols
    ticks=np.arange(3,7);valid=np.ones((4,2),bool);valid[2:,0]=False
    r=compare_gate_protocols(valid,ticks,[4,6])
    assert r['initial_accepted']==2 and r['first_invalid_tick']==[5,None]
    assert r['admission_only']['retained_full_controls']==12
    assert r['every_tick']['retained_full_controls']==11
    assert r['retained_primary_control_difference']==0
    mutant=compare_gate_protocols(np.ones_like(valid),ticks,[4,6])
    assert mutant['every_tick']['terminated']!=r['every_tick']['terminated']
    assert mutant['every_tick']['retained_full_controls']!=r['every_tick']['retained_full_controls']
    assert 'future actions' in r['not_identified']


def test_first_admission_rejection_and_h4_clock_are_explicit():
    from audit_voxel_failures_cpu import compare_gate_protocols
    valid=np.array([[False,True],[True,True]])
    r=compare_gate_protocols(valid,[3,4],[4,4])
    assert r['initial_accepted']==1 and r['admission_only']['rejected_at_admission']==1
    assert r['admission_only']['stop_tick']==r['every_tick']['stop_tick']==[3,4]
    with pytest.raises(ValueError,match='H4'):compare_gate_protocols(valid,[2,3],[4,4])


def test_episode_bounds_match_exhaustive_small_set_cardinalities():
    from audit_harm_budget_cpu import count_bounds
    for draws in range(1,7):
        universe=range(draws)
        for unknown in range(draws+1):
            unknown_set=set(range(unknown))
            for ended in range(draws+1):
                for voxel in range(ended+1):
                    observed={key:[] for key in ['ended_unknown','unknown_voxel_overlap','unknown_or_voxel','remaining_defined_terminal_nonvoxel']}
                    for completed_tuple in itertools.combinations(universe,ended):
                        completed=set(completed_tuple);u=unknown_set&completed
                        for voxel_tuple in itertools.combinations(completed_tuple,voxel):
                            v=set(voxel_tuple)
                            values=[len(u),len(u&v),len(u|v),len(completed-(u|v))]
                            for key,value in zip(observed,values):observed[key].append(value)
                    bound=count_bounds(draws,unknown,ended,voxel)
                    for key,values in observed.items():assert bound[key]==[min(values),max(values)]


def test_transition_bounds_are_not_episode_fraction_and_cover_feasible_durations():
    from audit_harm_budget_cpu import transition_bounds
    b=transition_bounds(15,3,[1,2],minimum=3,maximum=5)
    observed=[]
    for durations in itertools.product(range(3,6),repeat=3):
        if sum(durations)>15:continue
        for n in [1,2]:
            for affected in itertools.combinations(range(3),n):observed.append(sum(durations[i] for i in affected))
    assert b['transitions']==[min(observed),max(observed)]==[3,10]
    with pytest.raises(ValueError):transition_bounds(2,3,[1,2])


def test_actual_r3_bounds_have_nonzero_remaining_opportunity_and_no_causal_claim():
    from audit_harm_budget_cpu import count_bounds,transition_bounds
    r=count_bounds(56449,9314,53347,27869)
    assert r['ended_unknown']==[6212,9314]
    assert r['remaining_defined_terminal_nonvoxel']==[16164,25478]
    assert r['unknown_or_voxel']==[27869,37183]
    b=transition_bounds(16515072,53347,[27869,27869])
    assert b['transitions']==[83607,16438638]
    assert b['fraction_of_updated_transitions'][0]<.01 and b['fraction_of_updated_transitions'][1]>.99
    assert 'not useful-gradient loss' in b['definition']
