"""Red-capable CPU contracts for audit; no physics or model training."""
import json
from pathlib import Path
import pytest
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
TASK=ROOT/'src/task/ptv3-shared-wam-v6'
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src')]


def test_missing_geometry_scale_is_rejected(tmp_path):
    from wam_rl_policy_v3.wam_v4_runtime import load_geometry
    cache=tmp_path/'cache';cache.mkdir()
    surface=np.random.default_rng(18).uniform(-.02,.02,(512,3)).astype(np.float32)
    geometry=cache/'wrong-scale.npz'
    np.savez(geometry,surface=surface,normals=np.tile([0.,0.,1.],(512,1)).astype(np.float32),query=surface[:32])
    row=dict(object_code='test/object',scale=1.,geometry=str(geometry))
    (cache/'manifest.json').write_text(json.dumps(dict(records=[row])))
    with pytest.raises(ValueError,match='scale'):
        load_geometry(cache,[dict(object_code=row['object_code'],scale=1.234567)])


def test_existing_geometry_scale_still_loads():
    from wam_rl_policy_v3.wam_v4_runtime import load_geometry
    cache=ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'
    row=json.loads((cache/'manifest.json').read_text())['records'][0]
    geometry=load_geometry(cache,[dict(object_code=row['object_code'],scale=row['scale'])])
    assert geometry[0][0].shape==(512,3)


def test_contracts_require_positive_and_negative_controls():
    from ptv3_shared_wam_v6.negative_result_invariants import basic_checks,verified
    checks=basic_checks()
    assert len(checks)>=14
    assert all(c['positive_control'] and c['mutation_detected'] and c['status']=='PASS' for c in checks)
    assert verified('blind checker',True,True,True,'undetected corruption')['status']=='FAIL'


def test_real_v6_forward_future_labels_and_invalid_slots():
    import torch
    from ptv3_shared_wam_v6.negative_result_invariants import v6_model_checks
    torch.set_num_threads(2)
    assert all(c['status']=='PASS' for c in v6_model_checks())


def test_voxel_point_axis_coordinates_and_world_transform():
    from ptv3_shared_wam_v6.negative_result_invariants import voxel_first_failure
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    sample=dict(xyz=np.array([[0.,0.,0.]],np.float32),features=np.zeros((1,18),np.float32),point_valid=np.ones(1,bool),
        start_hand=np.zeros((2,11,3),np.float32),start_hand_valid=np.zeros((2,11),bool))
    config=dict(voxel_origin_m=[-2.]*3,voxel_m=.01,hand_in_ptv3=True)
    pose=np.eye(4);pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];pose[:3,3]=[1,2,3]
    runtime=object.__new__(V4Runtime);runtime.config=config
    assert voxel_first_failure(sample,config,pose) is None
    assert runtime.workspace_valid([sample])[0]
    sample['start_hand_valid'][0,2]=True;sample['start_hand'][0,2]=[.1,-2.01,.3]
    result=voxel_first_failure(sample,config,pose)
    assert result['point']==('hand',0,2) and result['axis']=='y'
    assert result['voxel_index'][1]<0 and not runtime.workspace_valid([sample])[0]
    np.testing.assert_allclose(result['world_xyz'],[3.01,2.1,3.3],atol=1e-6)


def test_two_environment_interleaved_reset_catches_history_corruption():
    sys.path.insert(0,str(TASK/'tools/audit'))
    from audit_action_reset_cpu import reset_replay
    good=reset_replay();bad=reset_replay(mutate=True)
    assert good['status']=='PASS'
    assert [row['control_step'] for row in good['records']]==[2,4]
    assert bad['status']=='FAIL'


def test_evaluation_identity_rejects_every_drift_including_tiny_state_change():
    import copy
    from ptv3_shared_wam_v6.negative_result_invariants import evaluation_identity_matches
    initial=dict(seed=194,envs=2,hz=30,controls=300,state=dict(q=[[1.,2.]],tick=[0,0]))
    assert evaluation_identity_matches(initial,copy.deepcopy(initial))
    for field in ['seed','envs','hz','controls']:
        wrong=copy.deepcopy(initial);wrong[field]+=1
        assert not evaluation_identity_matches(initial,wrong)
    wrong=copy.deepcopy(initial);wrong['state']['q'][0][0]+=1e-8
    assert not evaluation_identity_matches(initial,wrong)
