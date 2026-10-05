import importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('flow_geometry_audit',ROOT/'tools/audit/audit_candidate_flow_geometry.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_diversity_exact_duplicate_and_single_direction():
    flow=np.zeros((2,7,5,3))
    flow[:,1:,:,0]=np.arange(1,7)[None,:,None]*.001
    report=audit.diversity(flow)
    assert report['baseline_contrast_ranks']==[1,1]
    assert abs(report['baseline_rms_mm_mean'][1]-1/np.sqrt(3))<1e-12
    assert report['pair_rms_mm_mean'][2][2]==0


def test_mimic_coupling_and_limit_preserving_step():
    drivers,matrix=audit.mimic_matrix()
    assert matrix[7,drivers.index(6)]==1.05
    assert matrix[16,drivers.index(15)]==.6
    q=np.zeros(18);step=matrix.numpy()[:,drivers.index(6)]
    upper=np.ones(18);upper[7]=.525
    assert abs(audit.coupled_step_scale(q,step,-np.ones(18),upper)-.5)<1e-12
    assert audit.coupled_step_scale(q+2,step,-np.ones(18),upper) is None


def test_hand_frame_removes_wrist_translation():
    tips=torch.zeros(1,2,5,3)
    pose=torch.zeros(1,2,7);pose[:,:,6]=1
    tips[:,1,:,2]=.01;pose[:,1,2]=.01
    assert torch.equal(audit.hand_local(tips,pose),torch.zeros_like(tips))


def test_continuous_wrist_is_not_limited_to_pi():
    urdf=audit.ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    lower,upper,continuous=audit.geometry_joint_limits(urdf)
    assert continuous
    q=((lower+upper)/2).numpy()
    q[continuous]=4*np.pi
    step=np.zeros(18);step[continuous]=1.
    assert audit.coupled_step_scale(q,step,lower.numpy(),upper.numpy())==1.
