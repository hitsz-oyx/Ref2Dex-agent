from pathlib import Path
import sys
import torch
import pytest

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
from intervention import per_finger_residuals, PER_FINGER_ARM_NAMES, INDEPENDENT_FINGER_INDICES, decode_assignment
from src.task.CmResidual.dexplore_cm_geometry import native_joint_limits, dexplore_action_to_native_targets


def test_per_finger_isolates_six_drivers_including_yaw_with_matched_synergy():
    d=per_finger_residuals(.2)
    assert len(PER_FINGER_ARM_NAMES)==15 and d.shape==(15,18)
    for finger,index in enumerate(INDEPENDENT_FINGER_INDICES):
        assert d[1+finger*2].count_nonzero()==1 and d[1+finger*2,index]==.4
        assert torch.equal(d[2+finger*2],-d[1+finger*2])
    assert d[13,14]==0 and d[13].count_nonzero()==5
    assert torch.equal(d[14],-d[13]) and d[0].count_nonzero()==0
    arms,k=decode_assignment(torch.arange(15),[8],15)
    assert arms.tolist()==list(range(15)) and (k==8).all()
    with pytest.raises(ValueError): per_finger_residuals(.3)


def test_equal_driver_range_fraction_preserves_native_mimic_and_different_radians():
    lo,hi=native_joint_limits(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf','cpu')
    d=per_finger_residuals(.2); zero=torch.zeros_like(d)
    pd=dexplore_action_to_native_targets(d,zero,lo,hi)-dexplore_action_to_native_targets(zero,zero,lo,hi)
    followers={6:{7:1.05},8:{9:1.05},10:{11:1.05},12:{13:1.05},14:{},15:{16:.6,17:.8}}
    for finger,index in enumerate(INDEPENDENT_FINGER_INDICES):
        row=1+finger*2
        assert torch.allclose(pd[row,index]/(hi[index]-lo[index]),torch.tensor(.2))
        expected={index,*followers[index]}
        assert set(pd[row].nonzero().flatten().tolist())==expected
        for follower,ratio in followers[index].items():
            assert torch.allclose(pd[row,follower],ratio*pd[row,index],atol=1e-7)
        assert torch.allclose(pd[row+1],-pd[row],atol=2e-7)
    assert torch.allclose(pd[1,6],torch.tensor(.32),atol=1e-7)
    assert torch.allclose(pd[9,14],torch.tensor(.23),atol=1e-7)
    assert torch.allclose(pd[11,15],torch.tensor(.11),atol=1e-7)


def audit_modules():
    sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/tools/audit'))
    import audit_finger_amplitudes
    import probe_per_finger_control
    return audit_finger_amplitudes,probe_per_finger_control


def test_true_tip_audit_removes_rigid_hand_translation_and_rotation():
    audit,_=audit_modules()
    before_pose=torch.zeros(1,7); before_pose[:,6]=1
    after_pose=before_pose[:,None].repeat(1,32,1)
    after_pose[:,:,0]=1; after_pose[:,:,5:7]=torch.tensor([2**-.5,2**-.5])
    before_tips=torch.tensor([[[1.,0.,0.]]]).repeat(1,5,1)
    after_tips=torch.tensor([[[[1.,1.,0.]]]]).repeat(1,32,5,1)
    motion=audit.local_tip_motion(dict(before_hand_base_pose=before_pose,hand_base_pose=after_pose,
        before_fingertip_positions=before_tips,fingertip_positions=after_tips))
    assert torch.allclose(motion,torch.zeros_like(motion),atol=2e-7)


def test_attribution_requires_own_body_and_task_gain_and_repetition():
    import numpy as np
    _,probe=audit_modules()
    b=np.zeros((14,52)); t=np.ones(52)
    b[0,13]=.005; b[0,1]=.12; b[0,2]=-.15; t[[13,1,2]]=.05
    linked,useful=probe.candidates(b,t,[b,b])
    assert linked[0]['arm']==1 and linked[0]['direction']=='beneficial' and useful[0]['arm']==1
    no_own=b.copy(); no_own[0,13]=0; no_own[0,17]=.020
    assert probe.candidates(no_own,t,[no_own,no_own])==([],[])
    weak=b.copy(); weak[0,1]=-.01; weak[0,2]=.01
    assert probe.candidates(b,t,[b,weak])==([],[])
    harmful=b.copy(); harmful[0,1]=-.12; harmful[0,2]=.15
    linked,useful=probe.candidates(harmful,t,[harmful,harmful])
    assert linked[0]['direction']=='harmful' and not useful


def test_composite_control_cannot_rescue_single_finger_gate():
    import numpy as np
    _,probe=audit_modules()
    b=np.zeros((14,52)); t=np.full(52,.01)
    b[12,17]=.02; b[12,1]=.2; b[12,2]=-.2
    assert probe.candidates(b,t,[b,b])==([],[])


def test_audit_and_probe_cli_import_cleanly_without_pythonpath():
    import os,subprocess
    env=os.environ.copy(); env.pop('PYTHONPATH',None)
    for name in ('audit_finger_amplitudes.py','probe_per_finger_control.py'):
        result=subprocess.run([sys.executable,str(ROOT/'src/task/cm-interaction-oracle/tools/audit'/name),'--help'],
            cwd=ROOT,env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert '--dataset' in result.stdout
