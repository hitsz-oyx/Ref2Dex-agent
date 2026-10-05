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
