import pytest
import torch
from src.task.CmResidual.selective_finger_response import INDEPENDENT,PRIMITIVES,primitive_target,validate_dof_names
from src.task.CmResidual.finger_preload import preload_target


def legal_base():
    base = torch.full((8,18),.1)
    for parent,children in {6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}.items():
        for child,ratio in children:
            base[:,child] = base[:,parent]*ratio
    return base


def test_selective_control_does_not_change_other_fingers_or_wrist():
    base=legal_base();lo=torch.zeros(18);hi=torch.full((18,),2.)
    target=primitive_target(base,torch.tensor(PRIMITIVES),lo,hi)
    assert torch.equal(target[:,:6],base[:,:6])
    assert torch.equal(target[0],base[0])
    assert torch.equal(target[1],preload_target(base[1:2],.15,lo,hi)[0])
    affected=((6,7),(8,9),(10,11),(12,13),(15,16,17),(14,))
    for row,chain in enumerate(affected,2):
        assert torch.equal(target[row,list(set(range(18))-set(chain))],base[row,list(set(range(18))-set(chain))])
        assert target[row,chain[0]]>base[row,chain[0]]


def test_actual_adjustment_records_dependent_limit_saturation():
    base=legal_base();lo=torch.zeros(18);hi=torch.full((18,),2.)
    hi[7]=.12
    target=primitive_target(base,torch.tensor(PRIMITIVES),lo,hi)
    actual=target[:,INDEPENDENT]-base[:,INDEPENDENT]
    assert 0<float(actual[2,0])<.15
    assert abs(float(target[2,7])-.12)<1e-6
    assert abs(float(target[2,7]-target[2,6]*1.05))<1e-6
    assert torch.equal(actual[2,1:],torch.zeros(5))


def test_dof_mapping_checks_actual_pair_identity():
    names=['joint'+str(i) for i in range(1,7)]
    for finger in ('index','middle','pinky','ring'):
        names += [finger+'_proximal_joint',finger+'_intermediate_joint']
    names += ['thumb_proximal_yaw_joint','thumb_proximal_pitch_joint','thumb_intermediate_joint','thumb_distal_joint']
    validate_dof_names(names)
    names[7]='middle_intermediate_joint'
    with pytest.raises(ValueError):
        validate_dof_names(names)
