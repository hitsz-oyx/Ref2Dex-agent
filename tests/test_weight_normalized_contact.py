import pytest
import torch
from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts


def test_gentle_contact_scales_with_body_weight_and_table_only_is_not_a_grasp_pair():
    mass=torch.tensor([.002593613,.02593613,.002593613]);weight=mass*9.81
    obj=torch.zeros(3,3);obj[:,2]=weight
    hand=torch.zeros(3,5,3);hand[:2,0,2]=weight[:2]*.5
    contact,ratio=weight_normalized_contacts(hand,obj,mass)
    assert contact.tolist()==[[True,True],[True,True],[False,True]]
    assert torch.allclose(ratio[0],ratio[1])
    assert (obj[0].norm()<.1) and contact[0,1]


def test_no_force_airborne_case_is_not_contact_and_invalid_mass_is_rejected():
    force=torch.zeros(2,3);hand=torch.zeros(2,5,3)
    contact,_=weight_normalized_contacts(hand,force,.002593613)
    assert not contact.any()
    with pytest.raises(ValueError):weight_normalized_contacts(hand,force,0.)
