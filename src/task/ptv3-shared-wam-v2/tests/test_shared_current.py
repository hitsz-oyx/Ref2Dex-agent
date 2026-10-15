import copy
import pytest
import torch
from ptv3_shared_wam_v2.shared_actor import SharedQueryActor


def test_adapter_starts_at_original_function_and_only_reads_current_when_unmasked():
    torch.manual_seed(11)
    old=SharedQueryActor('shared',48,16)
    torch.nn.init.normal_(old.head.weight,std=.03)
    torch.manual_seed(12)
    a=SharedQueryActor('shared',48,16,current_observation='masked')
    b=copy.deepcopy(a);b.current_observation='delta0'
    for h in (a,b):h.load_state_dict(old.state_dict(),strict=False)
    state=torch.randn(2,87);shared=torch.randn(2,7,48);valid=torch.ones(2,7,dtype=torch.bool)
    current=torch.randn(2,721)
    original=old(state,shared=shared,shared_valid=valid)
    for h in (a,b):
        torch.testing.assert_close(h(state,shared=shared,shared_valid=valid,current_obs=current),original)
        h.current_mean.fill_(3);h.current_scale.fill_(2)
        torch.nn.init.normal_(h.current_encoder[-1].weight,std=.03)
    p=a(state,shared=shared,shared_valid=valid,current_obs=current)
    torch.testing.assert_close(p,a(state,shared=shared,shared_valid=valid,current_obs=current+2))
    q=b(state,shared=shared,shared_valid=valid,current_obs=current)
    assert not torch.allclose(q,b(state,shared=shared,shared_valid=valid,current_obs=current+2))
    q.sum().backward()
    assert b.current_encoder[0].weight.grad.abs().sum()>0
    assert b.memory_encoder[0].weight.grad.abs().sum()>0
    with pytest.raises(ValueError):b(state,shared=shared,shared_valid=valid)


def test_current_cannot_silently_enter_other_interfaces():
    with pytest.raises(ValueError):SharedQueryActor('pred_tau',48,16,current_observation='delta0')


def test_weight_only_parent_may_omit_only_new_branch(tmp_path):
    from ptv3_shared_wam_v2.current_initialization import initialize_from_robot
    old=torch.nn.Module();old.retargeter=SharedQueryActor('shared',48,16)
    new=torch.nn.Module();new.retargeter=SharedQueryActor('shared',48,16,current_observation='delta0')
    config=dict(schema='test',action_interface='shared')
    packet=dict(schema='test',stage='robot',step=7000,stats={'scale':1},config=config,
        provenance=dict(engineering_only=False,source=str(tmp_path)),model=old.state_dict())
    initialize_from_robot(new,packet,config,{'scale':1},tmp_path)
    for k,v in old.state_dict().items():torch.testing.assert_close(new.state_dict()[k],v)
    packet['model'].pop('retargeter.head.bias')
    with pytest.raises(ValueError,match='Only new'):initialize_from_robot(new,packet,config,{'scale':1},tmp_path)


def test_current_statistics_exclude_unadmitted_frames_and_dev():
    import numpy as np
    from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows
    dataset=object.__new__(V2RLRobotWindows)
    dataset.split='train';dataset.include_current_observation=True
    dataset.keys=[(0,0),(2,0),(1,1)]
    values=[np.tile(np.array([1.,999.,3.])[:,None],(1,721)),np.tile(np.array([999.,5.,999.])[:,None],(1,721))]
    dataset.sequence=lambda i:({'native_current_obs':values[i]},None)
    mean,scale=dataset.current_observation_statistics()
    np.testing.assert_allclose(mean,3.)
    np.testing.assert_allclose(scale,np.std([1.,3.,5.]),rtol=1e-6)
    dataset.split='val'
    with pytest.raises(ValueError):dataset.current_observation_statistics()
