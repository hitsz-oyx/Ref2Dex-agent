"""User-authorized world changes must preserve learned state and stay explicit."""
import copy
import pytest
import torch
from ptv3_shared_wam_v6.contracts import SCHEMA
from ptv3_shared_wam_v6.topology import transition_checkpoint
from ptv3_shared_wam_v6.retained_contexts import validate_owner
from ptv3_shared_wam_v5.training_contract import execution_contract,validate_resume


def checkpoint():
    parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
    optimizer=torch.optim.AdamW([parameter],lr=1e-4)
    parameter.grad=torch.tensor([.3,.4]);optimizer.step()
    config=dict(schema=SCHEMA,batch_size=128,geometry_lr=1e-4,shared_lr=1e-4,horizon=24,updates=80000)
    return dict(schema=SCHEMA,world_size=4,config=config,execution_contract=execution_contract(config,4),
                data_identity='same-data',step=17000,model={'weight':parameter.detach().clone(),'normalizer.hand_std':torch.ones(24,3)},
                optimizer=optimizer.state_dict(),random_state=[{'rank':i} for i in range(4)],elapsed_s=16000.,selection={'score':1.28})


def test_explicit_world_change_preserves_adam_next_step_and_normalization():
    saved=checkpoint();target=dict(saved['config'],batch_size=64)
    with pytest.raises(ValueError,match='world-size'):validate_resume(saved,target,'same-data',3)
    changed=transition_checkpoint(saved,target,'same-data')
    validate_resume(changed,target,'same-data',3)
    assert changed['execution_contract']['effective_global_batch']==192
    assert changed['random_state']==saved['random_state'][:3]
    assert changed['model'] is saved['model'] and changed['optimizer'] is saved['optimizer']
    assert changed['step']==17000 and changed['elapsed_s']==saved['elapsed_s']
    assert saved['world_size']==4 and saved['config']['batch_size']==128
    values=[]
    for state in (saved,changed):
        p=torch.nn.Parameter(state['model']['weight'].clone());o=torch.optim.AdamW([p],lr=1e-4)
        o.load_state_dict(copy.deepcopy(state['optimizer']));p.grad=torch.tensor([.2,.7]);o.step();values.append(p.detach())
    torch.testing.assert_close(values[0],values[1],rtol=0,atol=0)


def test_transition_rejects_hidden_data_config_and_rng_changes():
    saved=checkpoint();target=dict(saved['config'],batch_size=64)
    with pytest.raises(ValueError):transition_checkpoint(saved,target,'different-data')
    with pytest.raises(ValueError,match='only local batch'):transition_checkpoint(saved,dict(target,shared_lr=2e-4),'same-data')
    with pytest.raises(ValueError,match='RNG'):transition_checkpoint(dict(saved,random_state=[]),target,'same-data')
    with pytest.raises(ValueError,match='world4'):transition_checkpoint(dict(saved,world_size=3),target,'same-data')


def test_only_exiting_confirmed_owned_contexts_can_be_shared():
    args=dict(pid=123,uid=1006,started=1000.,cmdline=b'',parent=1,pending=1<<14,expected_start=1000.,current_uid=1006)
    validate_owner(**args)
    for change in [dict(uid=2000),dict(started=2000.),dict(cmdline=b'active training'),dict(parent=50),dict(pending=0)]:
        with pytest.raises(ValueError,match='identity'):validate_owner(**dict(args,**change))
