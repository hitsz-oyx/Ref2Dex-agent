"""Owner-scoped GPU allocation and production event-free model boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ptv3_shared_wam_v6.events import UNKNOWN_EVENT_POLICY, apply_event_input_policy, event_features
from ptv3_shared_wam_v6.data import V6Windows
from ptv3_shared_wam_v6 import curated_training

TASK=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('brief34_preflight',TASK/'tools/audit/freeze_event_free_tau.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def policy():
    return json.loads((TASK/'configs/tau_event_free_v1.json').read_text())


def test_frozen_recipe_changes_only_owner_resource_authorization():
    frozen=json.loads(subprocess.check_output(['git','show',
        'ca640870:src/task/ptv3-shared-wam-v6/configs/tau_event_free_v1.json'],text=True))
    current=policy()
    frozen.pop('resource_policy');current.pop('resource_policy')
    assert current==frozen


@pytest.mark.parametrize('available,expected',[([4,5,6],True),([4,5],False),
    ([0,1,2,3,7],False),([4,4,5],False)])
def test_explicit_owner_pool_replaces_global_spare_count(available,expected):
    resources=dict(status='MEASURED',allocation_policy=policy()['resource_policy'],
        available_authorized_gpus=available,available_unreserved_gpus=[])
    assert audit.gpu_allocation_gate(resources,3) is expected


def test_undocumented_spare_waiver_is_rejected_and_disk_gate_remains():
    allocation=policy()['resource_policy']
    allocation['spare_exemption']['authorization']='implicit'
    assert not audit.gpu_allocation_gate(dict(status='MEASURED',allocation_policy=allocation,
        available_authorized_gpus=[4,5,6]),3)
    resources=dict(status='MEASURED',allocation_policy=policy()['resource_policy'],
        available_authorized_gpus=[4,5,6],filesystem_free_bytes=21*2**30)
    result=audit.launch_gates(dict(production_training_wired=True,
        production_runtime_policy_enforced=True,reader_windows=6000),
        dict(counts=dict(scheduler_eligible_windows=6000)),resources,2*2**30)
    assert result['gates']['gpu_allocation_available']
    assert not result['gates']['disk_available']
    assert result['status']=='BLOCKED_PREFLIGHT'


@pytest.mark.parametrize('forecast_only',[False,True])
def test_real_production_predict_uses_checkpoint_policy_for_both_paths(forecast_only):
    from test_v6_executor_wiring import runtime,inputs
    rt=runtime();rt.forecast_only=forecast_only
    rt.config['event_input_policy']=UNKNOWN_EVENT_POLICY
    with torch.no_grad():rt.model.event_goal.encoder[-1].bias.fill_(7.)
    event_batches=[];goals=[]
    hook=rt.model.event_goal.register_forward_pre_hook(lambda module,args:event_batches.append(args[0]))
    original=rt._trajectory
    def capture(batch):
        assert not batch['event_valid'].any()  # after collate, before model
        goals.append(batch['goal_object_delta'].clone())
        return original(batch)
    rt._trajectory=capture
    args=inputs()
    # Malformed scheduler packets cannot leak through the unknown-only policy.
    args[3][0].update(event_type=999,event_delta_steps=float('nan'),event_hz=60)
    result=rt.predict(*args)
    hook.remove()
    assert result.shape==(2,24,11,3)
    assert len(event_batches)==2
    for batch,goal in zip(event_batches,goals):
        assert not event_features(batch)[0].any()
        assert torch.equal(batch['goal_object_delta'],goal)
        with torch.inference_mode():assert not rt.model.event_goal(batch).any()


def test_common_mask_does_not_mutate_inputs_or_known_none_or_model_state_keys():
    batch=dict(event_type=torch.tensor([0,1,2]),event_delta_steps=torch.tensor([0.,8.,40.]),
        event_valid=torch.ones(3,dtype=torch.bool),event_hz=torch.full((3,),30),
        goal_object_delta=torch.ones((3,32,3)),contact_target=torch.ones((3,1,24,4)))
    masked=apply_event_input_policy(batch,UNKNOWN_EVENT_POLICY)
    assert batch['event_valid'].all() and not masked['event_valid'].any()
    assert masked['goal_object_delta'] is batch['goal_object_delta']
    assert masked['contact_target'] is batch['contact_target']
    assert apply_event_input_policy(batch,None) is batch
    with pytest.raises(ValueError,match='unsupported event input policy'):
        apply_event_input_policy(batch,'known-NONE')
    from test_v6_executor_wiring import runtime
    rt=runtime();keys=set(rt.model.state_dict())
    rt.config['event_input_policy']=UNKNOWN_EVENT_POLICY
    assert set(rt.model.state_dict())==keys


def test_frozen_index_and_sampler_bind_without_reading_future_goals(monkeypatch):
    rows=[dict(source_id=1,sequence_id=i//60,sequence_key=f's1-{i//60:05}',
        slot=0,tick=3+i%60,split='train',admitted=True,phase='approach') for i in range(6000)]
    dataset=SimpleNamespace(split='train',index=np.array([[r['sequence_id'],0,r['tick']] for r in rows]),
        meta=dict(sequences=[dict(key=f's1-{i:05}',source_id=1) for i in range(100)]))
    monkeypatch.setattr(curated_training,'load_manifest',lambda path:dict(windows=rows))
    curated_training.bind_curated_index(dataset,'unused-manifest.json')
    assert len(dataset.index)==6000
    draws=curated_training.draw_curated(dataset,96,170,step=1,rank=0)
    assert len(draws)==96 and all(0<=i<6000 and not drop for i,drop in draws)
    sequences=dataset.index[[i for i,drop in draws],0]
    assert np.unique(sequences,return_counts=True)[1].max()<=8


def test_missing_scheduler_binding_fails_before_reading_any_labels():
    dataset=object.__new__(V6Windows)
    dataset.curated_manifest=Path('frozen-curated.json')
    with pytest.raises(ValueError,match='refusing realized-endpoint fallback'):
        dataset[0]
