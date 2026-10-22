#!/usr/bin/env python3
"""Read-only A model-boundary probe and B/C counts from frozen prior receipts.

No census/replay, optimizer, backward, goal rewriting on training inputs, or
production patch. Model forward pre-hooks stop BEFORE any network computation.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.events import UNKNOWN_EVENT_POLICY
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime

GOAL_KEYS=('goal_object_delta','goal_object_points','object_goal_valid')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BoundaryCaptured(Exception):pass


class UnusedGeometry(nn.Module):
    """Constructor seam only; a model pre-hook always stops before geometry."""
    def forward(self,*args,**kwargs):
        raise AssertionError('no network/geometry computation authorized')


def capture_boundary(model,invoke):
    captured=[]
    def hook(module,args):
        batch=args[0]
        captured.append({key:batch[key].detach().cpu().clone() for key in GOAL_KEYS})
        raise BoundaryCaptured()
    handle=model.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():invoke()
    except BoundaryCaptured:
        pass
    finally:handle.remove()
    if len(captured)!=1:raise AssertionError('one actual model-boundary call required')
    return captured[0]


def describe(batch):
    return {key:dict(dtype=str(value.dtype),shape=list(value.shape),values=value.tolist(),
                     nonzero=int(torch.count_nonzero(value))) for key,value in batch.items()}


def equal_bits(left,right):
    return {key:bool(left[key].dtype==right[key].dtype and left[key].shape==right[key].shape
        and torch.equal(left[key].contiguous().view(torch.uint8),right[key].contiguous().view(torch.uint8)))
        for key in GOAL_KEYS}


def count_rules(mothers,layers,prior_counts,anchor):
    """Pure analysis of already persisted catalogs, no original-file scanning."""
    by_path={row['path']:row for row in layers}
    def metadata_keys(value):
        keys=set()
        if isinstance(value,dict):
            keys.update(value)
            for item in value.values():keys.update(metadata_keys(item))
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,dict):keys.update(metadata_keys(item))
        return keys
    families=[
        ('B0-hold-or-zero','Hold at reset pose / zero displacement',
         ('goal_rule_id','task_goal_rule','hold_goal_at_reset')),
        ('B1-observed-reset-plus-fixed-offset','Observed reset pose + prescribed displacement',
         ('goal_displacement','task_displacement','desired_displacement')),
        ('B2-approach-direction','Reset approach direction + prescribed distance / orientation',
         ('approach_distance','goal_direction','approach_direction','task_direction')),
        ('B3-geometry-contact-goal','Geometry + prescribed target region / offset / side',
         ('goal_contact_region','target_contact_region','goal_surface_query_id','target_patch','goal_offset'))]
    results=[]
    for identity,rule,fields in families:
        present=[]
        for mother in mothers:
            available=set()
            for path in mother.get('layer_paths',[]):
                layer=by_path.get(path,{})
                available.update(layer.get('fields',{}))
                for metadata in layer.get('metadata',{}).values():
                    available.update(metadata_keys(metadata))
            if any(field in available for field in fields):present.append(mother)
        results.append(dict(id=identity,rule=rule,parameter_field_aliases=list(fields),
            literal_parameter_candidate_mothers=len(present),
            literal_parameter_candidate_windows=sum(m['windows'] for m in present),
            certified_full_waypoint_mothers=0,certified_full_waypoint_windows=0,
            status='NOT_DEPLOY_LEGAL_UNDER_FROZEN_RECORDED_INTENT_CONTRACT',
            missing='rule/parameter identity and <=reset intent binding; geometry/hand observations do not supply this',
            implemented=False))
    native=[m for m in mothers if 'native_goal_position_evidence' in m]
    native_ids={m['sequence_id'] for m in native}
    results.append(dict(id='B4-native-Shadow-constructor',
        rule='Native nominal object_start_pose + constructor goal_displacement, restored goal_init_state',
        parameter_mothers=len(native),parameter_windows=sum(m['windows'] for m in native),
        mother_fraction=len(native)/prior_counts['mothers'],
        window_fraction=sum(m['windows'] for m in native)/prior_counts['total_windows'],
        positive_evidence='brief35 stored goal=(100,3), constant; owner-approved frozen collector 6141c34e constructor/reset proof',
        caution='Nominal constructor anchor is not measured post-reset pose; no observed-reset +0.2 rule is inferred.',
        certified_full_waypoint_mothers=0,certified_full_waypoint_windows=0,
        missing='same deployed scheduler conversion/path/frame/goal-type binding is not recorded; other mothers lack this native task rule',
        anchor_rows_with_native_parameters=sum(r['sequence_id'] in native_ids for r in anchor['rows']),
        status='RECORDED_NATIVE_POSITION_PARAMETERS_ONLY_BELOW_80_PERCENT',implemented=False))
    results.append(dict(id='B5-recorded-reference-plan',rule='Predeclared source trajectory / TaskSpec',
        certified_full_waypoint_windows=prior_counts['certified_legal'],certified_full_waypoint_mothers=0,
        status='UNPROVEN_FROM_BRIEF35_NOT_RESCANNED',implemented=False))
    return results


def run(output):
    if output.exists():raise FileExistsError('fresh evidence directory required')
    if output.resolve().parent!=(TASK/'docs/audit').resolve():raise ValueError('task-local audit output required')
    start=time.monotonic();free=shutil.disk_usage('/home2').free;estimate=16*2**20
    if free-estimate<20*2**30:raise ValueError('disk floor')
    torch.set_num_threads(2)
    curated_path=ROOT/'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json'
    recipe_path=ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'
    config_path=ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/config.json'
    prior_path=TASK/'docs/audit/brief35-source-role-20261023-r1/receipt.json'
    mothers_path=prior_path.parent/'mother-provenance.json'
    inventory_path=TASK/'docs/audit/brief32-raw-signal-inventory-20261023.json'
    cost_path=TASK/'docs/audit/brief32-cost-envelope-20261023.json'
    anchor_path=ROOT/'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r2/train-anchor.json'
    policy_path=TASK/'configs/tau_event_free_v1.json'
    policy_before=sha(policy_path)
    config=read(config_path);recipe=read(recipe_path)
    # Import the real training sampler. Same reader, Steps, DataLoader, collate,
    # CPU transfer and SharedWAMV6 class as train_pretrain.py; no main()/optimizer.
    entry_path=TASK/'tools/run/train_pretrain.py'
    spec=importlib.util.spec_from_file_location('brief36_training_entry',entry_path)
    entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    dataset=V6Windows(recipe_path,'train',geometry_gt_object=True,event_input_policy=UNKNOWN_EVENT_POLICY)
    steps=entry.Steps(dataset,8,170,0,1,1)
    loader=DataLoader(dataset,batch_sampler=steps,collate_fn=entry.collate,num_workers=0)
    batch={key:value.to('cpu',non_blocking=True) for key,value in next(iter(loader)).items()}
    indices=dataset.draw(8,170,step=1,rank=0)
    reference=collate([dataset[index] for index in indices])
    all_equal=all(torch.equal(value,reference[key]) for key,value in batch.items())
    if not all_equal:raise AssertionError('probe differs from production Steps/collate')
    model=SharedWAMV6(read(recipe['stats']),config,geometry=UnusedGeometry()).cpu()
    model.train()
    training=capture_boundary(model,lambda:model(batch))
    # The real curated reader rejects before a tensor exists. Do not manufacture
    # a training unknown batch or bypass its frozen goal gate for this probe.
    bound=V6Windows(recipe_path,'train',event_input_policy=UNKNOWN_EVENT_POLICY,curated_manifest=curated_path)
    try:bound[0]
    except ValueError as error:curated_error=str(error)
    else:raise AssertionError('curated goal gate changed; inspect before interpreting probe')
    # Real runtime predict/collate/model call. Live fixture is explicitly
    # synthetic, goal query IDs are copied from the captured training batch.
    runtime=object.__new__(OnlineV6Runtime)
    runtime.device=torch.device('cpu');runtime.config=config;runtime.model=model.eval()
    runtime.forecast_only=False;runtime.inference_batch=8
    runtime.calls=runtime.samples=0;runtime.seconds=0.;runtime.geometries={}
    codes=[]
    for i in range(8):
        code=f'probe-{i}@1';codes.append(code)
        runtime.geometries[code]=dict(surface=batch['xyz'][i].numpy(),
            normals=batch['features'][i,:,:3].numpy(),query=batch['start_object_points'][i].numpy())
    objects=np.tile([.25,.25,.75,0.,0.,0.,1.],(8,4,1))
    hands=np.tile([.25,.25,.75],(8,4,11,1))
    args=(objects,hands,None,[None]*8,codes)
    missing=capture_boundary(model,lambda:runtime.predict(*args))
    poses=np.tile(np.eye(4),(8,1,1));poses[:,:3,3]=[.25,.25,.75]
    constant=capture_boundary(model,lambda:runtime.predict(objects,hands,poses,[None]*8,codes))
    prior=read(prior_path);mothers=read(mothers_path)
    # Reuse persisted path lists from brief32 for field availability; no paths
    # are stat'ed or reopened in B, and no inventory tools are imported/called.
    cached=read(inventory_path)
    previous={r['sequence_id']:r for r in cached['sequence_inventory']}
    mothers=[dict(m,layer_paths=previous[m['sequence_id']]['layer_paths']) for m in mothers]
    rules=count_rules(mothers,cached['recording_layers'],prior['counts'],read(anchor_path))
    snapshots={'training_legacy':training,'deployment_missing':missing,'deployment_constant_current_pose':constant}
    equality=equal_bits(training,missing)
    zero=np.zeros_like(missing['goal_object_delta'].numpy())
    if not np.array_equal(missing['goal_object_delta'].numpy(),zero) or missing['object_goal_valid'].any():
        raise AssertionError('runtime unknown contract changed')
    if sha(policy_path)!=policy_before:raise AssertionError('frozen policy modified')
    result=dict(schema='ref2dex.brief36.goal-unlock-readiness.v1',
        experiment_id='P-20261023-goal-unlock-readiness',run_id=output.name,
        frozen_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        A=dict(status='NOT_READY_TRAINING_DATA_PATH_HAS_NO_UNKNOWN_GOAL_ENCODING',
            training_entry_probe='V6Windows -> production Steps -> DataLoader(collate) -> CPU transfer -> SharedWAMV6 forward_pre_hook',
            equivalence=dict(all_collated_fields_equal_to_direct_reader=all_equal,field_count=len(batch),
                             entrypoint_source=str(entry_path),entrypoint_sha256=sha(entry_path)),
            model=dict(class_name=type(model).__name__,config_sha256=sha(config_path),
                geometry_constructor_seam='UnusedGeometry; never called',
                interception='forward_pre_hook before any model/geometry computation',
                no_optimizer_created=True,no_backward=True,no_network_forward_computed=True),
            source_ids=batch['source_id'].tolist(),sample_ids=batch['sample_id'].tolist(),
            training_unknown_batches=0,training_legacy_boundary_batches=1,deployment_boundary_batches=2,
            curated_model_calls=0,curated_error=curated_error,
            actual_training_equals_deployment_missing_bits=equality,
            unknown_training_vs_unknown_deployment_equality=None,
            constant_vs_missing_bits=equal_bits(constant,missing),
            masked_queries=dict(training_legacy=int((~training['object_goal_valid']).sum()),
                deployment_missing=int((~missing['object_goal_valid']).sum()),
                deployment_constant=int((~constant['object_goal_valid']).sum()))),
        B=dict(rules=rules,threshold_mothers_percent=80,
            any_recorded_full_rule_over_threshold=False,full_waypoint_certified_windows=0,
            no_rule_implemented=True,
            candidate_parameter_coverage_from_shadow=dict(mothers=3,windows=30,denominator_mothers=1064,denominator_windows=6000)),
        C=dict(source=str(cost_path),sha256=sha(cost_path),existing_envelope=read(cost_path)['new_recording_scale'],
               estimates_recomputed=False,new_collection=False),
        historical_comparability=dict(if_A_chosen_goal_validity_bits_changed_train=6000*32,
            if_A_chosen_fixed_panel_rows_changed=len(read(anchor_path)['rows']),
            if_A_chosen_fixed_panel_query_bits_changed=len(read(anchor_path)['rows'])*32,
            tau_error_effect_mm=None,reason='No newly trained model or panel inference; changed conditioning invalidates input identity, not an estimated EPE effect'),
        input_identities={str(p):sha(p) for p in (prior_path,mothers_path,inventory_path,cost_path,anchor_path,curated_path,policy_path)},
        prior_censuses_rerun=False,training_updates=0,gpu_used=False,
        production_modified=False,goal_policy_enabled=False,filesystem_free_before_bytes=free,
        write_estimate_bytes=estimate,write_hard_limit_bytes=2**30,elapsed_s=time.monotonic()-start,
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__))
    output.mkdir()
    for name,value in [('receipt.json',result),('boundary-tensors.json',{name:describe(value) for name,value in snapshots.items()})]:
        with (output/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    np.savez(output/'boundary-tensors.npz',**{name+'__'+key:value.numpy()
        for name,batch in snapshots.items() for key,value in batch.items()})
    actual=sum(p.stat().st_size for p in output.iterdir())
    if actual>estimate:raise ValueError('audit exceeded estimated write bound')
    print(json.dumps(dict(A=result['A']['status'],equality=equality,
        masked_queries=result['A']['masked_queries'],B_native_parameter_mothers=3,
        B_full_waypoint_windows=0,bytes=actual,elapsed_s=result['elapsed_s'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
