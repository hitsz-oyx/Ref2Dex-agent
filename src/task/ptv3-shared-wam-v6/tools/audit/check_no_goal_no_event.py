#!/usr/bin/env python3
"""brief37 minimal admission checks; no optimizer, GPU or provenance rescan."""
import argparse, hashlib, importlib.util, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
TASK=Path(__file__).resolve().parents[2]; ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.goal_policy import UNKNOWN_GOAL_POLICY,GOAL_KEYS,apply_goal_input_policy,assert_unknown_inputs
from ptv3_shared_wam_v6.events import UNKNOWN_EVENT_POLICY
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and np.asarray(a).tobytes()==np.asarray(b).tobytes()

def run(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.monotonic()
    before=sha(TASK/'configs/tau_event_free_v1.json')
    prior=read(TASK/'docs/audit/brief36-goal-unlock-20261023-r1/receipt.json')
    helpers=load('brief36_helpers',TASK/'tools/audit/audit_goal_unlock_readiness.py')
    entry=load('actual_training',TASK/'tools/run/train_pretrain.py')
    c=read(TASK/'configs/tau_no_goal_no_event_train_v1.json'); p=read(c['condition_recipe'])
    recipe=ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'
    options=dict(geometry_gt_object=True,event_input_policy=UNKNOWN_EVENT_POLICY)
    ds=V6Windows(recipe,'train',goal_input_policy=UNKNOWN_GOAL_POLICY,curated_manifest=c['curated_manifest'],**options)
    old=V6Windows(recipe,'train',**options)
    lookup={tuple(map(int,row)):i for i,row in enumerate(old.index)}
    excluded=set(GOAL_KEYS); counts=Counter();changed=Counter();valid_windows=Counter()
    examples={}; input_keys=set()
    # Same 6000 tuples, grouped only for bounded cache IO; sampler/order is NOT changed.
    for n,i in enumerate(sorted(range(len(ds)),key=lambda i:tuple(ds.index[i]))):
        sample=ds[i];raw=old[lookup[tuple(map(int,ds.index[i]))]]
        for k in sample:
            if k not in excluded and not bits(sample[k],raw[k]):changed[k]+=1
        for prefix,s in [('old',raw),('A',sample)]:
            hand=s['trajectory_valid']&s['start_hand_valid'][None]
            scene=s['scene_valid']&s['object_valid'][:,None,None]
            contact=s['interaction_valid']; part=contact&(s['contact_target']>.25)
            first=np.zeros(24,bool);first[:8]=True
            cw=(contact&(s['contact_target']>.25)).any((0,2))
            masks=dict(hand=hand,scene=scene,contact=contact,part=part,first8=hand&first[:,None,None],contact_window=hand&cw[:,None,None])
            for k,v in masks.items():counts[prefix+'_'+k]+=int(v.sum())
            valid_windows[prefix+'_hand']+=bool(hand.any());valid_windows[prefix+'_object']+=bool(scene.any())
            counts[prefix+'_geometry_future_valid']+=int(np.isfinite(s['geometry_object_future_gt']).all())
        if np.count_nonzero(sample['object_goal_valid']) or any(np.count_nonzero(sample[k]) for k in GOAL_KEYS):raise AssertionError('non-unknown reader')
        examples.setdefault(int(sample['source_id']),(i,sample,raw))
        if n%1000==0:print(json.dumps(dict(windows=n,elapsed_s=time.monotonic()-start)),flush=True)
    assert len(ds)==6000 and not changed
    assert all(counts['old_'+k]==counts['A_'+k] for k in ('hand','scene','contact','part','first8','contact_window','geometry_future_valid'))
    assert valid_windows['A_hand']==valid_windows['old_hand']==6000
    selected=list(examples.values()); samples=[x[1] for x in selected]
    batch=collate(samples)
    model=SharedWAMV6(read(read(recipe)['stats']),c,geometry=helpers.UnusedGeometry()).cpu().train()
    train=helpers.capture_boundary(model,lambda:model(batch)); assert_unknown_inputs(batch)
    # Actual production Steps/DataLoader path in addition to deterministic per-source examples.
    actual=next(iter(DataLoader(ds,batch_sampler=entry.Steps(ds,8,170,0,1,1),collate_fn=entry.collate,num_workers=0)))
    actual_capture=helpers.capture_boundary(model,lambda:model(actual)); assert_unknown_inputs(actual)
    runtime=object.__new__(OnlineV6Runtime);runtime.config=c;runtime.device=torch.device('cpu');runtime.model=model.eval()
    runtime.forecast_only=False;runtime.inference_batch=len(samples);runtime.calls=runtime.samples=0;runtime.seconds=0.;runtime.geometries={}
    codes=[]
    for i in range(len(samples)):
        key=f'boundary-{i}@1';codes.append(key);runtime.geometries[key]=dict(surface=batch['xyz'][i].numpy(),normals=batch['features'][i,:,:3].numpy(),query=batch['start_object_points'][i].numpy())
    b=len(samples);objects=np.tile([.25,.25,.75,0,0,0,1.],(b,4,1));hands=np.tile([.25,.25,.75],(b,4,11,1))
    goals=np.tile(np.eye(4),(b,1,1));goals[:,:3,3]=[.8,.3,1.2]
    events=[dict(event_type=2,event_delta_steps=15.,event_valid=True,event_hz=30) for _ in samples]
    deployment=helpers.capture_boundary(model,lambda:runtime.predict(objects,hands,goals,events,codes))
    missing=helpers.capture_boundary(model,lambda:runtime.predict(objects,hands,None,[None]*b,codes))
    equality=helpers.equal_bits(train,deployment);assert all(equality.values()) and all(helpers.equal_bits(deployment,missing).values())
    # Adversarial old goals cannot re-enter any other field: full batch byte equality.
    polluted=dict(batch)
    for key in GOAL_KEYS:polluted[key]=torch.ones_like(batch[key])
    sanitized=apply_goal_input_policy(polluted,UNKNOWN_GOAL_POLICY)
    assert all(torch.equal(batch[k].contiguous().view(torch.uint8),sanitized[k].contiguous().view(torch.uint8)) for k in batch)
    # Invoke existing deployment state machine fixtures read-only, no physical rollout.
    sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
    release=load('existing_release_checks',ROOT/'src/task/wam-rl-policy-v3/tests/test_terminal_release_timing.py')
    names=[n for n in dir(release) if n.startswith('test_')]
    for name in names:getattr(release,name)()
    poses,spec,points=release.fixture();packet_before=json.dumps(spec.packet(),sort_keys=True)
    scheduler=release.TaskScheduler(spec,points)
    d=scheduler.update(poses[0],0.,release.InteractionFeedback(grasp_established=True,stable=True))
    model_condition=dict(batch);apply_goal_input_policy(model_condition,UNKNOWN_GOAL_POLICY)
    assert json.dumps(spec.packet(),sort_keys=True)==packet_before and spec.terminal_mode.value=='RELEASE'
    assert before==sha(TASK/'configs/tau_event_free_v1.json')
    np.savez(out/'model-boundary.npz',**{side+'__'+k:v.numpy() for side,val in [('train',train),('runtime',deployment),('steps',actual_capture)] for k,v in val.items()})
    j=dict(schema='ref2dex.brief37.minimal-checks.v1',checks={'1':'PASS','2':'PASS','3':'PASS'},prior_brief36=prior['frozen_head'],prior_censuses_rerun=False,
        windows=len(ds),source_examples=[dict(source_id=int(s['source_id']),sample_id=s['sample_id'].tolist()) for s in samples],
        train_runtime_goal_bit_equal=equality,runtime_valid_goal_vs_missing_bit_equal=helpers.equal_bits(deployment,missing),
        boundary=helpers.describe(train),steps_boundary=helpers.describe(actual_capture),
        all_non_goal_fields_unchanged_windows=6000,changed_non_goal_fields=dict(changed),supervision_denominators=dict(counts),valid_supervision_windows=dict(valid_windows),
        full_batch_equal_after_adversarial_goal_replacement=True,release_tests_passed=names,task_spec_unchanged=True,terminal_mode=spec.terminal_mode.value,
        task_code_sha256={str(f):sha(f) for f in (ROOT/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/task_scheduler.py',ROOT/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/executor_contract.py')},
        old_recipe_sha256=before,condition_recipe_sha256=sha(c['condition_recipe']),data_identity=p['data_identity'],elapsed_s=time.monotonic()-start,training_updates=0,gpu_used=False)
    (out/'checks.json').write_text(json.dumps(j,indent=2)+'\n'); print(json.dumps(dict(checks=j['checks'],windows=len(ds),elapsed_s=j['elapsed_s'])),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);run(a.parse_args().output)
