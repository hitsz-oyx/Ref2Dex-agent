#!/usr/bin/env python3
"""Recompute fresh pilot admission, goal lineage and coverage from saved arrays.

Independent of the producer's counters, but intentionally shares the existing
admission predicate, instruction validator and source classifier.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(TASK.parent/'ptv3-shared-wam-v5/src'),
              str(TASK.parent/'wam-rl-policy-v3/src'),str(TASK/'tools/audit')]
from ptv3_shared_wam_v6.rigid_dataset import admission_reasons,geometry_class,pca_shape
from ptv3_shared_wam_v5.contracts import JOINT_NAMES,PART_NAMES,LABEL_SEMANTICS
from wam_rl_policy_v3.goal_channel import validate_instruction
from certify_waypoint_sources import classify,LEGAL
from curate_tau_training_set import phase_for


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):return {f.stem:np.load(f,allow_pickle=False) for f in p.glob('*.npy')}


def audit(out,check_only=False):
    from scipy.spatial.transform import Rotation
    from ptv3_shared_wam_v5.shadow import ShadowSurface
    from ptv3_shared_wam_v5.preparation import surface_contact
    started=time.monotonic();plan=read(out/'plan.json');summary=read(out/'audit-summary.json')
    surface=ShadowSurface(out/'assets/mjcf/open_ai_assets/hand/robot.xml')
    assert len(plan['jobs'])==96 and len({j['rollout_id'] for j in plan['jobs']})==96
    assets={r['object_code']:r for r in plan['objects']}
    verdicts=[json.loads(line) for line in (out/'window-verdicts.jsonl').read_text().splitlines()]
    by_rollout={j['rollout_id']:{} for j in plan['jobs']}
    for row in verdicts:
        target=by_rollout[row['rollout_id']]
        assert row['tick'] not in target
        target[row['tick']]=row
    counts=Counter(); admitted=certified=full=0; maximum_error=0.;native_controls=0
    for job in plan['jobs']:
        folder=out/job['rollout_id'];binding=read(folder/'reset-binding.json')
        asset=assets[job['object_code']];geo=dict(np.load(out/asset['registry'],allow_pickle=False))
        assert sha(out/asset['registry'])==asset['registry_sha256']
        assert geometry_class(pca_shape(geo['surface']))==job['geometry_class']
        raw=dict(np.load(folder/'raw.npz',allow_pickle=False));seq=arrays(folder/'sequence')
        contact=arrays(folder/'contact');event=arrays(folder/'events')
        assert len(raw['q'])==301 and len(raw['action'])==600
        native_controls+=len(raw['action'])
        assert np.array_equal(raw['native_progress'],np.arange(1,602,2))
        assert np.all(np.diff(raw['observed_at_ns'])>0)
        error=float(np.max(np.abs(raw['native_joint21']-seq['joints'][:,0])))
        maximum_error=max(maximum_error,error);assert error<=1e-4
        assert np.allclose(seq['object_poses'][:,0,:3,3],raw['object'][:,:3],rtol=0,atol=1e-7)
        assert np.allclose(seq['object_poses'][:,0,:3,:3],Rotation.from_quat(raw['object'][:,3:7]).as_matrix(),atol=1e-7)
        assert seq['joint_valid'][:,0].all() and not seq['joint_valid'][:,1].any()
        assert seq['surface_valid'][:,0].all() and not seq['surface_valid'][:,1].any()
        reconstructed,jv,vertices,sv,parts=surface.reconstruct(raw['q'],raw['hand_root'])
        assert np.array_equal(reconstructed,seq['joints']) and np.array_equal(jv,seq['joint_valid'])
        assert np.array_equal(parts,seq['surface_parts']) and np.array_equal(sv,seq['surface_valid'])
        assert np.array_equal((vertices-reconstructed[:,:,None,0]).astype('f2'),seq['surface_offsets'])
        for frame in (0,150,300):
            observed=seq['surface_offsets'][frame,0].astype('f4')+seq['joints'][frame,0,0]
            pose=seq['object_poses'][frame,0]
            query=geo['surface']@pose[:3,:3].T+pose[:3,3]
            c,p=surface_contact(observed,parts[0],query)
            assert np.array_equal(c,contact['contact'][frame,0]) and np.array_equal(p,contact['part'][frame,0])
        admission=read(folder/'admission.json')
        assert admission['joint_semantics']==list(JOINT_NAMES) and admission['part_semantics']==list(PART_NAMES)
        assert admission['label_semantics']==LABEL_SEMANTICS
        instruction=read(folder/'goal-channel/instruction.json');validate_instruction(instruction)
        assert instruction['reset']==binding['reset']
        intents=read(binding['intent_artifact'])
        intent=next(r for r in intents if r['target_execution_id']==binding['reset']['execution_id'])
        assert intent['object_code']==job['object_code'] and binding['job']==job
        assert intent['source_role']=='explicit_task_instruction'
        assert intent['source_execution_id']!=intent['target_execution_id']
        assert intent['available_at_ns']==binding['source_available_timestamp_ns']<binding['reset_timestamp_ns']
        assert binding['reset_timestamp_ns']<=int(raw['observed_at_ns'][0])
        assert sha(binding['intent_artifact'])==binding['source_artifact_sha256']
        assert np.array_equal(raw['goal_state'],np.broadcast_to(intent['goal_state'],raw['goal_state'].shape))
        assert binding['reset_target_pose_goal_restored'] and binding['reset_root_goal_restored']
        goal=np.eye(4);goal[:3,3]=intent['goal_state'][:3];goal[:3,:3]=Rotation.from_quat(intent['goal_state'][3:7]).as_matrix()
        assert np.array_equal(goal,instruction['task_goal_pose'])
        proof=dict(target_tuple_bound=True,source_role=intent['source_role'],
            source_execution_id=intent['source_execution_id'],target_execution_id=intent['target_execution_id'],
            available_before_reset=True,independent_of_target_rollout=True,artifact_digest_verified=True,
            reset_binding_verified=True,derivation_verified=True,goal_semantics=intent['goal_semantics'])
        assert classify(proof)==LEGAL
        repeated=read(folder/'repeated-initial-reset.json')
        assert repeated['before_first_physics_control'] and repeated['logical_reset']==binding['reset']
        assert binding['reset_timestamp_ns']<=repeated['at_ns']<=int(raw['observed_at_ns'][0])
        assert repeated['restored_goal_state']==intent['goal_state']
        evidence=dict(joint_semantics=list(JOINT_NAMES),part_semantics=list(PART_NAMES),label_semantics=LABEL_SEMANTICS,
            binding_verified=True,event_binding_verified=True,geometry_binding_verified=True)
        good=0
        for tick in range(3,277):
            reasons=admission_reasons(seq,geo,contact,event,evidence,tick,0)
            saved=by_rollout[job['rollout_id']][tick]
            assert reasons==saved['reasons'] and saved['admitted']==(not reasons)
            phase=phase_for(event,0,tick);assert saved['phase']==phase
            assert saved['goal_certification']==LEGAL
            if not reasons:
                good+=1;counts[phase,job['geometry_class'],4]+=1
        assert len(by_rollout[job['rollout_id']])==274
        admitted+=good;full+=good==274;certified+=1
    with (out/'coverage-before.csv').open() as f:before=list(csv.DictReader(f))
    with (out/'coverage-after.csv').open() as f:after=list(csv.DictReader(f))
    assert len(before)==len(after)
    for a,b in zip(before,after):
        key=(a['phase'],a['geometry_class'],int(a['source_id']))
        assert key==(b['phase'],b['geometry_class'],int(b['source_id']))
        assert int(b['windows'])==int(a['windows'])+counts[key]
    assert admitted==summary['admitted_windows'] and full==summary['fully_admitted_rollouts']
    assert certified==summary['certified_goal_rollouts'] and native_controls==57600
    assert len(np.load(out/'certified-candidate-tuples.npy'))==admitted
    result=dict(status='PASS',rollouts=96,output_ticks=28800,native_controls=native_controls,
        warmup_native_controls=96,fully_admitted_rollouts=full,certified_goal_rollouts=certified,
        admitted_windows=admitted,maximum_native21_vs_saved21_m=maximum_error,
        checked=['all saved windows through existing admission predicate','source hash and producer reset order',
                 'every native target reset restoration','goal instruction derivation','raw measured21 versus saved21',
                 'all saved surfaces and part IDs reconstructed from source asset','three contact/part frames per rollout recomputed',
                 'actual class and phase coverage','reset and native clock continuity'],elapsed_s=time.monotonic()-started)
    if not check_only:
        with (out/'independent-audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();audit(args.output.resolve(),args.check_only)
