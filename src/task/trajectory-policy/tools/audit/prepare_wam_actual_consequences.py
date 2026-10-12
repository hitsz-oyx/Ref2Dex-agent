"""Prepare an engineering-only native actual-consequence view, without training.

The view points at one authenticated source; it stores indices and eight small
audit samples, never a duplicated full future corpus. Scientific world training
needs a separate decision protocol. Source actor/mixed-future denials remain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src')]
from trajectory_policy.wam_consequence import (EXECUTORS, actual_consequence_sample,
                                               future_control_metadata)
from trajectory_policy.wam_shadow import NATIVE_TO_WAM

SOURCE_SHA = '27da16275bdbc2c482ac6bfa164bf206dba4781cebc0c6b07c9a01d180329102'
ROLES = ['teacher', 'gt_tau_R', 'wam_R', 'wam_command_replay_then_expert_tick32']
FIXED = [(0,0),(0,2),(16,1),(16,3),(32,2),(32,3),(168,0),(168,3)]
SCENE_KEYS = ('xyz','features','scene_object','points','radius','object_features',
    'start_hand','start_hand_history','start_hand_valid','start_object_points',
    'goal_object_points','object_goal_valid','goal_object_slot')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    native,out=args.native.resolve(),args.output.resolve()
    owner=ROOT/'outputs/trajectory-policy'
    if native!=owner/'wam-ref2-execution-aggcollect-20261012-r1/native':
        raise ValueError('only the fixed192/t32 source is admitted')
    if out.exists() or out.parent!=owner or not out.name.startswith('wam-ref2-'):
        raise ValueError('fresh task-owned wam-ref2-* output required')
    started=time.monotonic(); previous=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame):
        raise TimeoutError('60s engineering preparation bound')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(60)
    manifest=None
    try:
        evidence={}
        def bind(path,expected=None):
            path=Path(path).resolve(); digest=sha(path)
            if expected is not None and digest!=expected:
                raise ValueError('source identity differs: '+str(path))
            evidence[str(path)]=digest
            return digest
        def read(path):
            bind(path);return json.loads(Path(path).read_text())
        source=read(native/'manifest.json'); audit=read(native/'root_recovery_audit.json')
        source_result=read(native/'result.json'); bind(native/'trajectory.npz',SOURCE_SHA)
        if (source.get('status')!='COMPLETED' or source.get('controls')!=192 or source.get('takeover_tick')!=32
                or source.get('roles')!=ROLES or source.get('dt')!=1/30
                or source.get('partial_episode') is not True or source.get('trajectory_sha256')!=SOURCE_SHA
                or source.get('actor_future_hand_training_allowed') is not False
                or source.get('mixed_policy_future_hand_training_allowed') is not False
                or source.get('expert_PD_training_allowed') is not True
                or audit.get('pass') is not True or audit.get('check_count')!=3468
                or audit.get('checks_max')!=0 or audit.get('all_arrays_finite') is not True
                or audit.get('native_data_sha256')!=SOURCE_SHA or audit.get('controls')!=192
                or audit.get('measured_states')!=193 or audit.get('takeover_tick')!=32
                or source_result.get('status')!='COMPLETED'):
            raise ValueError('fixed source/root3468/actor-denial identity required')
        with np.load(native/'trajectory.npz',allow_pickle=False) as stream:
            data={k:stream[k].copy() for k in ('obj','hand','action','applied','expert_applied')}
        if (data['obj'].shape!=(193,4,4,4) or data['hand'].shape!=(193,4,11,3)
                or data['action'].shape!=(192,4,18) or data['applied'].shape!=(192,4,18)
                or data['expert_applied'].shape!=(192,2) or data['expert_applied'].dtype!=bool
                or any(not np.isfinite(v).all() for v in data.values())
                or not np.array_equal(data['action'],data['applied'])
                or not np.array_equal(data['expert_applied'],np.c_[np.zeros(192,bool),np.arange(192)>=32])):
            raise ValueError('complete finite measured states and actual control provenance required')
        rotation=data['obj'][...,:3,:3]
        if (not np.allclose(rotation.swapaxes(-1,-2)@rotation,np.eye(3),atol=1e-5)
                or not np.allclose(np.linalg.det(rotation),1.,atol=1e-5)
                or not np.allclose(data['obj'][...,3,:],[0,0,0,1],atol=1e-6)):
            raise ValueError('all actual object states must be SE3')
        reference_path=owner/'competent-reference-20261011-r1/reference.npz'
        canonical_path=ROOT/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz'
        source_inputs=source['input_sha256']; server_inputs=source['server_identity']['input_sha256']
        bind(reference_path,source_inputs[str(reference_path)])
        bind(canonical_path,server_inputs[str(canonical_path)])
        with np.load(reference_path,allow_pickle=False) as stream:
            reference=stream['obj'].copy()
        if reference.shape!=(543,4,4):
            raise ValueError('original planned543-state object reference required')
        with np.load(canonical_path,allow_pickle=False) as stream:
            canonical={k:stream[k].copy() for k in stream.files}
        for path in (Path(__file__),TASK/'src/trajectory_policy/wam_consequence.py',
                     TASK/'src/trajectory_policy/wam_shadow.py'):
            bind(path)
        def verify_inputs():
            if any(sha(path)!=digest for path,digest in evidence.items()):
                raise ValueError('actual view input drift')
        def guard():
            if time.monotonic()-started>=60:
                raise TimeoutError('60s engineering preparation bound')
            paths=[p for p in owner.glob('wam-ref2-*') if p.is_dir()]
            paths += [p for p in (ROOT/'tmp').rglob('wam-*') if p.is_dir()
                      and not any(q.name.startswith('wam-') for q in p.parents if q!=ROOT/'tmp')]
            size=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in paths)
            if size>=100_000_000:
                raise RuntimeError('all ref2 artifacts/scratch100MB bound')
            return size
        verify_inputs(); guard()
        indices=np.asarray([(t,r) for t in range(169) for r in range(4)],np.int64)
        codes=np.stack([future_control_metadata(data['expert_applied'],int(t),int(r))['executor_code']
                        for t,r in indices]).astype(np.uint8)
        out.mkdir()
        manifest=dict(schema='ref2dex.wam-actual-consequence-view.v1',status='RUNNING',task='goal-point-wam',
            experiment_id='P-20261012-goal-point-wam-ref2-recovery-geometry',run_id=out.name,
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            engineering_only=True,scientific_training_allowed=False,training_allowed=False,
            world_label_scope='executed own-row hand motion -> actual rigid object future only',
            actor_expert_future_training_allowed=False,actor_future_hand_training_allowed=False,
            source_denials={k:source[k] for k in ('actor_future_hand_training_allowed','mixed_policy_future_hand_training_allowed')},
            source_training_scope=source['training_scope'],physics_override_allowed=False,
            source_path=str(native/'trajectory.npz'),source_sha256=SOURCE_SHA,
            root_audit_path=str(native/'root_recovery_audit.json'),root_audit_sha256=sha(native/'root_recovery_audit.json'),
            source_manifest_path=str(native/'manifest.json'),source_manifest_sha256=sha(native/'manifest.json'),
            planned_reference_path=str(reference_path),canonical_path=str(canonical_path),input_sha256=evidence,
            indices=indices.tolist(),windows=676,roles=ROLES,history=4,horizon=24,fps=30.,
            future_states='t+1..t+24',causal_controls='t..t+23',fixed_query_anchor='inverse(obj[t, own_role])',
            goal='known original reference.obj[min(t+24,542)]; never actual effect endpoint',
            action9D=['future_XYZ_current_anchor','future_minus_current_m','future_minus_previous_frame_m'],
            right_hand_semantic_order='NATIVE_TO_WAM',left_hand_valid=False,trajectory_valid=False,
            executor_codes=EXECUTORS,executor_index_path=str(out/'indices.npz'),
            limitations='actual hand-motion proxy, not native-command causal dynamics or RL qualification')
        write(out/'manifest.json',manifest)
        packed={}; checks=[]
        for sample_id,(tick,role) in enumerate(FIXED):
            goal=reference[min(tick+24,542)]
            sample=actual_consequence_sample(data['obj'],data['hand'],tick,role,goal,canonical,sample_id)
            C=np.linalg.inv(data['obj'][tick,role].astype(np.float64))
            now=data['hand'][tick,role,NATIVE_TO_WAM].astype(np.float64)
            future=data['hand'][tick+np.arange(1,25),role][:,NATIVE_TO_WAM].astype(np.float64)
            current=now@C[:3,:3].T+C[:3,3]
            transformed=future@C[:3,:3].T+C[:3,3]
            action=sample['action'].reshape(24,2,11,9)
            expected=np.concatenate((transformed,transformed-current,
                np.diff(np.concatenate((current[None],transformed)),axis=0)),axis=-1)
            action_error=float(np.abs(action[:,0]-expected).max())
            effect_error=float(np.abs(sample['effect'][0]-C@data['obj'][tick+np.arange(1,25),role]).max())
            changed_obj=data['obj'].copy(); changed_hand=data['hand'].copy()
            changed_obj[tick+1:tick+25,role,:3,3]+=[.05,-.03,.02]
            changed_hand[tick+1:tick+25,role]+=[.02,.01,-.03]
            changed=actual_consequence_sample(changed_obj,changed_hand,tick,role,goal,canonical,sample_id)
            current_unchanged=all(np.array_equal(sample[k],changed[k]) for k in SCENE_KEYS)
            metadata=future_control_metadata(data['expert_applied'],tick,role)
            valid=(action_error<=2e-6 and effect_error<=2e-6 and current_unchanged
                and all(np.isfinite(v).all() for v in sample.values())
                and sample['action_valid'].reshape(24,2,11)[:,0].all()
                and not sample['action_valid'].reshape(24,2,11)[:,1].any()
                and not action[:,1].any() and not sample['trajectory_valid'].any()
                and not sample['trajectory_target'].any()
                and np.array_equal(metadata['source_expert_applied'],data['expert_applied'][tick:tick+24])
                and not np.array_equal(sample['action'],changed['action'])
                and not np.array_equal(sample['effect'],changed['effect']))
            if not valid:
                raise ValueError('fixed engineering sample check failed: '+str((tick,role)))
            for key in SCENE_KEYS+('action','action_valid','effect','trajectory_valid'):
                packed[f's{sample_id}_{key}']=np.asarray(sample[key])
            for key,value in metadata.items():
                packed[f's{sample_id}_{key}']=value
            checks.append(dict(tick=tick,role=role,pass_=True,action9D_max_error_m=action_error,
                effect_max_error=effect_error,current_scene_known_goal_unchanged_after_future_mutation=current_unchanged,
                expert_controls=int(metadata['own_expert_applied'].sum())))
        verify_inputs()
        np.savez_compressed(out/'indices.npz',indices=indices,executor_codes=codes,
                            source_expert_applied=data['expert_applied'])
        np.savez_compressed(out/'audited_samples.npz',**packed)
        result=dict(status='COMPLETED',engineering_pass=True,engineering_only=True,
            scientific_training_allowed=False,actor_expert_future_training_allowed=False,
            physics_override_allowed=False,samples=checks,windows=676,elapsed_s=time.monotonic()-started,
            scope=manifest['world_label_scope'],new_data_is_view_not_full_future_copy=True)
        write(out/'result.json',result)
        manifest.update(status='COMPLETED',result_sha256=sha(out/'result.json'),
            indices_sha256=sha(out/'indices.npz'),audited_samples_sha256=sha(out/'audited_samples.npz'))
        write(out/'manifest.json',manifest)
        if int(subprocess.check_output(['du','-sb',str(out)],text=True).split()[0])>=1_000_000:
            raise RuntimeError('new view output exceeds1MB')
        guard();verify_inputs()
        print(json.dumps(result),flush=True)
    except BaseException as error:
        if manifest is not None:
            manifest.update(status='FAILED',error=repr(error));write(out/'manifest.json',manifest)
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    main()
