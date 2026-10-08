"""Prepare independent ref4 S/P/M windows; never promote old audit-only data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within,K
from consequence_evaluator.data import object_effect,interaction_future,validate_rigid
from consequence_evaluator.value_outcomes import (RAW_SCHEMA,DATA_SCHEMA,RULE,PARAMETERS,
                                                  task_trace,validate_plan_execution,label_window,preference)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--seed',type=int,default=244)
    a=p.parse_args()
    out=a.output.resolve()
    if (out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator')
            or not 1<=len(a.source)<=3 or not 1<=a.stride<=K or not 100<=a.seed<=299):
        p.error('fresh task-owned output, <=3sources and fixed bounded stride/Probe seed required')
    started=time.monotonic();frozen={str(Path(__file__).resolve()):sha(__file__)}
    for name in ('value_outcomes.py','data.py','contracts.py'):
        path=TASK/'src/consequence_evaluator'/name;frozen[str(path)]=sha(path)
    arrays={name:[] for name in ('history','action','effect','interaction','current_object','current_hand',
            'success','progress','progress_mask','stage','progress_summary','margin','episode',
            'split','split_group','task','expert','motion','phase','tick')}
    seen=set();groups={};episode_stats=[];contracts=[]
    for source in a.source:
        source=source.resolve();path=source/'manifest.json';m=json.loads(path.read_text())
        if (m.get('schema')!=RAW_SCHEMA or m.get('status')!='COMPLETED'
                or m.get('value_outcomes') is not True or m.get('audit_only') is not False
                or m.get('training_allowed') is not True or m.get('task_success_rule')!=RULE
                or m.get('horizon')!=K or m.get('execution_horizon')!=K
                or m.get('rollout_kind')!='continuous' or m.get('fps')!=30 or m.get('units')!='m'):
            raise ValueError('only completed independent full-reference value sources are eligible')
        frozen[str(path)]=sha(path);contracts.append(m['history_contract'])
        if any(sha(key)!=value for key,value in m['sources'].items()):
            raise ValueError('source controller/reference/implementation drift')
        frozen.update(m['sources'])
        for record in m['episodes']:
            if time.monotonic()-started>120:
                raise TimeoutError('fixed120s label preparation budget')
            identity=record['episode'];group=record['split_group'];split=record['split']
            if identity in seen or split not in ('train','val','test') or groups.setdefault(group,split)!=split:
                raise ValueError('duplicated source episode or cross-split seed-group leakage')
            seen.add(identity)
            files=[source/record['path'],source/record['diagnostics']]
            if any(not is_within(path.resolve(),source) for path in files):
                raise ValueError('episode sidecars escape the frozen source')
            for path,key in zip(files,('sha256','diagnostics_sha256')):
                if sha(path)!=record[key]:
                    raise ValueError('episode/diagnostic hash mismatch')
                frozen[str(path)]=record[key]
            with np.load(files[0],allow_pickle=False) as data:
                packet={key:data[key] for key in data.files}
            with np.load(files[1],allow_pickle=False) as data:
                diagnostics={key:data[key] for key in data.files}
            steps=len(packet['action'])
            if (steps!=record['steps'] or packet['history'].shape!=(steps+1,*contracts[-1]['shape'])
                    or not np.isfinite(packet['history']).all()
                    or not np.allclose(packet['timestamps'],np.arange(steps+1)/30,atol=1e-9,rtol=0)):
                raise ValueError('history/full30Hz source clock mismatch')
            validate_rigid(packet['object_pose']);validate_rigid(diagnostics['reference_object_pose'])
            validate_plan_execution(packet,record)
            trace=task_trace(packet,diagnostics)
            ticks=set(range(0,steps-K+1,a.stride))
            if record['perturbation_tick']>=0:
                ticks.add(record['perturbation_tick'])
            count=0
            for tick in sorted(ticks):
                if not packet['plan_known'][tick]:
                    continue
                label=label_window(trace,tick)
                if label is None:
                    continue
                stop=tick+K+1;pose=packet['object_pose'][tick:stop];hand=packet['hand_keypoints'][tick:stop]
                values=dict(history=packet['history'][tick],action=packet['residual_plan'][tick],
                            effect=object_effect(pose),interaction=interaction_future(pose,hand),
                            current_object=pose[0],current_hand=hand[0],episode=identity,split=split,
                            split_group=group,task=record['task'],expert=record['expert'],motion=record['motion'],
                            phase=packet['phase'][tick],tick=tick,**label)
                for key in arrays:
                    arrays[key].append(values[key])
                count+=1
            episode_stats.append(dict(episode=identity,split=split,success=int(trace['task_success']),
                windows=count,max_stage=int(trace['stage'].max()),place_start=trace['place_start'],
                max_held_frames=int(trace['held_run'].max()),final_support_gap_m=float(trace['support_gap'][-1]),
                final_settled=bool(trace['settled'][-1]),assigned_phase=record['assigned_phase']))
    if not arrays['history'] or any(c!=contracts[0] for c in contracts):
        raise ValueError('empty windows or incompatible history contracts')
    result={key:np.asarray(value) for key,value in arrays.items()}
    rng=np.random.default_rng(a.seed);pairs=[];pair_kind=[];coverage={}
    for split in ('train','val','test'):
        candidates=[];indices={}
        for i in np.flatnonzero(result['split']==split):
            indices.setdefault(str(result['episode'][i]),[]).append(int(i))
        names=sorted(indices)
        for x,name in enumerate(names):
            candidates.extend((name,other) for other in names[x+1:])
        rng.shuffle(candidates)
        for first,second in candidates:
            if sum(coverage.get(split,{}).values())>=256:
                break
            i=int(rng.choice(indices[first]));j=int(rng.choice(indices[second]))
            if any(result[key][i]!=result[key][j] for key in ('task','motion','expert')):
                continue
            left={key:result[key][i] for key in ('success','progress_summary','margin')}
            right={key:result[key][j] for key in left};order=preference(left,right)
            if not order:
                continue
            pairs.append([i,j] if order>0 else [j,i])
            kind='success' if left['success']!=right['success'] else 'progress' if abs(left['progress_summary']-right['progress_summary'])>PARAMETERS['preference_progress_deadzone'] else 'margin'
            pair_kind.append(kind);counts=coverage.setdefault(split,{})
            counts[kind]=counts.get(kind,0)+1
    result['pairs']=np.asarray(pairs,np.int64).reshape(-1,2)
    result['pair_annotation']=np.asarray(pair_kind,dtype='U16')
    counts={}
    for split in ('train','val','test'):
        rows=[row for row in episode_stats if row['split']==split]
        counts[split]=dict(episodes=len(rows),success_episodes=sum(row['success'] for row in rows),
                          failure_episodes=sum(not row['success'] for row in rows),
                          windows=int((result['split']==split).sum()),preferences=sum(coverage.get(split,{}).values()))
    ready=all(counts[split]['success_episodes']>=minimum and counts[split]['failure_episodes']>=minimum
              for split,minimum in (('train',4),('val',2),('test',2)))
    if any(sha(path)!=value for path,value in frozen.items()):
        raise ValueError('label/source drift while preparing')
    out.mkdir(parents=True);np.savez_compressed(out/'windows.npz',**result)
    manifest=dict(schema=DATA_SCHEMA,status='COMPLETED',task='consequence-evaluator',run_id=out.name,
        training_allowed=ready,readiness_reason='requires independent train/val/test groups and at least4/2/2unique successful and failed episodes',
        history_contract=contracts[0],horizon=K,execution_horizon=K,fps=30,units='m',
        source_inputs=frozen,windows_sha256=sha(out/'windows.npz'),label_rule=RULE,label_parameters=PARAMETERS,
        source_schema=RAW_SCHEMA,counts=counts,preference_counts=coverage,episode_audit=episode_stats,
        preference_scope='same split/task/reference/controller; cross-episode S/P/M; no H matching',
        outcome_scope='whole-task factual outcome under fixed controller including completed past stages',
        main_label_force_proxy_used=False,elapsed_s=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:manifest[key] for key in ('status','training_allowed','counts','elapsed_s')},indent=2))


if __name__=='__main__':
    main()
