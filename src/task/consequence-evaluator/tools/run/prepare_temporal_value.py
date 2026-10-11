"""Prepare ref4_3 weak labels without reading old dense S/P/M targets."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.data import sha
from consequence_evaluator.contracts import is_within,K
from consequence_evaluator.value_outcomes import task_trace,DATA_SCHEMA
from consequence_evaluator.temporal_value import SCHEMA,LABEL_RULE,episode_labels,relative_time_labels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); source=a.data.resolve();out=a.output.resolve();start=time.monotonic()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator'):
        raise ValueError('fresh task-owned data output required')
    m=json.loads((source/'manifest.json').read_text())
    if m['schema']!=DATA_SCHEMA or m['status']!='COMPLETED' or not m['training_allowed']:
        raise ValueError('completed training-eligible outcome source required')
    if sha(source/'windows.npz')!=m['windows_sha256']:
        raise ValueError('window identity drift')
    hashes=dict(m['source_inputs'])
    for path,digest in hashes.items():
        if sha(path)!=digest:
            raise ValueError('frozen source drift: '+path)
    for path in (source/'manifest.json',source/'windows.npz',Path(__file__).resolve(),
                 TASK/'src/consequence_evaluator/temporal_value.py'):
        hashes[str(path)]=sha(path)
    with np.load(source/'windows.npz',allow_pickle=False) as f:
        d={k:f[k] for k in ('history','action','effect','interaction','episode','split','split_group','tick')}
    future=np.concatenate((d.pop('effect')[:,:,:3,:].reshape(-1,K,12),
                           d.pop('interaction').reshape(-1,K,33)),axis=-1)
    outcomes={}; audit=[]; recoveries={}; lengths={}
    old={r['episode']:r for r in m['episode_audit']}
    raw_manifests=[Path(path) for path in m['source_inputs'] if path.endswith('/manifest.json')]
    for path in raw_manifests:
        raw=json.loads(path.read_text())
        if raw.get('schema')!='ref2dex.consequence-value.episodes.v1':
            continue
        for r in raw['episodes']:
            if time.monotonic()-start>120:
                raise TimeoutError('weak label preparation120s budget')
            name=r['episode']
            # Provenance also includes ancestor/calibration episodes, which
            # must be hashed but must not enter this fixed 192-episode split.
            if name not in old:
                continue
            if name in outcomes:
                raise ValueError('duplicate episode')
            with np.load(path.parent/r['path'],allow_pickle=False) as f:
                packet={k:f[k] for k in ('action','object_pose')}
            with np.load(path.parent/r['diagnostics'],allow_pickle=False) as f:
                diag={k:f[k] for k in f.files}
            trace=task_trace(packet,diag)
            if int(trace['task_success'])!=old[name]['success']:
                raise ValueError('old outcome audit mismatch')
            success,recovery=episode_labels(trace,diag['object_velocity'])
            outcomes[name]=success;recoveries[name]=recovery;lengths[name]=r['steps']
            audit.append(dict(episode=name,split=r['split'],old_success=old[name]['success'],
                success=success,recovery_state_count=int(recovery.sum())))
    if set(outcomes)!=set(d['episode']):
        raise ValueError('missing complete episode supervision')
    success=np.array([outcomes[e] for e in d['episode']],np.int64)
    length=np.array([lengths[e] for e in d['episode']],np.int64)
    uncertain=np.array([bool(outcomes[e] and recoveries[e][t:t+K+1].any())
                        for e,t in zip(d['episode'],d['tick'])])
    target=relative_time_labels(success,d['tick'],length).astype('float32')
    arrays=dict(**d,future=future,label=target,episode_success=success,length=length,
                supervised=~uncertain,recovery_uncertain=uncertain)
    counts={}
    for split in ('train','val','test'):
        rows=[v for v in audit if v['split']==split];selected=d['split']==split
        counts[split]=dict(episodes=len(rows),success=sum(v['success'] for v in rows),
            failure=sum(not v['success'] for v in rows),windows=int(selected.sum()),
            supervised=int((selected & ~uncertain).sum()),masked_recovery=int((selected & uncertain).sum()))
    for name in np.unique(d['episode']):
        if len(np.unique(d['split'][d['episode']==name]))!=1:
            raise ValueError('episode split leakage')
    if any(sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('inputs changed during weak-label preparation')
    out.mkdir(parents=True);np.savez_compressed(out/'windows.npz',**arrays)
    manifest=dict(schema=SCHEMA,status='COMPLETED',training_allowed=True,rule=LABEL_RULE,
        source_label_rule=m['label_rule'],history_contract=m['history_contract'],
        action_semantics='decision_known_requested_residual_plan24_with_frozen_feedback_controller',
        future_semantics='measured_object_effect12_and_object_frame_hand_points33',
        model_input_whitelist=['history','action','future'],target='sign(final_outcome)*24/(episode_length-tick)',
        positive_recovery_mask='any recovery/uncertain state within current plus24future states',
        failed_episode_policy='retain weak negative windows; not action imitation',
        counts=counts,episode_audit=audit,input_sha256=hashes,windows_sha256=sha(out/'windows.npz'),
        elapsed_s=time.monotonic()-start)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(counts=counts,changed_outcomes=[v for v in audit if v['success']!=v['old_success']],
                          elapsed_s=manifest['elapsed_s']),indent=2))


if __name__=='__main__':
    main()
