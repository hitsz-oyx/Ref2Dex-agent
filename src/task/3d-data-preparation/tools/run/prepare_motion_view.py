"""Quarantine simulator impulse tails, deriving immutable windows and TRAIN moments."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
TASK=Path(__file__).resolve().parents[2]
REPO=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.motion_quality import events,rejected_ticks,validate_policy
from pose_mesh_data.cleaning import canonical_sequence
from pose_mesh_data.statistics import Moments,sequence_moments,subtract_moments
from clean_native_corpus import load_npz,save_npz,write_json,footprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','audit','policy','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    if out.exists() or REPO/'outputs'/TASK.name not in out.parents:raise ValueError('fresh Task view required')
    meta=json.loads((root/'manifest.json').read_text());policy=json.loads(a.policy.read_text())
    validate_policy(policy)
    audit=json.loads((a.audit/'progress.json').read_text())
    if audit['status']!='COMPLETED' or audit['input_manifest']!=str(root/'manifest.json'):
        raise ValueError('completed full motion scan of identical parent required')
    path=root/meta['window_index'];stamp=path.stat()
    if (stamp.st_size,stamp.st_mtime_ns)!=(meta['window_index_size_bytes'],meta['window_index_mtime_ns']):
        raise ValueError('frozen parent index changed')
    arrays=load_npz(path);onsets={};flagged={};seen=set();train_positives=Counter();by_split=Counter()
    with gzip.open(a.audit/'records.jsonl.gz','rt') as handle:
        for line in handle:
            record=json.loads(line);row=record['row'];r=meta['sequences'][row]
            if row in seen or r['source_id']!=4 or record['split']!=r['split']:raise ValueError('audit identity mismatch')
            seen.add(row)
            reasons=events({k:np.asarray(v) for k,v in record['features'].items()},policy)
            frames=sorted({int(f) for values in reasons.values() for f in values})
            if frames:
                onsets[row]=frames[0]
                flagged[row]=dict(row=row,split=r['split'],first_suspect_frame=frames[0],
                    reasons={k:v.tolist() for k,v in reasons.items() if len(v)},maxima=record['maxima'],
                    source_sequence=r['source_sequence'],source_manifest=r['source_manifest'])
                by_split[r['split']]+=1
                if r['split']=='train':train_positives[str(r.get('official_success'))]+=1
    if seen!={i for i,r in enumerate(meta['sequences']) if r['source_id']==4}:
        raise ValueError('scan does not cover every simulator sequence')
    removed={};kept={};groups={}
    for key,index in arrays.items():
        mask=np.asarray([int(row) in onsets and int(tick)+24>=onsets[int(row)] for row,tick in index],bool)
        removed[key]=index[mask].copy();kept[key]=index[~mask].copy()
    if not len(removed['verified_train']):raise ValueError('no TRAIN exclusions; inspect policy before deriving view')
    for row,tick in removed['verified_train']:groups.setdefault(int(row),[]).append(int(tick))
    if shutil.disk_usage(out.parent).free<20*2**30:raise RuntimeError('20GiB free floor')
    out.mkdir(parents=True);started=time.monotonic()
    progress=dict(status='RUNNING',run_id=out.name,training_allowed=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        input_manifest=str(root/'manifest.json'),audit=str(a.audit.resolve()),max_seconds=1200,
        artifact_budget_bytes=200_000_000,workers=4)
    def checkpoint(done):
        progress.update(processed_train_sequences=done,total_train_sequences=len(groups),elapsed_s=time.monotonic()-started)
        write_json(out/'progress.json',progress)
        if progress['elapsed_s']>1200 or footprint(out)>200_000_000 or shutil.disk_usage(out).free<20*2**30 or (out/'STOP').exists():
            raise RuntimeError('motion-view deadline/disk/artifact/STOP')
    try:
        checkpoint(0)
        needed={k for row in groups for k in meta['sequences'][row]['objects']}
        geometries={g['key']:load_npz(root/g['path']) for g in meta['objects'] if g['key'] in needed}
        def contribution(item):
            row,ticks=item;r=meta['sequences'][row];p=root/r['path'];s=p.stat()
            if r['split']!='train' or (s.st_size,s.st_mtime_ns)!=(r['payload_size_bytes'],r['payload_mtime_ns']):
                raise ValueError('removed TRAIN contribution identity mismatch')
            return sequence_moments(canonical_sequence(load_npz(p),r),ticks,[geometries[k] for k in r['objects']],r['target_slot'])
        accum=dict(hand=Moments(),scene=Moments(),windows=0)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for done,m in enumerate(pool.map(contribution,groups.items()),1):
                for name in ('hand','scene'):accum[name].combine(m[name])
                accum['windows']+=m['windows']
                if done%128==0:checkpoint(done)
        stats=json.loads((root/'source-moments.json').read_text())
        if stats['input_manifest']!=str(root/'manifest.json'):raise ValueError('parent moments binding mismatch')
        before=stats['source_moments']['4']
        after={**before,'windows':before['windows']-accum['windows']}
        for name in ('hand','scene'):after[name]=subtract_moments(before[name],accum[name].result())
        after['sequences']=len({int(row) for row,tick in kept['verified_train'] if meta['sequences'][row]['source_id']==4})
        stats['source_moments']['4']=after
        stats.update(input_manifest=str(out/'manifest.json'),windows=len(kept['verified_train']),
            parent_moments=str(root/'source-moments.json'),git_commit=progress['git_commit'],
            derivation='Subtract exactly the removed TRAIN windows; DEV does not fit moments.')
        save_npz(out/'windows.npz',kept);save_npz(out/'excluded-parent-windows.npz',removed)
        write_json(out/'source-moments.json',stats)
        stamp=(out/'windows.npz').stat()
        counts=Counter(int(row) for key in ('all_train','all_dev') for row,tick in kept[key])
        rows=[{**r,'path':str((root/r['path']).resolve()),'source_manifest_snapshot':str((root/r['source_manifest_snapshot']).resolve()),
            'parent_clean_windows':r.get('clean_windows'),'clean_windows':counts[i],
            **({'motion_quality':flagged[i]} if i in flagged else {})} for i,r in enumerate(meta['sequences'])]
        new={**meta,'run_id':out.name,'git_commit':progress['git_commit'],'sequences':rows,
            'objects':[{**g,'path':str((root/g['path']).resolve())} for g in meta['objects']],
            'window_index':'windows.npz','window_index_size_bytes':stamp.st_size,'window_index_mtime_ns':stamp.st_mtime_ns,
            'input_snapshots':[str(root/'manifest.json')],
            'motion_admission':dict(schema='simulator-impulse-prefix.v1',policy=policy,audit=str(a.audit.resolve()),
                scope='source4 only; preserve prefix; H4+K24 must end before first suspect transition; never filter official_success'),
            'limitations':meta['limitations']+['Versioned conservative simulator motion thresholds, not proof of physical validity.']}
        write_json(out/'quarantine.json',dict(policy=policy,sequences=list(flagged.values())))
        summary=dict(status='COMPLETED',windows={k:len(v) for k,v in kept.items()},
            excluded={k:len(v) for k,v in removed.items()},flagged_sequences=dict(by_split),
            fully_quarantined_sequences=sum(counts[i]==0 for i in onsets),preserved_prefix_sequences=sum(counts[i]>0 for i in onsets),
            flagged_train_official_success_counts=dict(train_positives),parent=str(root),
            retained_source4_train_windows=after['windows'],retained_source4_dev_windows=sum(meta['sequences'][row]['source_id']==4 for row,tick in kept['verified_dev']))
        write_json(out/'cleaning-summary.json',summary);write_json(out/'manifest.pending.json',new)
        checkpoint(len(groups));progress['status']='COMPLETED';write_json(out/'progress.json',progress)
        (out/'manifest.pending.json').replace(out/'manifest.json')
        print(json.dumps(summary),flush=True)
    except BaseException as error:
        progress.update(status='FAILED',reason=repr(error));write_json(out/'progress.json',progress);raise


if __name__=='__main__':main()
