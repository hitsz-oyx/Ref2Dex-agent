"""Derive a fixed-encoder window view and consistent TRAIN moments, retaining parent."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
from pose_mesh_data.cleaning import canonical_sequence,CLEANING_SCHEMA
from pose_mesh_data.statistics import Moments,observed_workspace,sequence_moments,subtract_moments
from clean_native_corpus import load_npz,save_npz,write_json,footprint


def prepare(root,moments_root,output,workers=4):
    root,moments_root,output=root.resolve(),moments_root.resolve(),output.resolve()
    if REPO/'outputs'/TASK.name not in output.parents or output.exists() or not 1<=workers<=8:
        raise ValueError('fresh bounded encoder view required')
    meta=json.loads((root/'manifest.json').read_text())
    stats=json.loads((moments_root/'source-moments.json').read_text())
    if (meta.get('cleaning_schema')!=CLEANING_SCHEMA or meta.get('status')!='COMPLETED'
            or meta.get('training_allowed') is not False or stats.get('status')!='COMPLETED'
            or stats.get('training_allowed') is not False or stats.get('split')!='train'
            or stats.get('input_manifest')!=str(root/'manifest.json')):
        raise ValueError('completed matching engineering snapshot and TRAIN moments required')
    for parent in (root,moments_root):
        if json.loads((parent/'progress.json').read_text()).get('status')!='COMPLETED':
            raise ValueError('incomplete parent run')
    index_path=root/meta['window_index']
    stamp=index_path.stat()
    if (stamp.st_size,stamp.st_mtime_ns)!=(meta['window_index_size_bytes'],meta['window_index_mtime_ns']):
        raise ValueError('parent index changed')
    arrays=load_npz(index_path)
    if not np.array_equal(arrays['all_train'],arrays['verified_train']) or len(arrays['verified_train'])!=stats['windows']:
        raise ValueError('this view requires the identical all/verified TRAIN scope used by its moments')
    bad_train=load_npz(moments_root/'workspace-incompatible-windows.npz')['row_tick']
    train_set={tuple(map(int,x)) for x in arrays['verified_train']}
    bad_set={tuple(map(int,x)) for x in bad_train}
    if len(bad_set)!=len(bad_train) or not bad_set<=train_set or len(bad_set)!=stats['workspace_incompatible_windows']:
        raise ValueError('TRAIN exclusion identity mismatch')
    dev_groups={}
    for row,tick in arrays['all_dev']:
        if meta['sequences'][row]['split']!='dev':
            raise ValueError('DEV index split mismatch')
        dev_groups.setdefault(int(row),[]).append(int(tick))
    train_groups={}
    for row,tick in bad_train:
        train_groups.setdefault(int(row),[]).append(int(tick))
    needed={k for i in set(dev_groups)|set(train_groups) for k in meta['sequences'][i]['objects']}
    if shutil.disk_usage(output.parent).free<20*2**30:
        raise RuntimeError('less than 20GiB free')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    progress=dict(status='RUNNING',training_allowed=False,run_id=output.name,workers=workers,max_seconds=900,
        artifact_budget_bytes=200_000_000,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    def checkpoint(done):
        progress.update(processed_dev_sequences=done,total_dev_sequences=len(dev_groups),elapsed_s=time.monotonic()-started)
        write_json(output/'progress.json',progress)
        if (progress['elapsed_s']>900 or shutil.disk_usage(output).free<20*2**30 or footprint(output)>200_000_000
                or (output/'STOP').exists()):
            raise RuntimeError('encoder view deadline/disk/artifact/STOP boundary')
    try:
        checkpoint(0)
        geometry={g['key']:load_npz(root/g['path']) for g in meta['objects'] if g['key'] in needed}
        def read(row_id):
            r=meta['sequences'][row_id];p=root/r['path'];s=p.stat()
            if (s.st_size,s.st_mtime_ns)!=(r['payload_size_bytes'],r['payload_mtime_ns']):
                raise ValueError('parent payload changed')
            sequence=canonical_sequence(load_npz(p),r)
            after=p.stat()
            if (after.st_size,after.st_mtime_ns)!=(s.st_size,s.st_mtime_ns):
                raise ValueError('parent payload changed during read')
            return sequence,[geometry[k] for k in r['objects']],r
        def check_dev(item):
            row_id,ticks=item;s,g,r=read(row_id)
            keep=observed_workspace(s,ticks,g,r['target_slot'],stats['workspace'])
            return [(row_id,t) for t,ok in zip(ticks,keep) if not ok]
        bad_dev=[]
        items=list(dev_groups.items())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0,len(items),256):
                for bad in pool.map(check_dev,items[offset:offset+256]):
                    bad_dev.extend(bad)
                checkpoint(min(offset+256,len(items)))
        removed={}
        for row_id,ticks in train_groups.items():
            s,g,r=read(row_id)
            if r['split']!='train' or observed_workspace(s,ticks,g,r['target_slot'],stats['workspace']).any():
                raise ValueError('TRAIN exclusions disagree with current-input predicate')
            m=sequence_moments(s,ticks,g,r['target_slot'])
            sid=str(r['source_id'])
            accum=removed.setdefault(sid,dict(hand=Moments(),scene=Moments(),windows=0))
            for name in ('hand','scene'):accum[name].combine(m[name])
            accum['windows']+=len(ticks)
        excluded={'train':bad_set,'dev':set(bad_dev)}
        for key,index in arrays.items():
            split=key.rsplit('_',1)[1]
            arrays[key]=np.asarray([x for x in index if tuple(map(int,x)) not in excluded[split]],np.int64).reshape(-1,2)
        remaining=Counter(str(meta['sequences'][row]['source_id']) for row,tick in arrays['verified_train'])
        source_sequences={sid:set() for sid in remaining}
        for row,tick in arrays['verified_train']:source_sequences[str(meta['sequences'][row]['source_id'])].add(int(row))
        for sid,s in stats['source_moments'].items():
            if sid in removed:
                for name in ('hand','scene'):s[name]=subtract_moments(s[name],removed[sid][name].result())
            s['parent_workspace_incompatible_windows']=s.pop('workspace_incompatible_windows')
            for axis in ('min','max'):s['parent_observed_'+axis+'_m']=s.pop('observed_'+axis+'_m')
            s.update(windows=remaining[sid],sequences=len(source_sequences[sid]),workspace_incompatible_windows=0)
        current_stamp=index_path.stat()
        if (current_stamp.st_size,current_stamp.st_mtime_ns)!=(meta['window_index_size_bytes'],meta['window_index_mtime_ns']):
            raise ValueError('parent index changed during view preparation')
        save_npz(output/'windows.npz',arrays)
        save_npz(output/'excluded-parent-windows.npz',dict(train=bad_train,dev=np.asarray(bad_dev,np.int64).reshape(-1,2)))
        stats.update(input_manifest=str(output/'manifest.json'),windows=len(arrays['verified_train']),workspace_incompatible_windows=0,
            parent_moments=str(moments_root/'source-moments.json'),derivation='Parent TRAIN sufficient statistics minus excluded TRAIN window contributions; DEV only participates in its own input-domain index.',
            git_commit=progress['git_commit'])
        stats['parent_geometry_files_read']=stats.pop('geometry_files_read')
        write_json(output/'source-moments.json',stats)
        write_json(output/'parent-manifest.json',meta)
        stamp=(output/'windows.npz').stat()
        effective=Counter(int(row) for key in ('all_train','all_dev') for row,tick in arrays[key])
        rows=[{**r,'path':str((root/r['path']).resolve()),'source_manifest_snapshot':str((root/r['source_manifest_snapshot']).resolve()),
            'parent_clean_windows':r.get('clean_windows'),'clean_windows':effective[i]} for i,r in enumerate(meta['sequences'])]
        new={**meta,'run_id':output.name,'git_commit':progress['git_commit'],'sequences':rows,
            'objects':[{**g,'path':str((root/g['path']).resolve())} for g in meta['objects']],
            'window_index':'windows.npz','window_index_size_bytes':stamp.st_size,'window_index_mtime_ns':stamp.st_mtime_ns,
            'input_snapshots':[str(root/'manifest.json')],
            'encoder_admission':dict(schema='observed-fixed-grid.v1',workspace=stats['workspace'],scope='Current observed scene surfaces and H4-present current hand points; no future, action, reward or success filter'),
            'limitations':meta['limitations']+['Explicit encoder compatibility view; complete parent corpus remains available.']}
        summary=dict(status='COMPLETED',training_allowed=False,windows={k:len(v) for k,v in arrays.items()},
            excluded_train=len(bad_train),excluded_all_dev=len(bad_dev),parent=str(root/'manifest.json'),
            retained_sequences=len(rows),window_bearing_sequences=len(effective),geometry_files=len(new['objects']))
        write_json(output/'cleaning-summary.json',summary)
        write_json(output/'manifest.pending.json',new)
        checkpoint(len(items))
        progress['status']='COMPLETED';write_json(output/'progress.json',progress)
        (output/'manifest.pending.json').replace(output/'manifest.json')
        return summary
    except BaseException as error:
        progress.update(status='FAILED',reason=str(error));write_json(output/'progress.json',progress)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--moments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    print(json.dumps(prepare(a.root,a.moments,a.output,a.workers)),flush=True)
