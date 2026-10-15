"""Join completed cleaning snapshots by reference, preserving split/clock gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
REPO=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.cleaning import CLEANING_SCHEMA
from clean_native_corpus import write_json, save_npz, footprint


def merge(inputs, output):
    output=output.resolve()
    if REPO/'outputs'/TASK.name not in output.parents or output.exists() or not 2<=len(inputs)<=8:
        raise ValueError('fresh bounded Task merge required')
    if shutil.disk_usage(output.parent).free<20*2**30:
        raise RuntimeError('less than 20GiB free')
    rows,geometry,profiles,seen=[],{}, {},set()
    arrays={k:[] for k in ('all_train','all_dev','verified_train','verified_dev')}
    policy=None
    parents=[]
    for root in inputs:
        root=root.resolve()
        meta=json.loads((root/'manifest.json').read_text())
        progress=json.loads((root/'progress.json').read_text())
        if (meta.get('cleaning_schema')!=CLEANING_SCHEMA or meta.get('status')!='COMPLETED'
                or progress.get('status')!='COMPLETED' or meta.get('training_allowed') is not False
                or meta.get('units')!='m' or meta.get('fps')!=30):
            raise ValueError('only completed metre30Hz engineering snapshots can merge')
        if policy is not None and policy!=meta['window_policy']:
            raise ValueError('window quality policies differ')
        policy=meta['window_policy']
        for sid,profile in meta['source_profiles'].items():
            if sid in profiles and profiles[sid]!=profile:
                raise ValueError('source profile changed between snapshots')
            profiles[sid]=profile
        for g in meta['objects']:
            g={**g,'path':str((root/g['path']).resolve())}
            if g['key'] in geometry and g!=geometry[g['key']]:
                raise ValueError('same geometry key has different cleaned payloads')
            geometry[g['key']]=g
        offset=len(rows)
        for row in meta['sequences']:
            identity=(row['source_manifest'],row['source_row'])
            if identity in seen:
                raise ValueError('duplicate source lineage across snapshots')
            seen.add(identity)
            rows.append({**row,'path':str((root/row['path']).resolve()),
                'source_manifest_snapshot':str((root/row['source_manifest_snapshot']).resolve())})
        path=root/meta['window_index']
        stamp=path.stat()
        if (stamp.st_size,stamp.st_mtime_ns)!=(meta['window_index_size_bytes'],meta['window_index_mtime_ns']):
            raise ValueError('parent window index changed')
        with np.load(path,allow_pickle=False) as z:
            for key in arrays:
                index=z[key].copy()
                index[:,0]+=offset
                arrays[key].append(index)
        parents.append(str(root/'manifest.json'))
    groups={s:{r['group_id'] for r in rows if r['split']==s} for s in ('train','dev')}
    if groups['train']&groups['dev']:
        raise ValueError('merged TRAIN/DEV group overlap')
    output.mkdir(parents=True,exist_ok=False)
    progress=dict(status='RUNNING',training_allowed=False,run_id=output.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    write_json(output/'progress.json',progress)
    try:
        arrays={k:np.concatenate(v,axis=0) for k,v in arrays.items()}
        save_npz(output/'windows.npz',arrays)
        stamp=(output/'windows.npz').stat()
        meta=dict(schema='ref2dex.pose-mesh.v1',cleaning_schema=CLEANING_SCHEMA,status='COMPLETED',
            units='m',fps=30,hand_order=['right','left'],training_allowed=False,
            source_profiles=profiles,window_policy=policy,window_index='windows.npz',
            window_index_size_bytes=stamp.st_size,window_index_mtime_ns=stamp.st_mtime_ns,
            objects=list(geometry.values()),sequences=rows,input_snapshots=parents,**{k:v for k,v in progress.items() if k not in ('status','training_allowed')},
            limitations=['Zero-copy merge of cleaned engineering snapshots; no training admission.',
                         'Clock-pending source windows remain outside the default verified index.',
                         'HOI4D synthetic frames remain marked in sequence payloads and source profile.'])
        summary=dict(status='COMPLETED',training_allowed=False,input_snapshots=parents,
            retained_sequences=len(rows),geometry_files=len(geometry),windows={k:len(v) for k,v in arrays.items()},
            source_sequences=dict(Counter(str(r['source_id'])+':'+r['split'] for r in rows)))
        write_json(output/'cleaning-summary.json',summary)
        write_json(output/'manifest.pending.json',meta)
        if footprint(output)>200_000_000 or shutil.disk_usage(output).free<20*2**30:
            raise RuntimeError('merged snapshot resource boundary')
        progress['status']='COMPLETED'
        write_json(output/'progress.json',progress)
        (output/'manifest.pending.json').replace(output/'manifest.json')
        return summary
    except BaseException as error:
        progress.update(status='FAILED',reason=str(error))
        write_json(output/'progress.json',progress)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(merge(a.input,a.output)),flush=True)
