"""Accumulate verified-clock TRAIN moments without choosing a training recipe."""
import argparse
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
from pose_mesh_data.cleaning import canonical_sequence, CLEANING_SCHEMA
from pose_mesh_data.statistics import Moments,sequence_moments
from pose_mesh_data.windows import validate_sequence
from clean_native_corpus import load_npz,save_npz,write_json,footprint


def accumulate(root,output,encoder_config,workers=4,max_seconds=1800):
    root,output=root.resolve(),output.resolve()
    if (REPO/'outputs'/TASK.name not in output.parents or output.exists()
            or not 1<=workers<=8 or not 60<=max_seconds<=3600):
        raise ValueError('fresh bounded Task statistics run required')
    meta=json.loads((root/'manifest.json').read_text())
    if (meta.get('cleaning_schema')!=CLEANING_SCHEMA or meta.get('status')!='COMPLETED'
            or meta.get('training_allowed') is not False
            or json.loads((root/'progress.json').read_text()).get('status')!='COMPLETED'):
        raise ValueError('completed engineering cleaning snapshot required')
    config=json.loads(encoder_config.read_text())
    workspace=dict(origin_m=config['voxel_origin_m'],voxel_m=config['voxel_m'],max_grid=65534)
    if (np.asarray(workspace['origin_m']).shape!=(3,) or not np.isfinite(workspace['origin_m']).all()
            or not np.isfinite(workspace['voxel_m']) or workspace['voxel_m']<=0):
        raise ValueError('finite fixed encoder workspace required')
    if shutil.disk_usage(output.parent).free<20*2**30:
        raise RuntimeError('less than 20GiB free')
    path=root/meta['window_index']
    expected=(meta['window_index_size_bytes'],meta['window_index_mtime_ns'])
    stamp=path.stat()
    if (stamp.st_size,stamp.st_mtime_ns)!=expected:
        raise ValueError('clean window index changed')
    with np.load(path,allow_pickle=False) as z:
        index=z['verified_train'].copy()
    groups={}
    for row,tick in index:
        if (not 0<=row<len(meta['sequences']) or meta['sequences'][row]['split']!='train'
                or meta['sequences'][row].get('clock_verified') is not True):
            raise ValueError('DEV/unverified row in TRAIN moments index')
        groups.setdefault(int(row),[]).append(int(tick))
    required_geometry={key for row in groups for key in meta['sequences'][row]['objects']}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'input-manifest.json',meta)
    write_json(output/'encoder-config.json',config)
    progress=dict(status='RUNNING',run_id=output.name,training_allowed=False,split='train',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        input=str(root),workers=workers,max_seconds=max_seconds,artifact_budget_bytes=100_000_000,
        device='CPU: exact array sufficient statistics; no neural computation')
    started=time.monotonic()
    def checkpoint(done):
        progress.update(processed_sequences=done,total_sequences=len(groups),elapsed_s=time.monotonic()-started,
            free_bytes=shutil.disk_usage(output).free,checked_at=time.time())
        write_json(output/'progress.json',progress)
        if (progress['elapsed_s']>max_seconds or progress['free_bytes']<20*2**30
                or footprint(output)>progress['artifact_budget_bytes'] or (output/'STOP').exists()):
            raise RuntimeError('statistics deadline/disk/artifact/STOP boundary')
    try:
        checkpoint(0)
        geometry={g['key']:load_npz(root/g['path']) for g in meta['objects'] if g['key'] in required_geometry}
        if set(geometry)!=required_geometry:
            raise ValueError('missing TRAIN geometry identities')
        def process(item):
            row_id,ticks=item
            row=meta['sequences'][row_id]
            if row['split']!='train' or row.get('clock_verified') is not True:
                raise ValueError('DEV/unverified row in TRAIN moments index')
            payload=root/row['path']
            for after in (False,True):
                s=payload.stat()
                if (s.st_size,s.st_mtime_ns)!=(row['payload_size_bytes'],row['payload_mtime_ns']):
                    raise ValueError('clean payload changed during moments')
                if not after:
                    seq=canonical_sequence(load_npz(payload),row)
                    validate_sequence(seq)
                    result=sequence_moments(seq,ticks,[geometry[k] for k in row['objects']],row['target_slot'],workspace)
            return row_id,str(row['source_id']),result
        counts,accumulators,failed={},{},[]
        entries=list(groups.items())
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0,len(entries),256):
                for row_id,sid,result in pool.map(process,entries[offset:offset+256]):
                    counts.setdefault(sid,dict(windows=0,sequences=0,workspace_incompatible_windows=0,
                        observed_min_m=[float('inf')]*3,observed_max_m=[float('-inf')]*3))
                    c=counts[sid]
                    c['windows']+=result['windows'];c['sequences']+=1
                    c['workspace_incompatible_windows']+=len(result['workspace_incompatible_ticks'])
                    c['observed_min_m']=np.minimum(c['observed_min_m'],result['observed_min_m']).tolist()
                    c['observed_max_m']=np.maximum(c['observed_max_m'],result['observed_max_m']).tolist()
                    failed.extend((row_id,tick) for tick in result['workspace_incompatible_ticks'])
                    a=accumulators.setdefault(sid,{'hand':Moments(),'scene':Moments()})
                    for name in a:
                        a[name].combine(result[name])
                checkpoint(min(offset+256,len(entries)))
                if offset%2048==0:
                    print(json.dumps(progress),flush=True)
        sources={sid:{**c,**{name:m.result() for name,m in accumulators[sid].items()},
            'source_profile':meta['source_profiles'][sid]} for sid,c in counts.items()}
        if sum(c['windows'] for c in counts.values())!=len(index):
            raise ValueError('TRAIN moment window count mismatch')
        for source in sources.values():
            for name in ('hand','scene'):
                if not all(c>0 for c in source[name]['count']):
                    raise ValueError('missing source/horizon moments')
        stamp=path.stat()
        if (stamp.st_size,stamp.st_mtime_ns)!=expected:
            raise ValueError('clean index changed during moments')
        save_npz(output/'workspace-incompatible-windows.npz',{'row_tick':np.asarray(failed,np.int64).reshape(-1,2)})
        artifact=dict(schema='ref2dex.native-train-moments.v1',status='COMPLETED',training_allowed=False,
            units='m',split='train',index_subset='verified_train',horizon=24,windows=len(index),
            geometry_files_read=len(geometry),
            input_manifest=str(root/'manifest.json'),git_commit=progress['git_commit'],
            aggregation='Per-source masked point count/mean/M2, all accepted TRAIN windows; no DEV reads.',
            source_moments=sources,workspace=workspace,workspace_incompatible_windows=len(failed),
            workspace_note='Current observed scene surfaces and present hand points only; compatibility diagnostic, no windows removed.',
            limitations=['No final source weights, std floor or training admission selected.',
                'Use mix_moments to compose common moments under source -> uniform-window sampling.',
                'Existing production v2 normalizer accepts main3 only; this is a separate preparation artifact.'])
        write_json(output/'source-moments.pending.json',artifact)
        checkpoint(len(entries))
        progress['status']='COMPLETED'
        write_json(output/'progress.json',progress)
        (output/'source-moments.pending.json').replace(output/'source-moments.json')
        return {k:v for k,v in artifact.items() if k!='source_moments'}
    except BaseException as error:
        progress.update(status='FAILED',reason=str(error))
        write_json(output/'progress.json',progress)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encoder-config',type=Path,default=TASK.parent/'ptv3-shared-wam-v2/configs/pretrain_remote_80k.json')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seconds',type=int,default=1800)
    a=p.parse_args()
    print(json.dumps(accumulate(a.root,a.output,a.encoder_config,a.workers,a.seconds)),flush=True)
