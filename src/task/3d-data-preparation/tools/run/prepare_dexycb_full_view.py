"""Apply existing fixed-grid and strict motion rules to full DexYCB; fit TRAIN only."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import subprocess
import sys
import shutil
import time
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.statistics import Moments,sequence_moments,observed_workspace
from pose_mesh_data.motion_diagnostics import transition_diagnostics
from clean_native_corpus import load_npz,save_npz,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encoder-config',type=Path,required=True)
    a=p.parse_args();source=a.clean.resolve();out=a.output.resolve();assert not out.exists()
    meta=json.loads((source/'manifest.json').read_text());assert meta['status']=='COMPLETED'
    assert all(r['source_id']==8 and r['split'] in ('train','dev') for r in meta['sequences'])
    out.mkdir(parents=True);started=time.monotonic()
    config=json.loads(a.encoder_config.read_text());workspace=dict(origin_m=config['voxel_origin_m'],voxel_m=config['voxel_m'],max_grid=65534)
    geometry={g['key']:load_npz(source/g['path']) for g in meta['objects']}
    arrays=load_npz(source/meta['window_index']);groups=defaultdict(list)
    for split in ('train','dev'):
        for row,tick in arrays['verified_'+split]:groups[int(row)].append(int(tick))
    accepted={};events=[];hand=Moments();scene=Moments();windows=0;train_sequences=0
    for row_id,row in enumerate(meta['sequences']):
        if (out/'STOP').exists() or time.monotonic()-started>1200 or shutil.disk_usage(out).free<20*2**30:raise RuntimeError('STOP/deadline/disk boundary')
        data=load_npz(source/row['path']);ticks=np.asarray(groups[row_id],np.int64);bad=np.zeros(len(data['time']),bool);reasons={}
        queries=np.asarray([geometry[k]['query'] for k in row['objects']])
        for name,values,valid,lo,hi in transition_diagnostics(data,queries):
            cap={'object_speed_mps':5.,'hand_speed_mps':5.,'angular_speed_radps':40.}.get(name)
            if cap is None:continue
            hit=(valid&(values>cap)).reshape(len(lo),-1).any(-1)
            if hit.any():
                reasons[name]=hi[hit].tolist()
                for x,y in zip(lo[hit],hi[hit]):bad[x:y+1]=True
        cumulative=np.r_[0,np.cumsum(bad)];motion=(cumulative[ticks+25]-cumulative[ticks-3])==0
        geos=[geometry[k] for k in row['objects']]
        grid=observed_workspace(data,ticks,geos,row['target_slot'],workspace)
        permitted=ticks[motion&grid];accepted[row_id]=set(permitted.tolist())
        if reasons or (~grid).any():events.append(dict(row=row_id,split=row['split'],sequence=row['source_sequence'],motion_flags=reasons,motion_removed=int((~motion).sum()),grid_removed=int((~grid).sum()),excluded_ticks=ticks[~(motion&grid)].tolist()))
        if row['split']=='train' and len(permitted):
            moments=sequence_moments(data,permitted,geos,row['target_slot']);hand.combine(moments['hand']);scene.combine(moments['scene']);windows+=len(permitted);train_sequences+=1
        if row_id%100==0:write_json(out/'progress.json',dict(status='RUNNING',processed=row_id,total=len(meta['sequences']),elapsed_s=time.monotonic()-started))
    kept={k:rows[np.asarray([int(t) in accepted[int(r)] for r,t in rows],bool)].copy() for k,rows in arrays.items()}
    save_npz(out/'windows.npz',kept);save_npz(out/'excluded-parent-windows.npz',{k:rows[np.asarray([int(t) not in accepted[int(r)] for r,t in rows],bool)] for k,rows in arrays.items()})
    counts=Counter(int(row) for key in ('all_train','all_dev') for row,t in kept[key]);stamp=(out/'windows.npz').stat()
    rows=[{**r,'path':str((source/r['path']).resolve()),'source_manifest_snapshot':str((source/r['source_manifest_snapshot']).resolve()),'clean_windows':counts[i]} for i,r in enumerate(meta['sequences'])]
    result={**meta,'run_id':out.name,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'sequences':rows,'objects':[{**g,'path':str((source/g['path']).resolve())} for g in meta['objects']],'window_index':'windows.npz','window_index_size_bytes':stamp.st_size,'window_index_mtime_ns':stamp.st_mtime_ns,'input_snapshots':[str(source/'manifest.json')],'encoder_admission':dict(schema='observed-fixed-grid.v1',workspace=workspace),'strict_motion_admission':dict(object_speed_mps=5.,hand_speed_mps=5.,angular_speed_radps=40.,scope='TRAIN and DEV, actual observed adjacent intervals; whole H4+K24 overlap excluded')}
    write_json(out/'source-moments.json',dict(schema='ref2dex.native-train-moments.v1',status='COMPLETED',training_allowed=False,split='train',units='m',horizon=24,windows=windows,input_manifest=str(out/'manifest.json'),source_moments={'8':dict(windows=windows,sequences=train_sequences,hand=hand.result(),scene=scene.result())},aggregation='Exact retained TRAIN windows only; no DEV/TEST fits.'))
    summary=dict(status='COMPLETED',windows={k:len(v) for k,v in kept.items()},excluded={k:len(arrays[k])-len(v) for k,v in kept.items()},source_sequences=dict(Counter(r['split'] for r in rows)),elapsed_s=time.monotonic()-started,events=len(events))
    write_json(out/'quality-events.json',events);write_json(out/'cleaning-summary.json',summary);write_json(out/'progress.json',dict(status='COMPLETED',elapsed_s=time.monotonic()-started));write_json(out/'manifest.json',result)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
