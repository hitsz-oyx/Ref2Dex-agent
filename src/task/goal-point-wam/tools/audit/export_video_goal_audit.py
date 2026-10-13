#!/usr/bin/env python3
"""Read-only compact video goal audit export; writes a ZIP to stdout only."""
import argparse
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--max-records',type=int,default=24)
    a=p.parse_args()
    if not 1<=a.max_records<=24:raise ValueError('bounded24-record export required')
    started=time.monotonic();cv2.setNumThreads(2)
    meta=json.loads((a.corpus/'manifest.json').read_text())
    if meta['status']!='COMPLETED' or not meta['training_allowed']:
        raise ValueError('admitted source corpus required')
    by_record={}
    for r in meta['records']:by_record.setdefault(int(r['recording_id']),[]).append(r)
    chosen=[]
    for key in sorted(by_record)[:a.max_records]:
        rows=sorted(by_record[key],key=lambda r:r['source_current_frame'])
        chosen.append(rows[len(rows)//2])
    summary=dict(schema='ref2dex.video-object-goal-audit-export.v1',source=str(a.corpus),
        selection='one clock-median window per raw recording in recording-id order, before inspecting motion',
        source_training_allowed=True,training_allowed=False,engineering_only=True,records=[],excluded=[])
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for r in chosen:
            label='r%02d'%r['recording_id']
            try:
                path=a.corpus/r['path'];payload=path.read_bytes()
                with np.load(io.BytesIO(payload),allow_pickle=False) as f:
                    xyz=f['query_xyz'].copy();valid=f['query_valid'].copy()
                    if f['object_goal_valid'].any():raise ValueError('expected original null-goal input')
                source=r['source_identity']
                raw=Path(source['acquisition_manifest']).parent/'raw'/(source['episode_id']+'.hdf5')
                video=raw.with_suffix('.mp4')
                tick=int(r['source_current_frame'])
                frames=tick+np.array([0,8,16,24])
                with h5py.File(raw,'r') as f:
                    if str(f.attrs['session_name']).replace('.mov','')!=r['session'].replace('.mov',''):
                        raise ValueError('raw source session differs')
                    intrinsic=f['camera/intrinsic'][:].astype(np.float32)
                    camera=f['transforms/camera'][frames].astype(np.float32)
                cap=cv2.VideoCapture(str(video))
                if not cap.isOpened():raise ValueError('raw RGB unavailable')
                wh=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                intrinsic[0]*=640/wh[0];intrinsic[1]*=360/wh[1]
                images=[]
                for frame in frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame));ok,bgr=cap.read()
                    if not ok:raise ValueError('requested RGB frame unavailable')
                    images.append(cv2.imencode('.png',cv2.resize(bgr,(640,360)))[1].tobytes())
                cap.release()
                projected=xyz@intrinsic.T
                uv=projected[:,:2]/projected[:,2:]
                image_valid=valid & np.isfinite(uv).all(-1) & (xyz[:,2]>0) & (uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<360)
                buffer=io.BytesIO();np.savez_compressed(buffer,current_uv=uv,projection_valid=image_valid,
                    intrinsic=intrinsic,camera_to_origin=camera,source_frame_ids=frames)
                z.writestr(label+'/clip.npz',payload);z.writestr(label+'/projection.npz',buffer.getvalue())
                for offset,png in zip([0,8,16,24],images):z.writestr(label+'/frame%02d.png'%offset,png)
                summary['records'].append(dict(id=label,record=r,raw_hdf5=str(raw),video=str(video),
                    current_projected_points=int(image_valid.sum()),query_points=len(xyz),original_wh=wh))
            except (OSError,ValueError,KeyError) as e:
                summary['excluded'].append(dict(id=label,path=r['path'],reason=repr(e)))
            if stream.tell()>100_000_000 or time.monotonic()-started>900:raise RuntimeError('export size/time budget')
        summary.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
        z.writestr('manifest.json',json.dumps(summary,indent=2))
    sys.stdout.buffer.write(stream.getvalue())


if __name__=='__main__':main()
