"""CPU range preflight of prescribed C4 draws; C5 exact known-fault exposure check."""
import sys,json,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/ptv3-shared-wam-v6';sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from diagnostic_components import DiagnosticWindows
from remote_staged_data import read,write
OUT=ROOT/'tmp/rh/diagnosis'

def group_check(arg):
 row,slot,items,geometry,origin,voxel=arg;folder=Path(row['path']);ticks=np.array([tick for _,tick in items],np.int64)
 j=np.load(folder/'joints.npy',mmap_mode='r');jv=np.load(folder/'joint_valid.npy',mmap_mode='r');sv=np.load(folder/'surface_valid.npy',mmap_mode='r');offsets=np.load(folder/'surface_offsets.npy',mmap_mode='r');poses=np.load(folder/'object_poses.npy',mmap_mode='r')
 anchor=np.linalg.inv(poses[ticks,slot]).astype(np.float32);current=j[ticks]@np.swapaxes(anchor[:,None,:3,:3],-1,-2)+anchor[:,None,None,:3,3]
 dense=offsets[ticks].astype(np.float32)+j[ticks,:,None,0,:];dense=dense@np.swapaxes(anchor[:,None,:3,:3],-1,-2)+anchor[:,None,None,:3,3]
 hv=jv[ticks].copy();dv=sv[ticks].copy()
 for lag in (1,2,3):hv &=jv[ticks-lag];dv &=sv[ticks-lag]
 hv=hv.reshape(len(ticks),-1);dv=np.broadcast_to(dv[:,:,None],(len(ticks),2,778)).reshape(len(ticks),-1)
 hand=np.concatenate([current.reshape(len(ticks),-1,3),dense.reshape(len(ticks),-1,3)],1);valid=np.concatenate([hv,dv],1);grid=np.floor((hand-origin)/voxel);bad=(valid&((grid<0).any(-1)|(grid>65535).any(-1))).any(1)
 surface=np.load(geometry,allow_pickle=False)['surface'];g=np.floor((surface-origin)/voxel)
 if ((g<0)|(g>65535)).any():bad[:]=True
 return [int(items[i][0]) for i in np.flatnonzero(bad)]

def main():
 started=time.monotonic();out={}
 for arm in ('C4-source1','C4-source10','C4-source0'):
  r=read(OUT/arm/'recipe-v2.json');cfg=read(r['config']);ds=DiagnosticWindows(r['manifest'],'train',r['geometry_manifest'],r['manifest_sha256'],geometry_gt_object=False,probabilities=r['source_probabilities']);first={}
  for step in range(1,501):
   for index in ds.draw(96,170,step,0):first.setdefault(index,step)
  groups=defaultdict(list)
  for index in first:
   sid,slot,tick=map(int,ds.index[index]);groups[(sid,slot)].append((index,tick))
  tasks=[]
  for (sid,slot),items in groups.items():
   row=ds.rows[sid];name=row['objects'][slot];key='s%d:%s'%(row['source_id'],name) if row['source_id']<3 else name;geometry=Path(r['geometry_manifest']).parent/ds.geometry[key]['path'];tasks.append((row,slot,items,geometry,np.array(cfg['voxel_origin_m'],np.float32),cfg['voxel_m']))
  bad=[]
  with ThreadPoolExecutor(8) as pool:
   for values in pool.map(group_check,tasks):bad.extend(values)
  errors=[]
  if bad:
   import torch
   from oakink_wm.pointworld import fixed_workspace_grid
   for index in sorted(bad,key=lambda i:first[i])[:5]:
    a=ds[index];xyz=np.concatenate([a['xyz'],a['start_hand'].reshape(-1,3),a['dense_hand'].reshape(-1,3)]);valid=np.concatenate([a['point_valid'],a['start_hand_valid'].ravel(),a['dense_hand_valid'].ravel()])
    try:fixed_workspace_grid(torch.from_numpy(xyz[None]),torch.from_numpy(valid[None]),cfg['voxel_origin_m'],cfg['voxel_m']);raise AssertionError('vector range flag not reproduced')
    except ValueError as exc:errors.append(dict(index=index,first_step=first[index],tuple=ds.index[index].tolist(),error=str(exc)))
  result=dict(status='PASS_500_PLANNED_DRAWS' if not bad else 'BLOCKED_FIXED_WORKSPACE',updates=0,planned_updates=500,batch=96,unique_windows=len(first),sequence_slot_groups=len(groups),bad_unique_windows=len(bad),bad_windows_production_confirmed=errors,elapsed_s=time.monotonic()-started)
  write(OUT/arm/'workspace-preflight.json',result,immutable=True);out[arm]=result;print(arm,json.dumps(result),flush=True)
 # First try the exact failed uniform step/rank with the actual C5 gate.
 r=read(OUT/'C5/recipe-v2.json');ds=DiagnosticWindows(r['manifest'],'train',r['geometry_manifest'],r['manifest_sha256'],geometry_gt_object=False,probabilities=r['source_probabilities'],validity_gate=r['validity_gate']);fault=read(OUT/'workspace-fault-reproduction.json')['bad_windows'][0];known=int(fault['index']);exposure=None
 for step,rank in [(104,2)]+[(s,k) for s in range(1,501) for k in range(4) if (s,k)!=(104,2)]:
  if known in ds.draw(96,170,step,rank):exposure=dict(step=step,rank=rank);break
 result=dict(status='BLOCKED_FIXED_WORKSPACE' if exposure else 'KNOWN_WINDOW_NOT_DRAWN',known_bad_window=fault,exposure=exposure,validity_gate_retains_window=bool(ds.valid_windows[known]),updates=0,scope='known production-confirmed geometry fault, not exhaustive C5 input range audit')
 write(OUT/'C5/workspace-preflight.json',result,immutable=True);print('C5',json.dumps(result),flush=True)
 write(OUT/'remaining-workspace-preflight.json',dict(arms=out,C5=result,elapsed_s=time.monotonic()-started),immutable=True)
if __name__=='__main__':main()
