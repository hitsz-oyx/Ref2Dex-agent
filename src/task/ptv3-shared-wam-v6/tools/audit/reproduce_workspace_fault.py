"""Reproduce exact failed C3 batch through production reader and grid validator on CPU."""
import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/ptv3-shared-wam-v6';sys.path[:0]=[str(TASK/'src'),str(TASK/'tools/run')]
from diagnostic_components import DiagnosticWindows
from remote_staged_data import read,write
from oakink_wm.pointworld import fixed_workspace_grid
OUT=ROOT/'tmp/rh/diagnosis'
def main():
 r=read(OUT/'C3-r2/recipe-v2.json');ds=DiagnosticWindows(r['manifest'],'train',r['geometry_manifest'],r['manifest_sha256'],geometry_gt_object=False,probabilities=r['source_probabilities']);config=read(r['config']);bad=[];xyzs=[];valids=[]
 for idx in ds.draw(96,170,104,2):
  a=ds[idx];xyz=np.concatenate([a['xyz'],a['start_hand'].reshape(-1,3),a['dense_hand'].reshape(-1,3)]).astype(np.float32);valid=np.concatenate([a['point_valid'],a['start_hand_valid'].ravel(),a['dense_hand_valid'].ravel()]);grid=np.floor((xyz-np.array(config['voxel_origin_m'],np.float32))/config['voxel_m']).astype(np.int64);off=valid&((grid<0).any(-1)|(grid>65535).any(-1));xyzs.append(xyz);valids.append(valid)
  if off.any():
   sid,slot,tick=map(int,ds.index[idx]);row=ds.rows[sid];bad.append(dict(index=int(idx),source_id=row['source_id'],sequence_id=sid,key=row['key'],slot=slot,tick=tick,path=row['path'],source_sequence=row.get('source_sequence'),invalid_valid_points=int(off.sum()),point_roles=dict(object=int(off[:2048].sum()),joint=int(off[2048:2090].sum()),dense_hand=int(off[2090:].sum())),bad_xyz_min=xyz[off].min(0).tolist(),bad_xyz_max=xyz[off].max(0).tolist(),valid_xyz_min=xyz[valid].min(0).tolist(),valid_xyz_max=xyz[valid].max(0).tolist(),hand_supervision_valid=bool(a['trajectory_valid'].any())))
 try:fixed_workspace_grid(torch.from_numpy(np.stack(xyzs)),torch.from_numpy(np.stack(valids)),config['voxel_origin_m'],config['voxel_m']);raise AssertionError('expected production grid rejection')
 except ValueError as exc:error=str(exc)
 result=dict(status='EXACT_FAILURE_REPRODUCED_CPU',step=104,rank=2,batch=96,production_validator_error=error,bad_windows=bad,model_executed=False,updates=0)
 write(OUT/'workspace-fault-reproduction.json',result,immutable=True);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
