"""Sample real dense2048 CAD for v6; keep original frozen flow query32."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 old=Path('outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache').resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 import trimesh
 m=json.loads((old/'manifest.json').read_text());(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 surface=json.loads((old/'surface-manifest.json').read_text());records=[]
 for i,row in enumerate(surface['objects']):
  name=row['key'].split('/')[1].split('@')[0];meshfile=Path('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/mjcf/objects')/name/(name+'.obj')
  mesh=trimesh.load(meshfile,force='mesh',process=False)
  # Fixed area-weighted barycentric sampling, without modifying global RNG.
  rng=np.random.default_rng(185+i);area=mesh.area_faces;face=rng.choice(len(area),2048,p=area/area.sum());u=rng.random((2048,2));s=np.sqrt(u[:,0]);b=np.stack((1-s,s*(1-u[:,1]),s*u[:,1]),1)
  points=(mesh.triangles[face]*b[:,:,None]).sum(1).astype(np.float32);normals=mesh.face_normals[face].astype(np.float32)
  with np.load(old/row['path']) as z:query=z['query'].copy();old_surface=z['surface'].copy()
  # Both sets must lie in the original simulator mesh coordinates and scale.
  if np.max(np.abs(query))>2 or np.max(np.abs(points))>2:raise ValueError('CAD metre range')
  if np.max(np.abs(points.mean(0)-old_surface.mean(0)))>.04:raise ValueError('dense/legacy frame mismatch')
  dest=out/(name+'.npz');np.savez_compressed(dest,surface=points,normals=normals,query=query)
  records.append(dict(key=row['key'],path=dest.name,source_mesh=str(meshfile),surface_points=2048,query_preserved=True,seed=185+i))
 (out/'surface-manifest.json').write_text(json.dumps(dict(schema='ref2dex.wam-v6-native-dense2048.v1',status='COMPLETED',objects=records,parent_cache=str(old)),indent=2)+'\n')
 print(json.dumps(dict(objects=len(records),surface_points=2048,query_preserved=True,output=str(out))))
if __name__=='__main__':main()
