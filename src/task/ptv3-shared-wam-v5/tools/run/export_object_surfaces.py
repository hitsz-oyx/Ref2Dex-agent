"""Export deterministic2048-point CAD surfaces, preserving original32 flow identities."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parents[5]
PREP=ROOT/'outputs/3d-data-preparation'
MAIN=ROOT/'outputs/cm-pointflow-effect-pretrain'


def mesh_file(path):
    mesh=trimesh.load(path,process=False,force='mesh',skip_materials=True)
    if isinstance(mesh,trimesh.Scene):mesh=mesh.dump(concatenate=True)
    return mesh


def sample(mesh,old,normals,query,key):
    tri=np.asarray(mesh.triangles,np.float64)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=-1)
    good=np.isfinite(tri).all((1,2))&(area>1e-15)
    tri,cross,area=tri[good],cross[good],area[good]
    if not len(tri):raise ValueError('no usable CAD triangles')
    old=np.asarray(old,np.float32);normals=np.asarray(normals,np.float32)
    if not np.isfinite(old).all() or len(np.unique(old,axis=0))!=len(old):raise ValueError('original distinct points required')
    # Catch unit/frame errors before mixing canonical surfaces.
    lo,hi=mesh.bounds
    if (old<lo-1e-5).any() or (old>hi+1e-5).any():raise ValueError('CAD and canonical bounds disagree')
    count=2048-len(old)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    ids=rng.choice(len(tri),count,p=area/area.sum());uv=rng.random((count,2))
    uv[uv.sum(-1)>1]=1-uv[uv.sum(-1)>1]
    t=tri[ids];extra=t[:,0]+uv[:,:1]*(t[:,1]-t[:,0])+uv[:,1:]*(t[:,2]-t[:,0])
    points=np.concatenate((old,extra)).astype(np.float32)
    normal=np.concatenate((normals,cross[ids]/area[ids,None])).astype(np.float32)
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True).clip(1e-8)
    if points.shape!=(2048,3) or len(np.unique(points,axis=0))!=2048:raise ValueError('2048 distinct surface points required')
    return dict(surface=points,normals=normal,query=np.asarray(query,np.float32))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=7200);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);(a.output/'geometry').mkdir(exist_ok=True)
    started=time.monotonic();plan=json.loads(a.plan.read_text());needed={}
    for job in plan['jobs']:
        for name in job['objects']:
            sid=job['source_id'];key=f's{sid}:{name}' if sid<3 else name
            needed[key]=(sid,name,job)
    native=json.loads((PREP/'native3d-clean-mixed-20261015-r1/manifest.json').read_text())
    objects={r['key']:r for r in native['objects']}
    taco=zipfile.ZipFile(PREP/'taco-native-full-20261014-r1/archives/Object_Models.zip')
    taco_names={Path(n).name:n for n in taco.namelist()}
    h2o=zipfile.ZipFile(PREP/'h2o-native-full-20261014-r1/archives/object.zip')
    hoi=zipfile.ZipFile(PREP/'hoi4d-native-full-20261014-r1/archives/HOI4D_CAD_Model_for_release.zip')
    hoi_names=hoi.namelist();rows=[];failures=[]
    for key,(sid,name,job) in sorted(needed.items()):
        if time.monotonic()-started>a.seconds or (a.output/'STOP').exists() or shutil.disk_usage(a.output).free<20*2**30:break
        try:
            dest='geometry/'+hashlib.sha256(key.encode()).hexdigest()[:20]+'.npz'
            if sid<3:
                source=MAIN/('oakink2-wm30-k24-20261006' if sid==0 else 'native-official-full-20261007')
                with np.load(source/'processed/canonical'/(name+'.npz'),allow_pickle=False) as z:
                    old,normals=z['points'],z['normals']
                    path=str(z['mesh_path']) if 'mesh_path' in z.files else None
                query=old[np.linspace(0,len(old)-1,32,dtype=int)]
                if sid==0:mesh=mesh_file(path)
                elif sid==1:
                    path=ROOT/'data/raw_data/GRAB/tools/object_meshes/contact_meshes'/(name[len('grab_'):]+'.ply')
                    mesh=mesh_file(path)
                else:
                    obj,part=name[len('arctic_'):].rsplit('_',1)
                    assets=ROOT/'data/raw_data/ARCTIC/arctic/data/arctic_data/data/meta/object_vtemplates'/obj
                    mesh=mesh_file(assets/'mesh.obj');mesh.vertices=np.asarray(mesh.vertices)/1000
                    labels=np.asarray(json.loads((assets/'parts.json').read_text()),bool)
                    faces=mesh.faces[(labels[mesh.faces]==(part=='bottom')).all(-1)]
                    mesh=trimesh.Trimesh(vertices=mesh.vertices,faces=faces,process=False);mesh.remove_unreferenced_vertices()
            else:
                row=objects[name]
                with np.load(row['path'],allow_pickle=False) as z:old,normals,query=z['surface'],z['normals'],z['query']
                if sid==4:
                    path=ROOT/'outputs/video-data-preparation/unidexgrasp-policy-assets-20261014-r1/assets/meshdatav3_scaled'/row['mesh_id']/'coacd'/('decomposed_%03d.obj'%round(row['scale']*100))
                    mesh=mesh_file(path)
                elif sid==5:
                    member=taco_names[name.split('/')[-1]+'_cm.obj']
                    mesh=trimesh.load(io.BytesIO(taco.read(member)),file_type='obj',process=False);mesh.vertices*=.01
                elif sid==6:mesh=mesh_file(PREP/'hot3d-native-full-20261014-r1/native/assets/assets'/(name+'.glb'))
                elif sid==8:
                    sys.path.insert(0,str(ROOT/'src/task/3d-data-preparation/tools/run'))
                    from convert_dexycb_full import CLASSES
                    mesh=mesh_file(ROOT/'data/raw_data/DexYCB/raw/models'/CLASSES[int(name.split('/')[-1])-1]/'textured_simple.obj')
                elif sid==9:
                    obj=name.split('/')[-1]
                    member=next(n for n in h2o.namelist() if n.startswith('object/'+obj+'/') and n.endswith('.obj'))
                    mesh=trimesh.load(io.BytesIO(h2o.read(member)),file_type='obj',process=False)
                elif sid==10:
                    category,instance=name.split('/')[1:]
                    sys.path.insert(0,str(ROOT/'src/task/3d-data-preparation/tools/run'))
                    from convert_hoi4d_native import CLASSES
                    member=f'HOI4D_CAD_Model_for_release/rigid/{CLASSES[int(category[1:])]}/{int(instance[1:]):03d}.obj'
                    mesh=trimesh.load(io.BytesIO(hoi.read(member)),file_type='obj',process=False)
            values=sample(mesh,old,normals,query,key)
            pending=(a.output/dest).with_suffix('.part.npz');np.savez_compressed(pending,**values);pending.replace(a.output/dest)
            rows.append(dict(key=key,path=dest,source_id=sid,surface_points=2048,flow_points=32))
        except Exception as e:failures.append(dict(key=key,source_id=sid,reason=type(e).__name__+': '+str(e)))
        if (len(rows)+len(failures))%50==0:print(json.dumps(dict(done=len(rows),failed=len(failures),total=len(needed),elapsed_s=time.monotonic()-started)),flush=True)
    result=dict(schema='ref2dex.wam-v5-object-surfaces.v1',status='COMPLETED' if len(rows)==len(needed) else 'PARTIAL',
        objects=rows,failures=failures,total=len(needed),elapsed_s=time.monotonic()-started,
        scope='Deterministic area-weighted original CAD samples; same coordinate frame; preserve original32 flow queries.')
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('objects','failures')}),flush=True)


if __name__=='__main__':main()
