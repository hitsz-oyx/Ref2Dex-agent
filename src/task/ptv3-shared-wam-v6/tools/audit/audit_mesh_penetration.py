"""Watertight CAD ray/triangle penetration check on existing fixed-panel forecasts.

The HAND is still the recorded-observation bone-transport proxy. Only the object
inside/distance test is mesh-based. Existing prediction32 extends CAD affinely.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import trimesh
from pytorch3d.loss.point_mesh_distance import point_face_distance

sys.path.insert(0,str(Path(__file__).resolve().parent))
import audit_identifiability as audit
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.geometry_consistency import skin_attachment,skin_future


def ray_inside(points,triangles,direction):
    direction=torch.tensor(direction,device=points.device);direction=direction/direction.norm()
    hits=torch.zeros(len(points),device=points.device,dtype=torch.int64)
    for tris in triangles.split(2048):
        edge1,edge2=tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]
        h=torch.cross(direction[None].expand_as(edge2),edge2,dim=-1);det=(edge1*h).sum(-1)
        inverse=torch.where(det.abs()>1e-10,1/det,0.)
        s=points[:,None]-tris[None,:,0];u=(s*h).sum(-1)*inverse
        q=torch.cross(s,edge1[None].expand_as(s),dim=-1)
        v=(q*direction).sum(-1)*inverse;t=(q*edge2).sum(-1)*inverse
        hit=(det.abs()[None]>1e-10)&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-7)
        hits+=hit.sum(-1)
    return hits%2==1


def mesh_metrics(points,triangles):
    zeros=torch.zeros(1,dtype=torch.long,device=points.device)
    squared=point_face_distance(points.contiguous(),zeros,triangles.contiguous(),zeros,len(points),1e-12)
    distance=squared.clamp_min(0).sqrt()
    first=ray_inside(points,triangles,(1.,.371,.733));second=ray_inside(points,triangles,(.229,1.,.517))
    agree=first==second;inside=agree&first&(distance>.001)
    return dict(vertices=len(points),ray_disagreements=int((~agree).sum()),valid_vertices=int(agree.sum()),
        inside_over1mm=int(inside.sum()),penetration_sum_mm=float(torch.where(inside,distance*1000,0.).sum()),
        unsigned_distance_sum_mm=float(distance[agree].sum()*1000))


def smoke():
    mesh=trimesh.creation.box(extents=(1.,1.,1.));tri=torch.tensor(mesh.triangles.copy(),dtype=torch.float32,device='cuda')
    points=torch.tensor([[0.,0.,0.],[1.,0.,0.],[-.6,.1,.1],[.4,.4,.4]],device='cuda')
    values=mesh_metrics(points,tri)
    if values['inside_over1mm']!=2 or values['ray_disagreements'] or abs(values['penetration_sum_mm']-600)>.01:
        raise ValueError('mesh inside/distance cube control failed')
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','stats','baseline','flow','output','mesh-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--samples',type=int,default=16);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('dedicated authorized GPU0 only')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15,0);start=time.monotonic()
    panel=audit.read(a.output/'panels.json');rows=panel['panels']['heldout_sequence'][:a.samples]
    dataset=V6Windows(a.manifest,'train',geometry_gt_object=True);stats=audit.read(a.stats)
    cube=smoke();records=[];skipped=[];assets={};summaries={}
    for label,path in [('healthy4371',a.baseline),('flow8500',a.flow)]:
        saved=torch.load(path,map_location='cpu',weights_only=False);config=dict(saved['config'],strict_fp32=True,amp=False,tf32=False,fp32_attention_backend='efficient')
        torch.manual_seed(178);model=SharedWAMV6(stats,config).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True);del saved;gc.collect()
        for number,row in enumerate(rows):
            if time.monotonic()-start>900:raise TimeoutError('15min mesh budget')
            audit.guard(a.output);name=row['object'][5:];mesh_path=a.mesh_root/(name+'.ply')
            if name not in assets:
                mesh=trimesh.load(mesh_path,process=False,force='mesh')
                assets[name]=dict(triangles=np.asarray(mesh.triangles,np.float32).copy(),watertight=bool(mesh.is_watertight),path=str(mesh_path.resolve()))
            asset=assets[name]
            if not asset['watertight']:skipped.append(dict(model=label,row=row,reason='nonwatertight CAD'));continue
            sample=audit.window(dataset,row);batch={k:v.cuda() for k,v in collate([sample]).items()}
            with torch.inference_mode():
                pred=model(batch);attachment=skin_attachment(batch['start_hand'],batch['dense_hand'],batch['start_hand_valid'],batch['dense_hand_valid'])
                skin=skin_future(batch['start_hand'][:,None]+pred['trajectory'],attachment)[0]
                gt_skin=skin_future(batch['start_hand'][:,None]+batch['trajectory_target'],attachment)[0]
                p32=batch['scene_points'][0,0];x=torch.cat((p32,torch.ones_like(p32[:,:1])),-1)
                current=batch['interaction_points'][0,0];sx=torch.cat((current,torch.ones_like(current[:,:1])),-1)
                canonical=torch.tensor(asset['triangles'],device='cuda');zero=torch.zeros(1,dtype=torch.long,device='cuda')
                check=point_face_distance(p32.contiguous(),zero,canonical.contiguous(),zero,32,1e-12).sqrt()*1000
                if check.mean()>1:skipped.append(dict(model=label,row=row,reason='CAD/query mean distance>1mm',distance_mm=float(check.mean())));continue
                ids=torch.linspace(0,777,32,device='cuda').long()
                for tick in (8,24):
                    k=tick-1;pred_map=torch.linalg.pinv(x)@(p32+pred['scene'][0,0,k]);gt_map=torch.linalg.pinv(sx)@batch['geometry_object_future_gt'][0,0,k]
                    for anchor,mapping,h in [('GT_object',gt_map,skin[k]),('pred_object',pred_map,skin[k]),('GT_hand_GT_object',gt_map,gt_skin[k])]:
                        triangles=canonical@mapping[:3]+mapping[3]
                        points=h[:,ids].flatten(0,1)
                        values=mesh_metrics(points,triangles)
                        records.append(dict(model=label,row=row,tick=tick,anchor=anchor,asset=asset['path'],faces=len(canonical),
                            CAD_query_distance_mean_mm=float(check.mean()),**values))
            print(json.dumps(dict(stage='mesh',model=label,done=number+1,total=len(rows))),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    for model in ('healthy4371','flow8500'):
        for tick in (8,24):
            for anchor in ('GT_object','pred_object','GT_hand_GT_object'):
                subset=[r for r in records if r['model']==model and r['tick']==tick and r['anchor']==anchor]
                n=sum(r['valid_vertices'] for r in subset);penetrations=sum(r['inside_over1mm'] for r in subset)
                summaries[model+'/'+str(tick)+'/'+anchor]=dict(windows=len(subset),captures=len(set(r['row']['capture'] for r in subset)),valid_vertex_frames=n,
                    inside_over1mm_fraction=penetrations/n if n else None,
                    mean_penetration_over_all_vertices_mm=sum(r['penetration_sum_mm'] for r in subset)/n if n else None,
                    mean_depth_when_penetrating_mm=sum(r['penetration_sum_mm'] for r in subset)/penetrations if penetrations else None,
                    ray_disagreements=sum(r['ray_disagreements'] for r in subset))
    audit.write(a.output/'mesh-penetration.json',dict(status='COMPLETED',cube_control=cube,seed=178,command=sys.argv,
        git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),physical_gpu=0,
        windows_requested=len(rows),ticks=[8,24],vertices_per_hand=32,summary=summaries,skipped=skipped,records=records,
        elapsed_s=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        limitations=['object CAD is watertight;predicted object is an affine extension of32 forecast points',
        'HAND skin remains observed-surface bone transport,not GT future mesh or measured physical hand',
        'fixed16 sequence-panel windows;two times and32vertices/hand,not full-surface continuous collision detection',
        'two generic-ray parity tests must agree;discard disagreements;1mm penetration tolerance;no measured forces']))


if __name__=='__main__':main()
