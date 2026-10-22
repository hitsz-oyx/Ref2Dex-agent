"""Full-DEV t8 watertight CAD occupancy, with affine depth bounds.

Neural endpoint pass uses authorized GPU; CPU BVH signed-distance queries reuse
the same ordered CAD across all windows. Hand remains observed-skin transport.
Predicted affine object: containment is exact under invertible affine maps;
world distance is bounded by canonical SDF times min/max singular values.
"""
import argparse
from collections import defaultdict,Counter
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from audit_goal_chain import (ROOT,OLDOUT,MANIFEST,NOISE_SEEDS,GOALS,load,dump,condition,
    SharedWAMV6,V6Windows,collate,old,pack,check,skin_attachment,skin_future)


def endpoint_forecasts(model,batch):
    shared,valid,geometry=model.encode(batch);offset=geometry['tokens'].shape[1]
    hz=model.hand_future(shared[:,offset:offset+42]).reshape(len(shared),24,2,21,3)
    points=batch['scene_points'];ov=batch['object_valid'].bool();pv=ov[...,None].expand(*points.shape[:-1])
    query=model.scene_query(points/model.position_scale).flatten(1,2);context=model.scene_context(shared)
    update,_=model.scene_readout(query,context,context,key_padding_mask=~valid,need_weights=False)
    latent=model.scene_norm(query+update).reshape(len(points),points.shape[1],points.shape[2],-1)
    sz=model.scene_future(latent.flatten(1,2)).reshape(len(points),24,*points.shape[1:3],3).transpose(1,2)
    enabled=model.config.get('flow_matching',False);seeds=NOISE_SEEDS if enabled else (None,);result=[]
    original=model.flow_future.config['flow_eval_seed'] if enabled else None
    for seed in seeds:
        if enabled:
            model.flow_future.config['flow_eval_seed']=seed
            hand,scene,_=model.flow_future(hz,sz,shared,valid,batch,model.normalizer)
        else:hand,scene=hz,sz
        ht=torch.where(batch['start_hand_valid'][:,None,...,None],model.normalizer.denormalize('hand',hand),0.)
        st=torch.where(ov[:,:,None,None,None],model.normalizer.denormalize('scene',scene),0.)
        result.append(pack(ht,st)[:,7])
    if enabled:model.flow_future.config['flow_eval_seed']=original
    return torch.stack(result,1)


def predict(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1','2','3'):raise ValueError('one authorized GPU only')
    out=a.output;started=time.monotonic();torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(12*2**30/torch.cuda.get_device_properties(0).total_memory)
    rows=[r for r in load(out/'panels.json')['rows'] if r['evaluate'] and r['panel']=='heldout_sequence']
    seqs=sorted(set(r['seq_id'] for r in rows));owned=set(seqs[a.shard::2]);rows=[r for r in rows if r['seq_id'] in owned]
    rows.sort(key=lambda r:(r['seq_id'],r['tick']));data=V6Windows(MANIFEST,'train');stats=load(MANIFEST.parent/'stats.json')
    with np.load(out/'planner-error-distribution.npz') as z:errors={k:z[k] for k in z.files}
    for label,path in [('healthy4371',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-ddp-20261021-r1/training/step004371.pt'),('flow8500',OLDOUT/'flow-step008500.pt')]:
        saved=torch.load(path,map_location='cpu',weights_only=False);config=dict(saved['config'],strict_fp32=True,amp=False,tf32=False,fp32_attention_backend='efficient')
        model=SharedWAMV6(stats,config).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'],strict=True);del saved;gc.collect()
        folder=out/'mesh-endpoints'/('worker%d'%a.shard)/label;folder.mkdir(parents=True,exist_ok=True)
        for first in range(0,len(rows),64):
            check(out,started,3600);selected=rows[first:first+64];path=folder/('%06d.npz'%first)
            if path.exists():continue
            if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('mesh endpoint inference GPU headroom below2GiB')
            batch={k:v.cuda() for k,v in collate([old.window(data,r) for r in selected]).items()};values=[]
            with torch.inference_mode():
                for goal in GOALS:
                    inputs=condition(batch,selected,goal,errors,181);pred=endpoint_forecasts(model,inputs)
                    if first==0 and goal=='oracle':
                        native=model(inputs);delta=float((pack(native['trajectory'],native['scene'])[:,7]-pred[:,0]).abs().max())
                        if delta>1e-6:raise AssertionError('trajectory-only inference differs from native forward')
                        dump(out/('mesh-native-%s-worker%d.json'%(label,a.shard)),dict(status='PASS',max_abs_m=delta))
                    values.append(pred.cpu().numpy())
            with open(str(path)+'.part','wb') as f:np.savez_compressed(f,endpoints=np.stack(values,1),ids=np.array([[r['seq_id'],r['slot'],r['tick']] for r in selected]))
            Path(str(path)+'.part').replace(path)
            dump(out/('mesh-progress-worker%d.json'%a.shard),dict(status='RUNNING',model=label,done=first+len(selected),total=len(rows),elapsed_s=time.monotonic()-started,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES']))
        del model;gc.collect();torch.cuda.empty_cache()
    dump(out/('complete-mesh-endpoints-worker%d.json'%a.shard),dict(status='COMPLETED',elapsed_s=time.monotonic()-started,windows=len(rows)))


def measure(a):
    import pysdf
    import trimesh
    torch.set_num_threads(2);out=a.output;rows=load(out/'panels.json')['rows'];lookup={tuple(r[k] for k in ('seq_id','slot','tick')):r for r in rows}
    data=V6Windows(MANIFEST,'train');cache={};summary={};skipped=Counter();records=[];gt_calibration={}
    cube=trimesh.creation.box(extents=(1.,1.,1.));sdf=pysdf.SDF(np.asarray(cube.vertices,np.float32),np.asarray(cube.faces,np.uint32))
    control=sdf(np.array([[0,0,0],[1,0,0],[.4,.4,.4]],np.float32),n_threads=2)
    if not np.allclose(control,[.5,-.5,.1],atol=1e-6):raise AssertionError('signed mesh cube control failed')
    meshroot=ROOT/'data/raw_data/GRAB/tools/object_meshes/contact_meshes'
    for folder in sorted((out/'mesh-endpoints').glob('worker[0-9]')):
        for modeldir in sorted(folder.iterdir()):
            for path in sorted(modeldir.glob('*.npz')):
                with np.load(path) as z:ids=z['ids'];endpoints=z['endpoints']
                for identity,ep in zip(ids,endpoints):
                    key=tuple(map(int,identity));r=lookup[key];label=modeldir.name
                    if r['source_id']!=1:skipped[label+'/non_GRAB_asset_unmapped']+=1;continue
                    name=r['object'][5:];meshpath=meshroot/(name+'.ply')
                    if name not in cache:
                        mesh=trimesh.load(meshpath,process=False,force='mesh');cache[name]=dict(watertight=bool(mesh.is_watertight),sdf=pysdf.SDF(np.asarray(mesh.vertices,np.float32),np.asarray(mesh.faces,np.uint32)))
                    asset=cache[name]
                    if not asset['watertight']:skipped[label+'/non_watertight']+=1;continue
                    s=old.window(data,r);b=collate([s]);att=skin_attachment(b['start_hand'],b['dense_hand'],b['start_hand_valid'],b['dense_hand_valid'])
                    present=(b['start_hand_valid']&b['trajectory_valid'][:,7]).all(-1)[0].numpy()
                    verts=np.linspace(0,777,64,dtype=int);valid=b['dense_hand_valid'][0].numpy()[:,verts]&present[:,None]
                    valid&=att[3][0].numpy()[:,verts]
                    p0=s['scene_points'][0];x=np.c_[p0,np.ones(32)];seq,_=data._sequence(r['seq_id']);cur=seq['object_poses'][r['tick'],r['slot']];future=seq['object_poses'][r['tick']+8,r['slot']]
                    transform=np.linalg.inv(cur)@future;gtmap=np.r_[transform[:3,:3].T,transform[:3,3][None]]
                    # Per-asset CAD/query registration check is independent of predictions.
                    if name not in gt_calibration:
                        distance=asset['sdf'](np.ascontiguousarray(p0,np.float32),n_threads=2)
                        if np.abs(distance).mean()>.001:raise AssertionError('CAD/query registration>1mm')
                        gt_calibration[name]=dict(mean_abs_CAD_query_distance_mm=float(np.abs(distance).mean()*1000))
                    record=dict(model=label,identity=list(key),goals={})
                    for gi,goal in enumerate(GOALS):
                        # One deterministic or eight FM t8 draws. Skin construction
                        # is the exact same attachment as prefix geometry diagnostics.
                        hz=torch.tensor(ep[gi,:,:126].reshape(-1,1,2,21,3))
                        attachment=tuple(v.expand(len(hz),*v.shape[1:]) for v in att)
                        hand=skin_future(b['start_hand'].expand(len(hz),-1,-1,-1)[:,None]+hz,attachment).numpy()[:,0][:,:,verts]
                        predicted=p0[None]+ep[gi,:,126:].reshape(-1,32,3);maps=np.linalg.pinv(x)[None]@predicted
                        anchor_metrics={}
                        for anchor,mapping in [('GT_object',np.broadcast_to(gtmap,maps.shape)),('own_object',maps)]:
                            validmaps=np.linalg.svd(mapping[:,:3],compute_uv=False);nonsingular=validmaps[:,-1]>1e-4
                            local=(hand-mapping[:,None,None,3])@np.linalg.pinv(mapping[:,:3])[:,None]
                            distances=asset['sdf'](np.ascontiguousarray(local.reshape(-1,3),np.float32),n_threads=2).reshape(len(hz),2,64)
                            inside=distances>0;lower=np.maximum(distances,0)*validmaps[:,-1,None,None]*1000;upper=np.maximum(distances,0)*validmaps[:,0,None,None]*1000
                            known=np.broadcast_to(valid,(len(hz),2,64))&nonsingular[:,None,None]
                            counts=dict(vertices=int(known.sum()),inside=int((inside&known).sum()),
                                certified_inside_over1mm=int(((lower>1)&known).sum()),possible_inside_over1mm=int(((upper>1)&known).sum()),
                                depth_lower_sum_mm=float(np.where(known,lower,0).sum()),depth_upper_sum_mm=float(np.where(known,upper,0).sum()),
                                singular_draws=int((~nonsingular).sum()))
                            sk=(label,goal,anchor)
                            if sk not in summary:summary[sk]=dict(windows=0,captures=set(),vertices=0,inside=0,certified_inside_over1mm=0,possible_inside_over1mm=0,depth_lower_sum_mm=0.,depth_upper_sum_mm=0.,singular_draws=0)
                            summary[sk]['windows']+=1;summary[sk]['captures'].add(r['capture'])
                            for k,v in counts.items():summary[sk][k]+=v
                            anchor_metrics[anchor]=counts
                        record['goals'][goal]=anchor_metrics
                    records.append(record)
    result={}
    for key,v in summary.items():
        v['captures']=len(v['captures']);v['inside_fraction']=v['inside']/max(v['vertices'],1)
        v['mean_depth_lower_mm']=v['depth_lower_sum_mm']/max(v['vertices'],1);v['mean_depth_upper_mm']=v['depth_upper_sum_mm']/max(v['vertices'],1)
        result['/'.join(key)]=v
    dump(out/'large-mesh-penetration.json',dict(status='COMPLETED',tick=8,hand_vertices_per_hand=64,cube_control_m=control.tolist(),
        results=result,skipped=dict(skipped),CAD_registration_controls=gt_calibration,records=records,
        interpretation='Watertight CAD inside test on128 sampled transported hand-skin vertices per draw;t8 only. Exact affine containment,world penetration depth bounded by canonical positive SDF times min/max singular values. Singular maps excluded/count reported;not full robot hand geometry.'))
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=('predict','measure'),required=True);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.phase=='predict':predict(a)
    else:measure(a)


if __name__=='__main__':main()
