"""Frozen PW inference with observed-hand oracle conditioning; no deployable A bridge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.data import sha
from consequence_evaluator.contracts import is_within


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--seconds',type=int,default=900);p.add_argument('--batch',type=int,default=8)
    a=p.parse_args();out=a.output.resolve();data=a.data.resolve()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator') or not 1<=a.seconds<=900 or not 1<=a.batch<=16:
        raise ValueError('fresh bounded PW inference required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    import trimesh
    from consequence_evaluator.old_utility import SCHEMA,pw_sample,future_from_prediction
    PW=ROOT/'src/task/cm-pointflow-effect-pretrain';sys.path.insert(0,str(PW/'src'))
    from oakink_wm.pointworld_temporal import model_from_config,capped_collate,VENDOR
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2);torch.manual_seed(292);torch.backends.cuda.matmul.allow_tf32=True
    install_fused_hilbert();start=time.monotonic()
    m=json.loads((data/'manifest.json').read_text())
    if m['schema']!=SCHEMA or m['windows_sha256']!=sha(data/'windows.npz'):raise ValueError('fixed old-U pack identity required')
    with np.load(data/'windows.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    checkpoint=a.checkpoint.resolve();meshpath=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects/airplane/airplane.obj'
    cfg=ROOT/'outputs/consequence-evaluator/baseline-transfer-inputs-20261007-r1/cfg/inspire.yaml'
    import yaml
    if yaml.safe_load(cfg.read_text())['env']['ballSize']!=1:raise ValueError('canonical scale must match native simulator')
    hashes={str(path):sha(path) for path in (data/'manifest.json',data/'windows.npz',checkpoint,meshpath,cfg,
        Path(__file__).resolve(),TASK/'src/consequence_evaluator/old_utility.py')}
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for base,key in ((PW,'implementation_sources'),(VENDOR,'vendor_sources')):
        for name,digest in state['identity'][key].items():
            path=(base/name).resolve()
            if sha(path)!=digest:raise ValueError('PW implementation identity drift: '+str(path))
            hashes[str(path)]=digest
    model=model_from_config(state['identity']['stats'],state['config']).cuda().eval();model.load_state_dict(state['model'])
    del state
    mesh=trimesh.load(meshpath,force='mesh',process=False);tri=np.asarray(mesh.triangles);rng=np.random.default_rng(292)
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=-1)
    ids=rng.choice(len(tri),512,p=area/area.sum());u=np.sqrt(rng.random(512));v=rng.random(512)
    points=tri[ids,0]*(1-u[:,None])+tri[ids,1]*(u*(1-v))[:,None]+tri[ids,2]*(u*v)[:,None]
    canonical=dict(points=points.astype('float32'),normals=(cross[ids]/area[ids,None]).astype('float32'),
        center=np.asarray(mesh.centroid,'float32'),radius=float(np.sqrt(np.mean(np.sum((points-points.mean(0))**2,-1)))))
    out.mkdir(parents=True);np.savez_compressed(out/'canonical.npz',**canonical)
    manifest=dict(schema=SCHEMA,status='RUNNING',checkpoint=str(checkpoint),checkpoint_sha256=hashes[str(checkpoint)],
        checkpoint_step=46000,oracle_observed_hand=True,deployable_planner=False,physical_gpu=a.gpu,
        input_sha256=hashes,windows_sha256=m['windows_sha256'],rows=len(d['label']),batch=a.batch,budget_s=a.seconds,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');future=np.zeros_like(d['future'])
    with torch.inference_mode():
        for begin in range(0,len(future),a.batch):
            if time.monotonic()-start>a.seconds:raise TimeoutError('PW inference deadline')
            end=min(begin+a.batch,len(future))
            samples=[pw_sample(d['pw_object_history'][i],d['pw_hand_history'][i],d['pw_hand_future'][i],canonical,i)
                     for i in range(begin,end)]
            batch={k:v.cuda() for k,v in capped_collate(samples).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(batch,'action')
            rotation=pred['rotation'][:,0].float().cpu().numpy();translation=pred['translation'][:,0].float().cpu().numpy()
            if begin==0:
                with torch.autocast('cuda',dtype=torch.bfloat16):repeat=model(batch,'action')
                if not torch.equal(pred['rotation'],repeat['rotation']) or not torch.equal(pred['translation'],repeat['translation']):
                    raise ValueError('PW inference is not repeatable')
            for j,i in enumerate(range(begin,end)):
                future[i]=future_from_prediction(rotation[j],translation[j],d['pw_hand_future'][i])
            if begin==0 or end%(64*a.batch)==0 or end==len(future):
                elapsed=time.monotonic()-start;gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                    '--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                print(json.dumps(dict(rows=end,total=len(future),elapsed_s=round(elapsed,1),
                    eta_s=round(elapsed/end*(len(future)-end),1),gpu=gpu)),flush=True)
    np.savez_compressed(out/'future.npz',future=future)
    gt=d['future'][:,:,:12].reshape(-1,24,3,4);pw=future[:,:,:12].reshape(-1,24,3,4)
    errors={}
    for split in ('train','val','test','panel'):
        rows=d['split']==split
        relative=pw[rows,:,:,:3]@gt[rows,:,:,:3].swapaxes(-1,-2)
        angles=np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))
        errors[split]=dict(object_h24_translation_error_m=float(np.linalg.norm(pw[rows,-1,:,3]-gt[rows,-1,:,3],axis=-1).mean()),
            object_h24_rotation_error_rad=float(angles[:,-1].mean()))
    if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('PW inputs drifted')
    manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-start,future_sha256=sha(out/'future.npz'),
        canonical_sha256=sha(out/'canonical.npz'),errors=errors,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(errors,indent=2),flush=True)


if __name__=='__main__':main()
