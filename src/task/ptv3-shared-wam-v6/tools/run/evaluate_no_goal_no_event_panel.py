#!/usr/bin/env python3
"""Same frozen TRAIN panel under A for init/trained; cluster paired inference only."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cluster_ci(sums,counts,mothers,reference=None,repeats=4000):
    import numpy as np
    ids=np.unique(mothers);s=np.asarray([sums[mothers==m].sum(0) for m in ids]);c=np.asarray([counts[mothers==m].sum(0) for m in ids]);rng=np.random.default_rng(170)
    other=None if reference is None else np.asarray([reference[mothers==m].sum(0) for m in ids])
    result=[]
    for _ in range(repeats):
        chosen=rng.integers(0,len(ids),len(ids));den=c[chosen].sum(0);value=s[chosen].sum(0)/den
        if other is not None:value-=other[chosen].sum(0)/den
        result.append(value)
    return np.percentile(result,[2.5,97.5],axis=0).T.tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--gpu',type=int,default=7);p.add_argument('--batch-size',type=int,default=4);args=p.parse_args()
    if args.gpu!=7:raise ValueError('panel only authorized on GPU7 spare share')
    if args.output.exists():raise FileExistsError(args.output)
    free=int(subprocess.check_output(['nvidia-smi','-i','7','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
    if free<8*2**30 or shutil.disk_usage('/home2').free<20*2**30:raise RuntimeError('panel resource preflight failed')
    os.environ['CUDA_VISIBLE_DEVICES']='7';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']='2'
    import numpy as np
    import torch
    from ptv3_shared_wam_v6.data import V6Windows,collate
    from ptv3_shared_wam_v6.model import SharedWAMV6
    from ptv3_shared_wam_v6.goal_policy import UNKNOWN_GOAL_POLICY,assert_unknown_inputs
    from ptv3_shared_wam_v6.events import UNKNOWN_EVENT_POLICY
    torch.set_num_threads(2);torch.manual_seed(170);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    panel=read(TASK/'configs/tau-fixed-panel-v1.json');anchor=read(ROOT/panel['anchor']);rows=anchor['rows']
    ds=V6Windows(ROOT/panel['event_recipe'],'train',goal_input_policy=UNKNOWN_GOAL_POLICY,event_input_policy=UNKNOWN_EVENT_POLICY)
    lookup={tuple(map(int,x)):i for i,x in enumerate(ds.index)};positions=[lookup[(r['sequence_id'],r['slot'],r['tick'])] for r in rows]
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);config=dict(checkpoint['config'],goal_input_policy=UNKNOWN_GOAL_POLICY,event_input_policy=UNKNOWN_EVENT_POLICY)
    model=SharedWAMV6(read(ROOT/panel['stats']),config).cuda().eval().requires_grad_(False);model.load_state_dict(checkpoint['model'],strict=True);step=checkpoint['step'];del checkpoint
    start=time.monotonic();sums=[];counts=[];hands=[];scenes=[];hand_masks=[];scene_masks=[];repeated=False;repeat_max=None
    for offset in range(0,len(rows),args.batch_size):
        if time.monotonic()-start>1200:raise TimeoutError('bounded panel inference exceeds1200s; training deadline is separate')
        batch={k:v.cuda() for k,v in collate([ds[i] for i in positions[offset:offset+args.batch_size]]).items()};assert_unknown_inputs(batch)
        with torch.inference_mode():
            pred=model(batch)
            if offset==0:
                repeat=model(batch);repeated=bool(torch.equal(pred['trajectory'],repeat['trajectory']) and torch.equal(pred['scene'],repeat['scene']));repeat_max=float((pred['trajectory']-repeat['trajectory']).abs().max())*1000
        hv=batch['trajectory_valid']&batch['start_hand_valid'][:,None];sv=batch['scene_valid']&batch['object_valid'][:,:,None,None]
        cf=(batch['interaction_valid']&(batch['contact_target']>.25)).any((1,3));f=torch.arange(24,device='cuda')[None]<8
        masks=[hv&f[:,:,None,None],hv&cf[:,:,None,None],sv&f[:,None,:,None]]
        errors=[(pred['trajectory']-batch['trajectory_target']).norm(dim=-1)*1000]*2+[(pred['scene']-batch['scene_target']).norm(dim=-1)*1000]
        sums.append(torch.stack([torch.where(m,e,0.).flatten(1).sum(1,dtype=torch.float64) for e,m in zip(errors,masks)],1).cpu().numpy());counts.append(torch.stack([m.flatten(1).sum(1) for m in masks],1).cpu().numpy())
        hands.append(pred['trajectory'].cpu().numpy());scenes.append(pred['scene'].cpu().numpy());hand_masks.append(torch.stack(masks[:2],1).cpu().numpy());scene_masks.append(masks[2].cpu().numpy())
        print(json.dumps(dict(rows=min(offset+args.batch_size,len(rows)),elapsed_s=time.monotonic()-start)),flush=True)
    sums=np.concatenate(sums);counts=np.concatenate(counts);mothers=np.array([r['sequence_id'] for r in rows]);hand=np.concatenate(hands);scene=np.concatenate(scenes);hm=np.concatenate(hand_masks);sm=np.concatenate(scene_masks)
    means=sums.sum(0)/counts.sum(0);ci=cluster_ci(sums,counts,mothers)
    difference=None
    if args.reference:
        with np.load(args.reference/'paired-panel.npz') as z:
            assert np.array_equal(z['mothers'],mothers) and np.array_equal(z['counts'],counts)
            hs=np.linalg.norm(hand-z['hand'],axis=-1)*1000;ss=np.linalg.norm(scene-z['scene'],axis=-1)*1000
            changes=[float(np.where(hm[:,i],hs,0.).sum()/hm[:,i].sum()) for i in range(2)]+[float(np.where(sm,ss,0.).sum()/sm.sum())]
            difference=dict(epe_delta_mm=(means-z['sums'].sum(0)/counts.sum(0)).tolist(),paired_cluster_ci95_mm=cluster_ci(sums,counts,mothers,z['sums']),prediction_change_mm=changes)
    args.output.mkdir(parents=True);np.savez(args.output/'paired-panel.npz',sums=sums,counts=counts,mothers=mothers,hand=hand,scene=scene)
    report=dict(schema='ref2dex.no-goal-no-event.fixed-panel.v1',scope='TRAIN distribution only; no new rollout or executor benefit',checkpoint=str(args.checkpoint),step=step,rows=len(rows),mothers=len(np.unique(mothers)),
        metrics_order=['first8_hand','fixed_contact_hand','first8_object'],means_mm=means.tolist(),cluster_ci95_mm=ci,bootstrap_repeats=4000,paired_vs_reference=difference,
        same_A_conditions=True,goal_policy=UNKNOWN_GOAL_POLICY,event_policy=UNKNOWN_EVENT_POLICY,event_valid_count=0,goal_valid_count=0,contact_definition=panel['metrics']['contact_window'],contact_threshold=.25,
        repeated_prediction_bit_exact=repeated,repeat_max_abs_hand_mm=repeat_max,elapsed_s=time.monotonic()-start,physical_gpu=7,free_gpu_bytes_before=free,peak_allocated_bytes=torch.cuda.max_memory_allocated(),anchor_sha256=sha(ROOT/panel['anchor']))
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
