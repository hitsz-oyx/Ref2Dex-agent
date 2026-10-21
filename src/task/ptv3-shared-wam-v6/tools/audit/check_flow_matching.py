"""Real-input FM gradients, teacher-field learning and deterministic ODE audit."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import torch

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.objectives import forecast_loss
from ptv3_shared_wam_v6.geometry_consistency import geometry_components
from ptv3_shared_wam_v6.local_subset import initialize_flow_weights
from ptv3_shared_wam_v6.local_scale import data_identity
from ptv3_shared_wam_v6.flow_matching import pack,current_mask,linear_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','manifest','stats','checkpoint','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--initialize',action='store_true');args=parser.parse_args()
    args.output.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(170);torch.cuda.set_device(0)
    config=json.loads(args.config.read_text());stats=json.loads(args.stats.read_text())
    model=SharedWAMV6(stats,config).cuda()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if args.initialize:initialize_flow_weights(model,saved,args.checkpoint,config,data_identity(args.manifest,args.stats))
    else:model.load_state_dict(saved['model'],strict=True)
    source_step=saved['step'];del saved;gc.collect()
    dataset=V6Windows(args.manifest,'train',geometry_gt_object=True)
    indices=dataset.draw(4,170,0,0)
    batch={k:v.cuda() for k,v in collate([dataset[i] for i in indices]).items()}
    model.train();torch.cuda.reset_peak_memory_stats();prediction=model(batch)
    loss,terms=forecast_loss(prediction,batch,config,model.normalizer,step=100)
    gradients=torch.autograd.grad(loss,(model.flow_future.output.weight,prediction['trajectory'],prediction['scene']),retain_graph=True)
    components,_=geometry_components(prediction,batch,config)
    geo=sum(n/d.clamp_min(1)*(config['geometry_clearance_weight'] if k=='geometry_clearance' else 1)
            for k,(n,d) in components.items())
    geometry_gradients=torch.autograd.grad(geo,(prediction['trajectory'],prediction['scene']),allow_unused=True)
    if (any(not torch.isfinite(g).all() or g.norm()==0 for g in gradients)
            or geometry_gradients[0] is None or geometry_gradients[0].norm()==0
            or geometry_gradients[1] is not None):
        raise ValueError('FM or GT-object geometry gradient contract failed')
    report=dict(status='PASS',checkpoint=str(args.checkpoint.resolve()),source_step=source_step,
        indices=indices,scope='four fixed TRAIN samples; no optimizer update',
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        precision=model.precision_runtime,training_loss=float(loss),
        flow_terms={k:float(terms[k]) for k in ('flow_hand','flow_scene')},
        total_gradient_norms=[float(g.norm()) for g in gradients],
        geometry_direct_hand_norm=float(geometry_gradients[0].norm()),geometry_direct_scene_gradient=None,
        parameter_tensors=len(list(model.parameters())))
    del prediction,loss,terms,gradients,components,geo,geometry_gradients;gc.collect()
    model.eval()
    with torch.no_grad():
        config['flow_matching']=False
        try:base=model(batch,return_shared=True)
        finally:config['flow_matching']=True
        shared=base['shared_tokens'];valid=base['shared_valid'];flow=model.flow_future
        hand=base['trajectory_normalized'];scene=base['scene_normalized']
        generated=flow(hand,scene,shared,valid,batch,model.normalizer)
        repeated=flow(hand,scene,shared,valid,batch,model.normalizer)
        if not all(torch.equal(a,b) for a,b in zip(generated[:2],repeated[:2])):
            raise ValueError('fixed-seed ODE sampler is not deterministic')
        config['flow_eval_seed']+=1
        alternate=flow(hand,scene,shared,valid,batch,model.normalizer)
        config['flow_eval_seed']-=1
        previous_steps=config['flow_steps'];config['flow_steps']=8
        finer=flow(hand,scene,shared,valid,batch,model.normalizer);config['flow_steps']=previous_steps
        mask=current_mask(batch);center=pack(hand,scene)
        hv=batch['trajectory_valid']&batch['start_hand_valid'][:,None]
        sv=batch['scene_valid']&batch['object_valid'][:,:,None,None]
        ht=model.normalizer.normalize('hand',torch.where(hv[...,None],batch['trajectory_target'],0.))
        st=model.normalizer.normalize('scene',torch.where(sv[...,None],batch['scene_target'],0.))
        supervised=pack(hv[...,None].expand_as(ht),sv[...,None].expand_as(st))
        residual_target=torch.where(supervised,pack(ht,st)-center,0.)
        noise=torch.where(mask,flow.noise(center,batch),0.)
        mixed,target=linear_path(residual_target,noise,torch.linspace(.1,.9,len(center),device='cuda'))
        velocity=flow.velocity(mixed,torch.linspace(.1,.9,len(center),device='cuda'),center,flow.context(shared),valid,mask)
        mse=torch.where(supervised,(velocity-target).square(),0.).sum()/supervised.sum().clamp_min(1)
        zero=torch.where(supervised,target.square(),0.).sum()/supervised.sum().clamp_min(1)
        sample=pack(*generated[:2]);different=pack(*alternate[:2]);refined=pack(*finer[:2])
        if not all(torch.isfinite(x).all() for x in (sample,different,refined,mse)):
            raise ValueError('nonfinite FM sample')
        report.update(teacher_velocity_mse=float(mse),zero_velocity_mse=float(zero),
            teacher_to_zero_ratio=float(mse/zero.clamp_min(1e-12)),
            sample_seed_difference_rms_normalized=float((sample-different).square().mean().sqrt()),
            euler4_vs8_rms_normalized=float((sample-refined).square().mean().sqrt()),
            sampler_repeat_exact=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            elapsed_s=time.monotonic()-start)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
