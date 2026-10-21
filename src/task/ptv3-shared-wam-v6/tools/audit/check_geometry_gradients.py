"""Real-checkpoint TRAIN-only geometry gradient, coverage and weight preflight."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.local_subset import initialize_geometry_weights
from ptv3_shared_wam_v6.local_scale import data_identity
from ptv3_shared_wam_v6.geometry_consistency import geometry_components
from ptv3_shared_wam_v5.objectives import forecast_loss as base_loss


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','manifest','stats','checkpoint','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');a=parser.parse_args();started=time.monotonic()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('CUDA real-checkpoint preflight required')
    torch.set_num_threads(2);torch.manual_seed(170);torch.cuda.set_device(a.device)
    config=json.loads(a.config.read_text());stats=json.loads(a.stats.read_text())
    torch.backends.cuda.matmul.allow_tf32=config.get('tf32',True)
    model=SharedWAMV6(stats,config).to(a.device)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    binding=initialize_geometry_weights(model,saved,a.checkpoint,config,data_identity(a.manifest,a.stats));del saved
    ds=V6Windows(a.manifest,'train');samples=[];indices=[]
    for index,_ in ds.draw(64,170,0,0):
        row=ds[index]
        if ((row['contact_target']>.25)&row['interaction_valid']).any():
            samples.append(row);indices.append(index)
        if len(samples)==4:break
    if len(samples)<4:raise ValueError('four real TRAIN contact samples required for gradient calibration')
    batch={k:v.to(a.device) for k,v in collate(samples).items()}
    torch.cuda.reset_peak_memory_stats();model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.get('amp',True)):
        prediction=model(batch)
    base,base_terms=base_loss(prediction,batch,config,model.normalizer)
    components,diagnostics=geometry_components(prediction,batch,config,torch.float64)
    means={k:n/d.clamp_min(1) for k,(n,d) in components.items()}
    geometry=means['geometry_contact']+config['geometry_clearance_weight']*means['geometry_clearance']
    weights=(model.hand_future.mlp[-1].weight,model.scene_future.mlp[-1].weight)
    base_grad=torch.autograd.grad(base,weights,retain_graph=True)
    old_contact_grad=torch.autograd.grad(base_terms['contact']+base_terms['part'],weights,retain_graph=True,allow_unused=True)
    new_grad=torch.autograd.grad(geometry,weights+(prediction['trajectory'],prediction['scene']))
    norm=lambda values:float(torch.stack([v.float().square().sum() for v in values]).sum().sqrt())
    base_norm,geometry_norm=norm(base_grad),norm(new_grad[:2])
    recommended=min(.05,max(.0001,.5*base_norm/max(geometry_norm,1e-12)))
    coverage=diagnostics['geometry/positive_coverage'];coverage=coverage['sum']/max(coverage['count'],1)
    report=dict(status='PASS',scope='four real TRAIN contact examples, actual current checkpoint, GPU autograd; no optimizer updates',
        initialization=binding,indices=indices,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,device=a.device,
        base_loss=float(base),geometry_contact=float(means['geometry_contact']),geometry_clearance=float(means['geometry_clearance']),
        base_readout_gradient_norm=base_norm,geometry_readout_gradient_norm=geometry_norm,
        old_contact_readout_gradients=[None if g is None else float(g.norm()) for g in old_contact_grad],
        new_geometry_gradient_norms=dict(zip(('hand_readout','object_readout','decoded_hand','decoded_object'),[float(g.norm()) for g in new_grad])),
        positive_coverage=coverage,geometry_diagnostics=diagnostics,recommended_geometry_weight=recommended,
        recommended_to_base_readout_gradient_ratio=recommended*geometry_norm/max(base_norm,1e-12),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_s=time.monotonic()-started)
    if (coverage<.8 or any(not torch.isfinite(g).all() or g.norm()==0 for g in new_grad)
            or any(g is not None for g in old_contact_grad)):
        report['status']='FAIL'
    a.output.mkdir(exist_ok=False);(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report),flush=True)
    if report['status']!='PASS':raise ValueError('geometry coverage or direct gradient preflight failed')


if __name__=='__main__':main()
