#!/usr/bin/env python3
"""Decision-only factual H2 diagnosis after a terminal failed utility gate."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    start=time.monotonic();m=json.loads((args.run/'run_manifest.json').read_text());utility=json.loads(args.utility.read_text())
    if m['run_status']!='COMPLETED' or utility['label']!='UNPROMISING' or args.output.exists():raise ValueError('terminal negative original gate/unique diagnosis required')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):raise ValueError('source/input drift')
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from src.task.CmResidual.catalog_consequence_selector import FrozenCatalogSelector
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=ROOT/'src/task/CmResidual/research/contact_consequence/output/P-20261002-catalog-control-fit-r1/catalog_controls.pt';selector=FrozenCatalogSelector(checkpoint,'cuda');norm=selector.normalization
    errors={mode:dict(dz=[],clear=[],contact=[]) for mode in ['catalog_cm','catalog_shuffled']};cv=[];persist=[];ood=[];program=[]
    for phase in m['phases']:
        path=Path(phase['directory'])/'records.pt'
        if sha(path)!=phase['result']['record_sha256']:raise ValueError('record drift')
        b=torch.load(path,map_location='cpu',weights_only=False);n=len(b['assignment'])
        history=b['decision_history'].reshape(-1,10,69);native=b['native_observation'][:,::2].reshape(-1,b['native_observation'].shape[-1]);chosen=b['program'][:,::2].reshape(-1)
        pre=torch.cat((b['state'][:,None],b['future_state'][:,:-1]),1)[:,::2];initial=torch.cat((b['initial_clearance'][:,None],b['future_clearance'][:,:-1]),1)[:,::2]
        dz=(b['future_state'].reshape(n,5,2,49)[:,:,:,38]-pre[:,:,None,38]).reshape(-1,2)*1000
        dc=(b['future_clearance'].reshape(n,5,2)-initial[:,:,None]).reshape(-1,2)*1000
        joint=b['future_contact'].reshape(n,5,2,2).all(-1).float().reshape(-1,2)
        # Each actual two-step prefix obeys the selected feedback program before
        # replan; the full H10 controller future does not obey one fixed program.
        cv.append(pre[:,:,45].reshape(-1,1)*torch.tensor([[1/30,2/30]])*1000-dz)
        persist.append(history[:,-1,49:51].bool().all(-1)[:,None].float().expand(-1,2)-joint);ood.append(b['ood'][:,:,0].reshape(-1));program.append(chosen)
        for offset in range(0,len(history),96):
            h=((history[offset:offset+96].cuda()-norm['history_mean'])/norm['history_std']).clamp(-8,8);o=((native[offset:offset+96].cuda()-norm['native_mean'])/norm['native_std']).clamp(-8,8)
            for mode in errors:
                with torch.no_grad():pred=torch.stack([model(h,o) for model in selector.models[mode]]).mean(0).cpu()
                row=torch.arange(len(pred));selected=pred[row,chosen[offset:offset+96]]
                errors[mode]['dz'].append(selected[:,:2]*10-dz[offset:offset+96]);errors[mode]['clear'].append(selected[:,10:12]*10-dc[offset:offset+96]);errors[mode]['contact'].append(torch.sigmoid(selected[:,20:22])-joint[offset:offset+96])
    ood=torch.cat(ood);program=torch.cat(program);cv=torch.cat(cv);persist=torch.cat(persist);errors={mode:{k:torch.cat(v) for k,v in e.items()} for mode,e in errors.items()}
    result={}
    for name,mask in [('all',torch.ones_like(ood)),('in_fit_domain',~ood),('out_of_domain',ood)]:
        if not mask.any():continue
        result[name]=dict(two_step_prefixes=int(mask.sum()),actual_program_counts=torch.bincount(program[mask],minlength=8).tolist(),constant_velocity_height_rmse_mm=float(cv[mask].square().mean().sqrt()),persist_joint_brier=float(persist[mask].square().mean()),models={mode:dict(height_rmse_mm=float(e['dz'][mask].square().mean().sqrt()),clearance_mae_mm=float(e['clear'][mask].abs().mean()),joint_brier=float(e['contact'][mask].square().mean())) for mode,e in errors.items()})
    elapsed=time.monotonic()-start;out=dict(run_status='COMPLETED',diagnosis=result,checkpoint_sha256=sha(checkpoint),gpu=admission,elapsed_seconds=elapsed,original_gate='UNPROMISING_PRESERVED',cumulative_slot2_seconds=utility['cumulative_seconds_including_fit_engineering_audits_analysis']+elapsed,scope='posthoc decision diagnosis of factual first2 prefixes only; no full10 forecast validation under replanning, no counterfactual oracle, no threshold/model adjustment or gate rescue')
    if out['cumulative_slot2_seconds']>3600:raise ValueError('same slot2 budget exceeded')
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--utility',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
