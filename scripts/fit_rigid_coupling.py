"""Three matched causal coupling fits; held inference uses no outcome labels."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.rigid_coupling import ARMS,STEPS,STATE_IDS,CouplingPredictor,component_weights,coupling_loss,deploy,classify
from src.task.CmResidual.rigid_transport_capacity import episode_report


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--capacity-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert torch.cuda.is_available() and ROOT in a.output.resolve().parents and not a.output.exists()
    torch.set_num_threads(2);torch.manual_seed(4201);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device=torch.device('cuda:0')
    raw=np.load(a.data/'inputs.npz');mean=torch.from_numpy(raw['mean'].astype(np.float32)).to(device);std=torch.from_numpy(raw['std'].astype(np.float32)).to(device)
    inputs={g:(torch.from_numpy(raw[g]).to(device)-mean)/std for g in ('train','held')}
    fields=np.load(a.data/'train_fields.npz');train={k:torch.from_numpy(fields[k].astype(np.float32)).to(device) for k in ('anchor','target','endpoints','coefficients','shuffled_coefficients')}
    permutation=fields['permutation'];held_fields=np.load(a.capacity_source/'capacity/fields.npz')
    held_anchor=torch.from_numpy(held_fields['anchor']).to(device);held_endpoints=torch.from_numpy(held_fields['causal']).to(device)
    # Outcomes are outside deploy() and are read only for subsequent reports.
    held_target=held_fields['target'];old=a.execution_source/'qualified'
    with np.load(old/'held_rows.npz') as f:held_rows={k:f[k] for k in f.files}
    schedule=torch.randint(6144,(STEPS,32),generator=torch.Generator().manual_seed(4202))
    template={k:v.detach().cpu().clone() for k,v in CouplingPredictor().state_dict().items()};a.output.mkdir();predictions={};outputs={};reports={};training={}
    for arm in ARMS:
        state_ids=torch.from_numpy(STATE_IDS).to(device)
        xtrain=inputs['train'][:,state_ids] if arm=='state_only' else inputs['train']
        xheld=inputs['held'][:,state_ids] if arm=='state_only' else inputs['held']
        endpoints=train['endpoints'][:,state_ids] if arm=='state_only' else train['endpoints']
        eval_endpoints=held_endpoints[:,state_ids] if arm=='state_only' else held_endpoints
        coefficient_label=train['shuffled_coefficients'] if arm=='shuffled' else (train['coefficients'][:,state_ids] if arm=='state_only' else train['coefficients'])
        target=train['target'][torch.from_numpy(permutation).to(device)] if arm=='shuffled' else train['target']
        weights=component_weights(arm,device);model=CouplingPredictor().to(device);model.load_state_dict(template)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);losses=[];early=[]
        for step in range(STEPS):
            ids=schedule[step].to(device);optimizer.zero_grad(set_to_none=True);coefficient,score=model(xtrain[ids])
            loss=coupling_loss(coefficient,score,coefficient_label[ids],train['anchor'][ids],endpoints[ids],target[ids],weights)
            assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(gradient);optimizer.step();losses.append(float(loss.detach()))
            if step<3:early.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            if (step+1)%500==0:print(json.dumps(dict(arm=arm,step=step+1,loss=losses[-1])),flush=True)
        chunks={k:[] for k in ('coefficients','scores','winner','prediction')};model.eval()
        for start in range(0,6144,128):
            result=deploy(model,xheld[start:start+128],held_anchor[start:start+128],eval_endpoints[start:start+128])
            for k,v in result.items():chunks[k].append(v.cpu().numpy())
        output={k:np.concatenate(v) for k,v in chunks.items()};outputs[arm]=output;predictions[arm]=output['prediction']
        reports[arm]=episode_report(output['prediction'],held_target,held_rows['env'])
        changed=max(float((v.detach().cpu()-template[k]).abs().max()) for k,v in model.state_dict().items());assert changed>0
        checkpoint=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},common_initial=template,
                        optimizer=optimizer.state_dict(),batch_schedule=schedule,losses=losses,early_states=early,
                        steps=STEPS,arm=arm,component_weights=weights.cpu(),maximum_parameter_change=changed,
                        input_mean=mean.cpu(),input_std=std.cpu(),label_permutation=torch.from_numpy(permutation) if arm=='shuffled' else None,
                        state_replica_ids=torch.from_numpy(STATE_IDS) if arm=='state_only' else None)
        torch.save(checkpoint,a.output/(arm+'.pt'));np.savez(a.output/(arm+'_outputs.npz'),**output)
        training[arm]=dict(steps=STEPS,parameters=sum(p.numel() for p in model.parameters()),window_draws=STEPS*32,component_forward_rows=STEPS*32*15,maximum_parameter_change=changed)
        print(json.dumps(dict(arm=arm,complete=True,episode_epe_mm=reports[arm]['episode_epe_mm'])),flush=True)
        del model,optimizer
    baselines={name:episode_report(value,held_target,held_rows['env']) for name,value in (('persistence',held_fields['anchor']),('zero',np.zeros_like(held_target)))}
    with np.load(old/'features.npz') as f:near=f['stationary'][...,15].min(-1)*.05<.02
    masks={**{'motion_'+str(i):held_rows['motion']==i for i in range(3)},**{'arm_'+str(i):held_rows['arm']==i for i in range(4)},'near':near,'far':~near};diagnostics={}
    for name,mask in masks.items():
        diagnostics[name]={arm:episode_report(value[mask],held_target[mask],held_rows['env'][mask]) for arm,value in predictions.items()}
        diagnostics[name]['persistence']=episode_report(held_fields['anchor'][mask],held_target[mask],held_rows['env'][mask])
    gates,label=classify(reports,baselines,diagnostics['near'])
    result=dict(run_status='COMPLETED',label=label,gates=gates,reports=reports,baselines=baselines,diagnostics=diagnostics,training=training,
                actual_optimizer_updates=STEPS*3,new_native_ticks=0,held_outcome_fitting=False,checkpoint_selection=False,
                shared_current_state_geometry_command_information=True,seen_same_seed_held_distribution=True,policy_utility_unmeasured=True)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=label,gates=gates)),flush=True)


if __name__=='__main__':main()
