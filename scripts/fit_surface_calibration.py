"""Four fixed frozen-feature residual heads, identical data and actual updates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.surface_calibration import ARMS,STEPS,HEAD_SEED,SCHEDULE_SEED,PersistenceResidualHead,encoder_states,load_encoder,encode_frozen,gates_and_label
from src.task.CmResidual.surface_motion_prior import metrics
from scripts.run_contact_response_probe import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--prior-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert torch.cuda.is_available() and not a.output.exists() and ROOT in a.output.resolve().parents
    torch.set_num_threads(2);torch.manual_seed(HEAD_SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0');old=a.execution_source/'qualified';data={};labels={};parents={};rows={}
    with np.load(a.data/'train_features.npz') as f:data['train']=torch.from_numpy(f['features']).to(device);labels['train']=torch.from_numpy(f['target']).to(device)
    with np.load(old/'features.npz') as f:data['held']=torch.from_numpy(f['action_velocity']).to(device);labels['held']=torch.from_numpy(f['target']).to(device)
    for group in ('train','held'):
        with np.load(old/(group+'_rows.npz')) as f:rows[group]={k:f[k] for k in f.files}
        parents[group]=[str(env) for env in rows[group]['env']]
    assert not set(parents['train'])&set(parents['held'])
    persistence={k:v[...,19:22].clone() for k,v in data.items()}
    template={k:v.detach().cpu().clone() for k,v in PersistenceResidualHead().state_dict().items()}
    schedule=torch.randint(6144,(STEPS,32),generator=torch.Generator().manual_seed(SCHEDULE_SEED))
    states=encoder_states(a.prior_source);a.output.mkdir();predictions={};reports={};training={};diagnostics={}
    for arm in ARMS:
        encoder=load_encoder(states[arm],device)
        encoded={group:torch.cat([encode_frozen(encoder,value[i:i+32],arm=='hand_flow_removed') for i in range(0,len(value),32)]) for group,value in data.items()}
        head=PersistenceResidualHead().to(device);head.load_state_dict(template)
        optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
        losses=[];early=[]
        for step in range(STEPS):
            ids=schedule[step].to(device);optimizer.zero_grad(set_to_none=True)
            prediction=head(encoded['train'][ids],persistence['train'][ids])
            loss=(prediction-labels['train'][ids]).square().mean();assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(head.parameters(),10);assert torch.isfinite(grad)
            optimizer.step();losses.append(float(loss.detach()))
            if step<3:early.append({k:v.detach().cpu().clone() for k,v in head.state_dict().items()})
            if (step+1)%400==0:print(json.dumps(dict(arm=arm,step=step+1,loss=losses[-1])),flush=True)
        assert all(parameter.grad is None for parameter in encoder.parameters())
        assert all(torch.equal(v.detach().cpu(),states[arm][k]) for k,v in encoder.state_dict().items())
        change=max(float((v.detach().cpu()-template[k]).abs().max()) for k,v in head.state_dict().items());assert change>0
        with torch.no_grad():
            predictions[arm]=torch.cat([head(encoded['held'][i:i+32],persistence['held'][i:i+32]).cpu() for i in range(0,6144,32)]).numpy()
        target=labels['held'].cpu().numpy();reports[arm]=metrics(predictions[arm],target,parents['held'])
        checkpoints=dict(state={k:v.detach().cpu() for k,v in head.state_dict().items()},encoder_state=states[arm],
                         common_initial=template,optimizer=optimizer.state_dict(),batch_schedule=schedule,
                         losses=losses,early_head_states=early,steps=STEPS,maximum_parameter_change=change,
                         encoder_frozen=True,hand_flow_removed=arm=='hand_flow_removed')
        torch.save(checkpoints,a.output/(arm+'.pt'))
        training[arm]=dict(steps=STEPS,trainable_parameters=sum(p.numel() for p in head.parameters()),
                           frozen_parameters=sum(p.numel() for p in encoder.parameters()),window_draws=STEPS*32,
                           maximum_parameter_change=change,encoder_unchanged=True)
        print(json.dumps(dict(arm=arm,complete=True,parent_epe_mm=reports[arm]['parent_epe_mm'])),flush=True)
        del encoded,encoder,head,optimizer
    target=labels['held'].cpu().numpy();persist=persistence['held'].cpu().numpy()
    baselines=dict(persistence=metrics(persist,target,parents['held']),zero=metrics(np.zeros_like(target),target,parents['held']))
    masks={**{'motion_'+str(i):rows['held']['motion']==i for i in range(3)},
           **{'arm_'+str(i):rows['held']['arm']==i for i in range(4)},
           'near':data['held'][...,15].min(-1).values.cpu().numpy()*.05<.02}
    masks['far']=~masks['near']
    for name,mask in masks.items():
        subset=[parents['held'][i] for i in np.flatnonzero(mask)]
        diagnostics[name]=dict(windows=int(mask.sum()),episodes=len(set(subset)),
                              arms={arm:metrics(value[mask],target[mask],subset)['parent_epe_mm'] for arm,value in predictions.items()},
                              persistence_mm=metrics(persist[mask],target[mask],subset)['parent_epe_mm'])
    gates,label=gates_and_label(reports,baselines['persistence']['parent_epe_mm'])
    result=dict(run_status='COMPLETED',label=label,gates=gates,reports=reports,baselines=baselines,training=training,
                diagnostics=diagnostics,actual_optimizer_updates=4*STEPS,train_episodes=384,held_episodes=384,
                train_windows=6144,held_windows=6144,checkpoint_selection=False,new_native_ticks=0,
                reused_same_seed_held_episodes=True,policy_utility_unmeasured=True,
                data_sha256=sha(a.data/'train_features.npz'),held_sha256=sha(old/'features.npz'))
    np.savez(a.output/'predictions.npz',**predictions)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=label,gates=gates)),flush=True)


if __name__=='__main__':main()
