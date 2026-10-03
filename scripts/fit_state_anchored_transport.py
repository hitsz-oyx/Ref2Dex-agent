"""Matched direct-flow transport fits, sharing an immutable learned state anchor."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.rigid_coupling import CouplingPredictor,STATE_IDS,deploy
from src.task.CmResidual.rigid_transport_capacity import episode_report
from src.task.CmResidual.state_anchored_transport import ARMS,STEPS,STATE,TransportHead,tokens,log_prior,predict,flow_loss,classify


def main():
    p=argparse.ArgumentParser()
    for name in ('coupling-source','execution-source','capacity-source','calibration-source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert torch.cuda.is_available() and ROOT in a.output.resolve().parents and not a.output.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;device='cuda:0'
    banks=np.load(a.coupling_source/'data/inputs.npz');train_fields=np.load(a.coupling_source/'data/train_fields.npz');held_fields=np.load(a.capacity_source/'capacity/fields.npz')
    mean=banks['mean'].astype(np.float32);std=banks['std'].astype(np.float32)
    checkpoint=torch.load(a.coupling_source/'fit/state_only.pt',map_location='cpu',weights_only=False)
    base_model=CouplingPredictor().to(device);base_model.load_state_dict(checkpoint['state']);base_model.eval()
    for parameter in base_model.parameters():parameter.requires_grad_(False)
    before={k:v.detach().cpu().clone() for k,v in base_model.state_dict().items()}
    calibration=json.loads((a.calibration_source/'run_manifest.json').read_text());assert calibration['run_status']=='COMPLETED'
    source_train=Path(calibration['source_run'])/'data/train_features.npz'
    with np.load(source_train) as f:train_gate=f['features'][...,15].min(-1)*.05<.02
    with np.load(a.execution_source/'qualified/features.npz') as f:held_gate=f['stationary'][...,15].min(-1)*.05<.02
    groups={};bases={};a.output.mkdir()
    base_mean=torch.from_numpy(mean).to(device);base_std=torch.from_numpy(std).to(device)
    for group,fields,gate in [('train',train_fields,train_gate),('held',held_fields,held_gate)]:
        endpoints=fields['endpoints'] if group=='train' else fields['causal']
        # Reproduce the frozen checkpoint's original GPU normalization arithmetic.
        normalized=(torch.from_numpy(banks[group]).to(device)-base_mean)/base_std
        state_inputs=normalized[:,torch.from_numpy(STATE_IDS).to(device)]
        parts={k:[] for k in ('coefficients','scores','winner','prediction')}
        with torch.no_grad():
            for start in range(0,6144,128):
                x=state_inputs[start:start+128]
                y=deploy(base_model,x,torch.from_numpy(fields['anchor'][start:start+128]).to(device),torch.from_numpy(endpoints[start:start+128,STATE_IDS]).to(device))
                for key,value in y.items():parts[key].append(value.cpu().numpy())
        bases[group]={k:np.concatenate(v) for k,v in parts.items()};np.savez(a.output/(group+'_base.npz'),**bases[group],gate=gate)
        groups[group]=dict(raw=banks[group],base=torch.from_numpy(bases[group]['prediction']).to(device),
                           endpoints=torch.from_numpy(endpoints).to(device),target=torch.from_numpy(fields['target']).to(device),gate=torch.from_numpy(gate).to(device))
    previous=np.load(a.coupling_source/'fit/state_only_outputs.npz')['prediction']
    assert np.max(np.abs(bases['held']['prediction']-previous))<1e-12
    torch.manual_seed(4301);template={k:v.detach().cpu().clone() for k,v in TransportHead().state_dict().items()}
    schedule=torch.randint(6144,(STEPS,32),generator=torch.Generator().manual_seed(4302));permutation=np.random.default_rng(4303).permutation(6144)
    reports={};predictions={};training={}
    for arm in ARMS:
        x={g:torch.from_numpy((tokens(groups[g]['raw'],arm)-mean)/std).to(device) for g in groups}
        ends={g:(groups[g]['endpoints'][:,STATE] if arm=='state_only' else groups[g]['endpoints'][:,2:]) for g in groups}
        prior=torch.from_numpy(log_prior(arm)).to(device)
        model=TransportHead().to(device);model.load_state_dict(template)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);losses=[];early=[]
        target=groups['train']['target'][torch.from_numpy(permutation).to(device)] if arm=='shuffled' else groups['train']['target']
        for step in range(STEPS):
            ids=schedule[step].to(device);optimizer.zero_grad(set_to_none=True)
            result=predict(model,x['train'][ids],groups['train']['base'][ids].float(),ends['train'][ids].float(),groups['train']['gate'][ids],prior)
            loss=flow_loss(result['prediction'],target[ids].float());assert torch.isfinite(loss)
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(gradient);optimizer.step()
            losses.append(float(loss.detach()))
            if step<3:early.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            if (step+1)%500==0:print(json.dumps(dict(arm=arm,step=step+1,loss=losses[-1])),flush=True)
        output={k:[] for k in ('logits','weights','prediction')};model.eval()
        with torch.no_grad():
            for start in range(0,6144,128):
                z=predict(model,x['held'][start:start+128],groups['held']['base'][start:start+128],ends['held'][start:start+128],groups['held']['gate'][start:start+128],prior)
                for key,value in z.items():output[key].append(value.cpu().numpy())
        output={k:np.concatenate(v) for k,v in output.items()};np.savez(a.output/(arm+'_outputs.npz'),**output)
        assert np.array_equal(output['prediction'][~held_gate],bases['held']['prediction'][~held_gate])
        with np.load(a.execution_source/'qualified/held_rows.npz') as f:env=f['env']
        predictions[arm]=output['prediction'];reports[arm]=episode_report(output['prediction'],held_fields['target'],env)
        changed=max(float((v.detach().cpu()-template[k]).abs().max()) for k,v in model.state_dict().items());assert changed>0
        ck=dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},common_initial=template,
                optimizer=optimizer.state_dict(),batch_schedule=schedule,losses=losses,early_states=early,
                steps=STEPS,arm=arm,log_prior=prior.cpu(),input_mean=torch.from_numpy(mean),input_std=torch.from_numpy(std),
                label_permutation=torch.from_numpy(permutation) if arm=='shuffled' else None,
                state_replica_ids=torch.from_numpy(STATE) if arm=='state_only' else None,
                frozen_state=before,maximum_parameter_change=changed)
        torch.save(ck,a.output/(arm+'.pt'));training[arm]=dict(steps=STEPS,parameters=sum(v.numel() for v in model.parameters()),window_draws=STEPS*32,head_token_rows=STEPS*32*14,maximum_parameter_change=changed)
        print(json.dumps(dict(arm=arm,complete=True,episode_epe_mm=reports[arm]['episode_epe_mm'])),flush=True)
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in base_model.state_dict().items())
    with np.load(a.execution_source/'qualified/held_rows.npz') as f:rows={k:f[k] for k in f.files}
    baselines={name:episode_report(value,held_fields['target'],env) for name,value in [('frozen_state',bases['held']['prediction']),('persistence',held_fields['anchor']),('zero',np.zeros_like(held_fields['target']))]}
    masks={**{'motion_'+str(i):rows['motion']==i for i in range(3)},**{'arm_'+str(i):rows['arm']==i for i in range(4)},'near':held_gate,'far':~held_gate};diagnostics={}
    for name,mask in masks.items():
        diagnostics[name]={arm:episode_report(value[mask],held_fields['target'][mask],env[mask]) for arm,value in {**predictions,'frozen_state':bases['held']['prediction'],'persistence':held_fields['anchor']}.items()}
    gates,label=classify(reports,baselines,diagnostics['near'])
    result=dict(run_status='COMPLETED',label=label,gates=gates,reports=reports,baselines=baselines,diagnostics=diagnostics,training=training,
                actual_optimizer_updates=4500,inherited_shared_state_updates=1500,new_native_ticks=0,
                train_near_windows=int(train_gate.sum()),held_near_windows=int(held_gate.sum()),frozen_state_unchanged=True,
                far_predictions_identical_to_frozen_state=True,held_outcome_fitting=False,checkpoint_selection=False,
                convex_mixture_expands_old_single_segment_family=True,previously_viewed_holdout=True,policy_utility_unmeasured=True)
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=label,gates=gates)),flush=True)


if __name__=='__main__':main()
