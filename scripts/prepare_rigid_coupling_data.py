"""Build missing train transport/oracles and causal inputs for both splits."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.rigid_transport_capacity import transport_fields,optimal_segments
from src.task.CmResidual.rigid_coupling import causal_inputs
from src.task.CmResidual.surface_execution import predict_execution
from scripts.run_contact_response_probe import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--capacity-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert torch.cuda.is_available() and ROOT in a.output.resolve().parents and not a.output.exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;device=torch.device('cuda:0');old=a.execution_source/'qualified'
    with np.load(old/'geometry.npz') as f:local=f['object_local']
    ancestor=json.loads((a.execution_source/'run_manifest.json').read_text());initial=torch.load(Path(ancestor['source_native'])/'s655/initial.pt',map_location='cpu',weights_only=False)
    fit=json.loads((old/'execution_fit.json').read_text());fit['scale']=np.asarray(fit['scale']);fit['coefficients']={k:np.asarray(v) for k,v in fit['coefficients'].items()}
    scale=(initial['native_upper']-initial['native_lower']).numpy().astype(np.float64);scale[:3]=.1;scale[3:6]=np.pi;assert (scale>0).all()
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    rows={};inputs={};train_banks={k:[] for k in ('anchor','target','endpoints','coefficients','oracle_errors_m')};held_max=0.
    inherited=np.load(a.capacity_source/'capacity/fields.npz');stored_held={k:inherited[k] for k in ('anchor','target','causal')}
    for group in ('train','held'):
        with np.load(old/(group+'_rows.npz')) as f:rows[group]={k:f[k] for k in f.files}
        predictions=predict_execution(rows[group],fit,initial['native_lower'].numpy(),initial['native_upper'].numpy())['action_velocity']
        parts=[]
        for start in range(0,6144,128):
            stop=min(start+128,6144);subset={k:v[start:stop] for k,v in rows[group].items()}
            fields=transport_fields(subset,predictions[start:stop],local,urdf,device)
            parts.append(causal_inputs(subset,fields,local,scale,device).cpu().numpy())
            if group=='train':
                oracle=optimal_segments(fields['anchor'],fields['candidates']['causal'],fields['target'])
                for key,value in dict(anchor=fields['anchor'],target=fields['target'],endpoints=fields['candidates']['causal'],coefficients=oracle['coefficients'],oracle_errors_m=oracle['segment_errors_m']).items():train_banks[key].append(value.cpu().numpy())
            else:
                for name,key in (('anchor','anchor'),('target','target'),('causal','causal')):
                    value=fields['candidates']['causal'] if key=='causal' else fields[key]
                    error=float(np.abs(value.cpu().numpy()-stored_held[name][start:stop]).max());held_max=max(held_max,error);assert error<1e-12
            if (start//128+1)%16==0:print(json.dumps(dict(group=group,windows=stop,total=6144)),flush=True)
        inputs[group]=np.concatenate(parts)
    assert not set(rows['train']['env'])&set(rows['held']['env'])
    train={k:np.concatenate(v) for k,v in train_banks.items()}
    permutation=np.random.default_rng(4203).permutation(6144);shuffled_coefficients=[];shuffled_errors=[]
    for start in range(0,6144,128):
        stop=min(start+128,6144)
        result=optimal_segments(torch.from_numpy(train['anchor'][start:stop]).to(device),torch.from_numpy(train['endpoints'][start:stop]).to(device),torch.from_numpy(train['target'][permutation[start:stop]]).to(device))
        shuffled_coefficients.append(result['coefficients'].cpu().numpy());shuffled_errors.append(result['segment_errors_m'].cpu().numpy())
    train.update(shuffled_coefficients=np.concatenate(shuffled_coefficients),shuffled_errors_m=np.concatenate(shuffled_errors),permutation=permutation)
    mean=inputs['train'].astype(np.float64).mean((0,1));std=np.maximum(inputs['train'].astype(np.float64).std((0,1)),.01)
    a.output.mkdir();np.savez(a.output/'train_fields.npz',**train);np.savez(a.output/'inputs.npz',**inputs,mean=mean,std=std,joint_scale=scale)
    record=dict(run_status='COMPLETED',train_episodes=384,held_episodes=384,train_windows=6144,held_windows=6144,
                input_size=120,held_field_discrepancy_max_m=held_max,new_oracle_segments=6144*15*2,
                train_fields_sha256=sha(a.output/'train_fields.npz'),inputs_sha256=sha(a.output/'inputs.npz'),normalization_from_train_inputs_only=True)
    (a.output/'results.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
