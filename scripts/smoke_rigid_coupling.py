"""Four-row causal inference and three-step independent-gradient CPU smoke."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.rigid_transport_capacity import transport_fields,optimal_segments
from src.task.CmResidual.rigid_coupling import causal_inputs,CouplingPredictor,coupling_loss,deploy
from scripts.audit_rigid_coupling import independent_inputs,replay


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);old=a.execution_source/'qualified'
    with np.load(old/'held_rows.npz') as f:rows={k:f[k][[0,40,100,180]] for k in f.files}
    with np.load(old/'geometry.npz') as f:local=f['object_local']
    with np.load(old/'joint_predictions.npz') as f:predicted=f['action_velocity'][[0,40,100,180]]
    ancestor=json.loads((a.execution_source/'run_manifest.json').read_text());initial=torch.load(Path(ancestor['source_native'])/'s655/initial.pt',map_location='cpu',weights_only=False)
    scale=(initial['native_upper']-initial['native_lower']).numpy().astype(np.float64);scale[:3]=.1;scale[3:6]=np.pi
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    fields=transport_fields(rows,predicted,local,urdf,'cpu');features=causal_inputs(rows,fields,local,scale,'cpu')
    changed={k:v.copy() for k,v in rows.items()};changed['next_q']+=.1;changed['next_obj'][:,:3]+=.1
    altered=transport_fields(changed,predicted,local,urdf,'cpu');new_features=causal_inputs(changed,altered,local,scale,'cpu')
    assert torch.equal(features,new_features)
    rebuilt=independent_inputs(rows,fields['anchor'].numpy(),fields['candidates']['causal'].numpy(),local,scale,urdf)
    discrepancy=float(np.abs(rebuilt-features.numpy()).max());assert np.allclose(rebuilt,features.numpy(),atol=2e-5,rtol=2e-6)
    labels=optimal_segments(fields['anchor'],fields['candidates']['causal'],fields['target'])['coefficients'].float()
    anchor=fields['anchor'].float();endpoint=fields['candidates']['causal'].float();target=fields['target'].float()
    mean=features.double().mean((0,1)).float();std=features.double().std((0,1),unbiased=False).clamp_min(.01).float();x=(features-mean)/std
    torch.manual_seed(4201);model=CouplingPredictor();initial_state={k:v.detach().clone() for k,v in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    losses=[];early=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);c,s=model(x);loss=coupling_loss(c,s,labels,anchor,endpoint,target,torch.ones(15));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
        losses.append(float(loss.detach()));early.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    cp=dict(common_initial=initial_state,arm='full',component_weights=torch.ones(15),batch_schedule=torch.tensor([[0,1,2,3]]*3),input_mean=mean,input_std=std,early_states=early,losses=losses)
    bank=dict(endpoints=endpoint.numpy(),anchor=anchor.numpy(),target=target.numpy(),coefficients=labels.numpy())
    parameter_max,loss_max=replay(cp,features.numpy(),bank)
    result=deploy(model,x,fields['anchor'],fields['candidates']['causal'])
    second=deploy(model,(new_features-mean)/std,altered['anchor'],altered['candidates']['causal'])
    assert all(torch.equal(v,second[k]) for k,v in result.items())
    print(json.dumps(dict(engineering_smoke='PASS',cpu_reason='4rows/3updates avoids GPU startup',independent_input_max=discrepancy,
                         future_label_deployment_isolation=True,independent_numpy_adamw_parameter_max=parameter_max,independent_numpy_loss_max=loss_max)),flush=True)


if __name__=='__main__':main()
