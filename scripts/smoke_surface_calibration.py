"""Two-row, three-step CPU engineering check; no research performance claim."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.surface_calibration import PersistenceResidualHead,encoder_states,load_encoder,encode_frozen,causal_features
from scripts.audit_surface_calibration import replay_early


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--native-source',type=Path,required=True);p.add_argument('--prior-source',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);old=a.execution_source/'qualified'
    with np.load(old/'held_rows.npz') as f:rows={k:f[k][:2] for k in f.files}
    with np.load(old/'geometry.npz') as f:geometry={k:f[k] for k in f.files}
    fitted=json.loads((old/'execution_fit.json').read_text());fitted['scale']=np.asarray(fitted['scale']);fitted['coefficients']={k:np.asarray(v) for k,v in fitted['coefficients'].items()}
    initial=torch.load(a.native_source/'s655/initial.pt',map_location='cpu',weights_only=False)
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    args=(geometry,fitted,initial['native_lower'].numpy(),initial['native_upper'].numpy(),urdf,'cpu')
    rebuilt=causal_features(rows,*args)
    changed={k:v.copy() for k,v in rows.items()};changed['next_q']+=.3;changed['next_obj'][:,:3]+=.1
    perturbed=causal_features(changed,*args)
    assert np.array_equal(rebuilt['features'],perturbed['features']) and not np.array_equal(rebuilt['target'],perturbed['target'])
    with np.load(old/'features.npz') as f:
        maximum=float(np.abs(rebuilt['features']-f['action_velocity'][:2]).max());assert maximum<2e-5
        assert np.allclose(rebuilt['target'],f['target'][:2],atol=2e-5,rtol=2e-6)
    state=encoder_states(a.prior_source)['pretrained'];encoder=load_encoder(state,'cpu')
    x=torch.from_numpy(rebuilt['features']);y=torch.from_numpy(rebuilt['target']);anchor=x[...,19:22]
    z=encode_frozen(encoder,x);head=PersistenceResidualHead();assert torch.equal(head(z,anchor),anchor)
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=1e-4)
    initial_head={k:v.detach().clone() for k,v in head.state_dict().items()};early=[];losses=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True);loss=(head(z,anchor)-y).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),10);optimizer.step()
        early.append({k:v.detach().clone() for k,v in head.state_dict().items()});losses.append(float(loss.detach()))
    assert all(p.grad is None for p in encoder.parameters()) and all(torch.equal(v,state[k]) for k,v in encoder.state_dict().items())
    assert head.layers[-1].weight.abs().max()>0
    replay_max,replay_loss=replay_early(dict(common_initial=initial_head,encoder_state=state,
        batch_schedule=torch.tensor([[0,1]]*3),hand_flow_removed=False,early_head_states=early,losses=losses),rebuilt['features'],rebuilt['target'])
    print(json.dumps(dict(engineering_smoke='PASS',cpu_reason='2rows/3updates lower cost than GPU startup',
                         feature_max_vs_inherited=maximum,future_label_isolation=True,zero_head_equals_persistence=True,
                         encoder_frozen=True,numpy_adamw_parameter_max=replay_max,numpy_adamw_loss_max=replay_loss)),flush=True)


if __name__=='__main__':main()
