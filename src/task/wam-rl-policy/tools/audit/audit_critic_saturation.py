"""Read-only checkpoint/online-curve diagnostics; no training or scientific verdict."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from wam_rl_policy.bc_mlp import CurrentMLP
from wam_rl_policy.ppo import Actor, Critic, observation, append_episode_clock


@torch.no_grad()
def inspect(folder, device):
    packet=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    config=packet['config'];dim=config.get('observation_dim',327)
    model=Critic(dim,config.get('network_kind','small'),config.get('critic_activation','tanh')).to(device)
    model.load_state_dict(packet['critic'],strict=True)
    if config.get('observation_kind','native').startswith('wam-'):
        source=folder/'first-wam-input.pt'
        x=torch.load(source,map_location='cpu',weights_only=False)['observation'].to(device)
        input_scope='actual first saved WAM input, potentially stale; interpret with online curve'
    else:
        source=folder/'last-episode.npz'
        with np.load(source,allow_pickle=False) as data:
            controls=data['reward'].shape[0];ids=np.linspace(0,controls-1,16,dtype=int)
            get=lambda key:torch.as_tensor(data[key][ids].copy(),device=device)
            raw=get('observation');relative=get('goal')-get('object')[...,:3];base=get('base')
            kind=config.get('observation_kind','native');features=None
            if kind!='native':
                bc=CurrentMLP().to(device).eval().requires_grad_(False)
                bc.load_state_dict(torch.load(config['base_checkpoint'],map_location='cpu',weights_only=False)['model'])
                normalized=(raw.clamp(-5,5)-bc.x_mean)/bc.x_scale
                if kind=='bc-hidden':features=bc.net[:-1](normalized)
                elif kind=='normalized':features=torch.nn.functional.pad(normalized,(0,212))
                else:raise ValueError('Unknown saved current representation')
            x=observation(raw.flatten(0,1),relative.flatten(0,1),base.flatten(0,1),
                          features.flatten(0,1) if features is not None else None).reshape(len(ids),len(base[0]),-1)
            x=torch.stack([append_episode_clock(row,int(t)+1,controls,config.get('clock_kind','none')) for row,t in zip(x,ids)]).flatten(0,1)
        input_scope='16 fixed times from last complete stored episode; model may be newer than that trace'
    assert x.shape[-1]==dim and torch.isfinite(x).all()
    def layers(net):
        h=x;result=[]
        for layer in net:
            h=layer(h)
            if isinstance(layer,nn.Tanh):
                result.append(dict(saturated_fraction=float((h.abs()>.99).float().mean()),
                    near_constant_units_fraction=float((h.std(0,unbiased=False)<1e-5).float().mean())))
        return result
    saturation=layers(model.net)
    actor=Actor(obs_dim=dim,network_kind=config.get('network_kind','small'),
                action_kind=config.get('action_kind','residual')).to(device)
    actor.load_state_dict(packet['actor'],strict=True)
    value=model(x)
    curve=[row for row in json.loads((folder/'curve.json').read_text()) if row['update']<=packet['update']]
    def all_zero(row):
        values=[s.get('value_explained_variance') for s in row.get('ppo_segments',[row['ppo']])]
        return len(values)==13 and all(v is not None and abs(v)<1e-4 for v in values)
    streak=0
    for row in reversed(curve):
        if not all_zero(row):break
        streak+=1
    return dict(run=str(folder),checkpoint_update=packet['update'],input=str(source),input_scope=input_scope,
        samples=len(x),dim=dim,value_mean=float(value.mean()),value_std=float(value.std(unbiased=False)),
        value_min=float(value.min()),value_max=float(value.max()),tanh_layers=saturation,
        actor_tanh_layers=layers(actor.net),actor_log_std=actor.log_std.tolist(),
        actor_learned_mean_correction_std=float(actor.net(x).std(0,unbiased=False).mean()),
        trailing_all_13_segments_abs_ev_below_1e_minus4=streak,
        last_segments_ev=[s.get('value_explained_variance') for s in curve[-1].get('ppo_segments',[curve[-1]['ppo']])],
        interpretation='Stale-state saturation alone is insufficient; compare sustained online EV. No policy efficacy verdict.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh audit artifact required')
    torch.set_num_threads(2)
    results=[inspect(folder,torch.device(a.device)) for folder in a.runs]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(audits=results),indent=2,allow_nan=False)+'\n')
    for row in results:print(json.dumps({k:row[k] for k in ['run','checkpoint_update','value_std','trailing_all_13_segments_abs_ev_below_1e_minus4','tanh_layers']}))


if __name__=='__main__':main()
