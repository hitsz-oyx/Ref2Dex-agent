#!/usr/bin/env python3
"""Exactly one shared-data return-gradient update per newly audited panel."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
import torch
from src.task.CmResidual.support_feature_policy import actor,offpolicy_score_loss,SCHEMA
from src.task.CmResidual.paired_evaluation import fingerprint

VARIANTS=('cm','state_only','global_motion_arm')


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return type(value)((k,cpu(v)) for k,v in value.items())
    if isinstance(value,list):
        return [cpu(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--previous',type=Path)
    p.add_argument('--panel',type=Path)
    args=p.parse_args()
    if args.output.exists() or ROOT not in args.output.resolve().parents:
        raise ValueError('unique isolated update')
    if (args.previous is None) != (args.panel is None):
        raise ValueError('initialize or exactly one audited update')
    begin=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    models={v:actor() for v in VARIANTS}
    optimizers={v:torch.optim.Adam(n.parameters(),lr=.01,foreach=False) for v,n in models.items()}
    if args.previous is None:
        initial={v:fingerprint(n.state_dict()) for v,n in models.items()}
        if len(set(initial.values()))!=1:
            raise ValueError('matched initial weights')
        payload=dict(schema=SCHEMA,seed=752,updates=0,initial_fingerprints=initial,
                     variants={v:dict(model=cpu(n.state_dict()),optimizer=cpu(optimizers[v].state_dict())) for v,n in models.items()},
                     source_actor_weights_used=False,device_reason='CPU tiny weight/optimizer initialization; no training or repeated inference')
    else:
        if not torch.cuda.is_available():
            raise ValueError('GPU training required')
        previous=torch.load(args.previous,map_location='cpu',weights_only=False)
        data=torch.load(args.panel/'audited.pt',map_location='cpu',weights_only=False)
        audit=json.loads((args.panel/'panel_audit.json').read_text())
        if audit['run_status']!='COMPLETED' or sha(args.panel/'audited.pt')!=audit['dataset_sha256']:
            raise ValueError('independent panel audit')
        seed=int(data['seed'][0]);update=previous['updates']+1
        if previous['schema']!=SCHEMA or previous['seed']!=752 or seed!=528+update or update>12 or len(data['state'])!=768 or not (data['seed']==seed).all():
            raise ValueError('frozen update/cohort order')
        device=torch.device('cuda:0')
        from src.task.CmResidual.support_feature_bank import PhysicalFeatureBank
        bank=PhysicalFeatureBank(args.source,device)
        state=data['state'].to(device);motion=data['motion'].to(device)
        inputs,physical_probability=bank.features(state,motion)
        reward=data['physical105'].to(device,dtype=torch.float32)
        action=data['arm'].to(device);behavior=torch.full((768,),.125,device=device)
        baseline=bank.reward_baseline[motion]
        records={};variants={}
        for variant in VARIANTS:
            network=models[variant].to(device)
            network.load_state_dict(previous['variants'][variant]['model'])
            optimizer=torch.optim.Adam(network.parameters(),lr=.01,foreach=False)
            optimizer.load_state_dict(previous['variants'][variant]['optimizer'])
            before=cpu(network.state_dict());optimizer_before=cpu(optimizer.state_dict())
            logits=network(inputs[variant]);probability=logits.softmax(-1)
            loss=offpolicy_score_loss(logits,action,reward,behavior,baseline)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            gradient={k:cpu(p.grad) for k,p in network.named_parameters()}
            norm=torch.nn.utils.clip_grad_norm_(network.parameters(),10)
            if not torch.isfinite(norm):raise ValueError('finite gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in network.parameters()):raise ValueError('finite updated actor')
            variants[variant]=dict(model=cpu(network.state_dict()),optimizer=cpu(optimizer.state_dict()))
            records[variant]=dict(before=before,optimizer_before=optimizer_before,
                input=cpu(inputs[variant]),physical_probability=cpu(physical_probability[variant]),
                logits=cpu(logits),probability=cpu(probability),loss=float(loss),
                gradient_before_clipping=gradient,gradient_norm=float(norm),
                after_fingerprint=fingerprint(network.state_dict()))
        bank.assert_frozen()
        payload=dict(schema=SCHEMA,seed=752,updates=update,update_seed=seed,
            initial_fingerprints=previous['initial_fingerprints'],variants=variants,
            previous_sha256=sha(args.previous),audited_panel_sha256=sha(args.panel/'audited.pt'),
            frozen_physical_input_sha256=bank.hashes,source_actor_weights_used=False)
        packet=dict(records=records,reward=cpu(reward),action=cpu(action),motion=cpu(motion),
            behavior_probability=cpu(behavior),state_baseline=cpu(baseline),raw_state=cpu(state),
            source_dataset_sha256=sha(args.panel/'audited.pt'),optimizer_hyperparameters=dict(lr=.01,betas=[.9,.999],eps=1e-8,foreach=False),
            source=args.source.resolve(),seed=seed,update=update)
    args.output.mkdir()
    torch.save(payload,args.output/'policy_heads.pt')
    if args.previous is not None:torch.save(packet,args.output/'update_packet.pt')
    result=dict(run_status='COMPLETED',updates=payload['updates'],seed=752,
        checkpoint_sha256=sha(args.output/'policy_heads.pt'),no_model_reward=True,
        one_fresh_full_batch_update=args.previous is not None,wall_seconds=time.monotonic()-begin)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
