"""Fresh Gaussian complete actor-parameter gradients; no model fitting or update."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.func import functional_call,vmap,grad
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.option_model_policy import DYNAMIC,VARIANTS,initialized_network,scores,trace_extra,state_features,known_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();begin=time.monotonic();torch.set_num_threads(2);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=json.loads((root/'run_manifest.json').read_text());assert m['experiment_id']=='P-20261002-physical-gradient-control'
    d=root/'s618';assert json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED'
    initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);metadata=json.loads((d/'physical_metadata.json').read_text());rows=json.loads((d/'rows.json').read_text())
    cp=torch.load(m['source_models'],map_location='cpu',weights_only=False);bundle=torch.load(m['source_actors'],map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    assert cp['model_updates_each']==1500 and bundle['actor_updates_each']==1000
    env=np.arange(768);ticks=initial['decision_steps'].numpy();motion=initial['motion'].numpy();stop=initial['phase_stop'].numpy()[motion]
    compact=np.concatenate((trace['normalized_context'].numpy()[ticks,env],np.ones((768,1),np.float32),((stop+30-ticks)/np.float32(202))[:,None]),-1).astype(np.float32)
    extra=trace_extra(trace,ticks,env,metadata);current=state_features(compact,extra,cp['extra_mean'],cp['extra_std']);known=known_plan(initial,ticks+8,base)
    mask=initial['policy_group'].numpy()>0;assert mask.sum()==576
    reward=np.array([r['physical105'] for r in rows],np.float32)[mask];epsilon=initial['option_raw12'].numpy()[mask]
    gpu=lambda v:torch.from_numpy(v).cuda()
    x=gpu(current[mask]);plan=gpu(known[mask]);noise=gpu(epsilon);y=gpu(reward)
    models={name:initialized_network('dynamics' if name in ('cm','dynamics_off') else 'value').cuda().eval() for name in cp['models']}
    for name,model in models.items():
        model.load_state_dict(cp['models'][name])
        for parameter in model.parameters():parameter.requires_grad_(False)
    actor=initialized_network('actor').cuda().eval();actor.load_state_dict(bundle['actor_initial_parameters']);parameters=dict(actor.named_parameters())
    mean=actor(x);assert not mean.any() and not actor[4].weight.any() and not actor[4].bias.any()
    zero=torch.zeros_like(mean);baseline=scores(x,zero,plan,models,gpu(cp['delta_mean']),gpu(cp['delta_std']),'direct_q').detach()
    derivatives={};outputs={}
    for name in VARIANTS:
        raw=zero.clone().requires_grad_(True);value=scores(x,raw,plan,models,gpu(cp['delta_mean']),gpu(cp['delta_std']),name)
        derivatives[name]=torch.autograd.grad(value.sum(),raw)[0].detach();outputs[name]=value.detach().cpu().numpy()
    vectors={'baseline':(y-baseline)[:,None]*noise}
    for name,b in derivatives.items():vectors[name]=(y-baseline-(b*noise).sum(-1))[:,None]*noise+b
    def scalar(params,observation,vector):return (functional_call(actor,params,(observation[None],))[0]*vector).sum()
    gradients={name:vmap(grad(scalar),in_dims=(None,0,0))(parameters,x,vector) for name,vector in vectors.items()}
    active={}
    for name,g in gradients.items():
        assert all(torch.isfinite(v).all() for v in g.values())
        assert all(not g[k].any() for k in ('0.weight','0.bias','2.weight','2.bias'))
        active[name]=torch.cat((g['4.weight'].flatten(1),g['4.bias']),-1).detach().double()
    generator=torch.Generator(device='cpu').manual_seed(3557);motion_selected=motion[mask]
    idx=np.concatenate([np.flatnonzero(motion_selected==j)[torch.randint(192,(1000,192),generator=generator).numpy()] for j in range(3)],1)
    weights=torch.zeros((1000,576),dtype=torch.float64,device=x.device);weights.scatter_add_(1,gpu(idx),torch.full((1000,576),1/576,dtype=torch.float64,device=x.device))
    variance={};bootstrap={}
    for name,g in active.items():
        norm=g.square().sum(-1);variance[name]=float((norm.mean()-g.mean(0).square().sum())*576/575)
        bootstrap[name]=((weights@norm)-(weights@g).square().sum(-1))*576/575
    intervals={name:np.quantile((bootstrap['cm']-bootstrap[name]).cpu().numpy(),(.025,.975)).tolist() for name in ('baseline','dynamics_off','direct_q')}
    gates=dict(gain10percent_all=all(variance['cm']<=.9*variance[name] for name in intervals),paired_interval_upper_negative_all=all(value[1]<0 for value in intervals.values()))
    packet=dict(current=current,compact=compact,extra=extra,known=known,mask=mask,epsilon=epsilon,reward=reward,baseline=baseline.cpu().numpy(),derivatives={k:v.cpu().numpy() for k,v in derivatives.items()},critic_outputs=outputs,raw_gradient_vectors={k:v.cpu().numpy() for k,v in vectors.items()},parameter_gradients={k:{n:v.detach().cpu() for n,v in g.items()} for k,g in gradients.items()},bootstrap_indices=idx)
    assert not (root/'gradients.pt').exists();torch.save(packet,root/'gradients.pt')
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',complete_parameter_gradient_variance=variance,paired_bootstrap_interval_cm_minus_control=intervals,gates=gates,stochastic_episodes=576,all_native_trajectories=768,actor_parameter_count=sum(p.numel() for p in actor.parameters()),hidden_parameter_gradients_exact_zero=True,physical_future_not_used=True,models_fixed_before_fresh_actions_and_returns=True,no_actor_updates=True,no_policy_utility_claim=True,formal_validation=False,source_model_pretraining_steps=6000,source_prior_offline_actor_steps=3000,new_optimizer_steps=0,gradients_sha256=sha(root/'gradients.pt'),wall_seconds=time.monotonic()-begin)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
