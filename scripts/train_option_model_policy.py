"""Fit actual successors/returns and optimize three matched offline option actors."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.option_model_policy import SCHEMA,VARIANTS,DYNAMIC,read_options,state_features,initialized_network,scores,numpy_forward


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists() and torch.cuda.is_available()
    begin=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=json.loads((root/'run_manifest.json').read_text());assert m['experiment_id']=='P-20261002-option-model-policy';base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    parts=[read_options(root/f's{seed}',base) for seed in (603,604)]
    data={k:np.concatenate([part[k] for part in parts]) for k in parts[0]};assert len(data['reward'])==1536
    extra=np.concatenate((data['extra'],data['future_extra']));mean=extra.mean(0);std=extra.std(0).clip(.001)
    current=state_features(data['current'],data['extra'],mean,std);future=state_features(data['future'],data['future_extra'],mean,std)
    residual=future[:,DYNAMIC]-current[:,DYNAMIC];ym=residual.mean(0);ys=residual.std(0).clip(.001)
    gpu=lambda x:torch.from_numpy(x).cuda()
    xs=gpu(current);yn=gpu((residual-ym)/ys);y=gpu(data['reward']);known=gpu(data['known']);raw=gpu(data['raw']);action=torch.tanh(raw);xf=gpu(future)
    models={k:initialized_network('dynamics' if k in ('cm','dynamics_off') else 'value').cuda() for k in ('cm','dynamics_off','value','direct_q')}
    optim={k:torch.optim.Adam(v.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False) for k,v in models.items()}
    generators={seed:torch.Generator(device='cpu').manual_seed(seed) for seed in (3552,3554)};last={}
    for step in range(1500):
        physical_indices=torch.randint(len(y),(256,),generator=generators[3552]).cuda();value_indices=torch.randint(len(y),(256,),generator=generators[3554]).cuda()
        for name,model in models.items():
            indices=physical_indices if name in ('cm','dynamics_off') else value_indices
            option=action[indices] if name!='dynamics_off' else torch.zeros_like(action[indices])
            states=xf[indices] if name=='value' else xs[indices]
            output=model(torch.cat((states,option),-1))
            loss=(output-yn[indices]).square().mean() if name in ('cm','dynamics_off') else torch.nn.functional.binary_cross_entropy(output.flatten(),y[indices])
            optim[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(norm);optim[name].step();last[name]=float(loss.detach())
        if step%500==499:print(json.dumps(dict(model_updates_each=step+1,loss=last)),flush=True)
    for model in models.values():
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
    model_states={k:{n:v.detach().cpu() for n,v in model.state_dict().items()} for k,model in models.items()};model_errors={};model_outputs={}
    with torch.no_grad():
        for name in models:
            x=torch.cat((xf if name=='value' else xs,torch.zeros_like(action) if name=='dynamics_off' else action),-1)
            actual=models[name](x).cpu().numpy();rebuilt=numpy_forward(model_states[name],x.cpu().numpy(),'sigmoid' if name in ('value','direct_q') else None)
            model_outputs[name]=actual;model_errors[name]=float(np.abs(actual-rebuilt).max());assert model_errors[name]<=2e-5
    actors={k:initialized_network('actor').cuda() for k in VARIANTS};initial={k:{n:v.detach().cpu().clone() for n,v in actor.state_dict().items()} for k,actor in actors.items()}
    assert all(all(torch.equal(initial['cm'][n],initial[k][n]) for n in initial['cm']) for k in VARIANTS)
    actor_optim={k:torch.optim.Adam(actor.parameters(),lr=3e-4,foreach=False) for k,actor in actors.items()};generator=torch.Generator(device='cpu').manual_seed(3556);dmean=gpu(ym);dstd=gpu(ys);actor_loss={};first_gradient={}
    for step in range(1000):
        indices=torch.randint(len(y),(256,),generator=generator).cuda()
        for name,actor in actors.items():
            proposal=actor(xs[indices]);value=scores(xs[indices],proposal,known[indices],models,dmean,dstd,name)
            loss=-value.mean()+.05*proposal.square().mean();actor_optim[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),10);assert torch.isfinite(norm)
            if step==0:first_gradient[name]=float(norm)
            actor_optim[name].step();actor_loss[name]=float(loss.detach())
        if step%500==499:print(json.dumps(dict(actor_updates_each=step+1,loss=actor_loss)),flush=True)
    actor_states={k:{n:v.detach().cpu() for n,v in actor.state_dict().items()} for k,actor in actors.items()};errors={};parameter_change={};means={};actor_outputs={}
    with torch.no_grad():
        for name,actor in actors.items():
            actual=actor(xs).cpu().numpy();rebuilt=numpy_forward(actor_states[name],current,'tanh');errors[name]=float(np.abs(actual-rebuilt).max());assert errors[name]<=2e-5 and np.isfinite(actual).all() and np.abs(actual).max()<=1+1e-6
            actor_outputs[name]=actual;parameter_change[name]=max(float((actor_states[name][k]-initial[name][k]).abs().max()) for k in initial[name]);means[name]=dict(mean_absolute=float(np.abs(actual).mean()),maximum_absolute=float(np.abs(actual).max()))
    out.mkdir()
    torch.save(dict(schema=SCHEMA,models=model_states,extra_mean=mean,extra_std=std,delta_mean=ym,delta_std=ys,model_updates_each=1500,physical_indices=DYNAMIC),out/'models.pt')
    torch.save(dict(schema=SCHEMA,actors=actor_states,extra_mean=mean,extra_std=std,actor_updates_each=1000,common_initial_fingerprint=fingerprint(initial['cm']),actor_initial_parameters=initial['cm'],source_policy_sha256=m['policy_sha256'],pretraining_only=True,deploy_model_or_future_input=False),out/'actors.pt')
    torch.save(dict(raw_data=data,current=current,future=future,model_gpu_outputs=model_outputs,actor_gpu_outputs=actor_outputs),out/'fit_rows.pt')
    result=dict(run_status='COMPLETED',fit_episodes=1536,actual_model_optimizer_steps=6000,actual_actor_optimizer_steps=3000,model_forward_numpy_maximum=model_errors,actor_forward_numpy_maximum=errors,actor_parameter_max_change=parameter_change,actor_first_gradient_norm=first_gradient,actor_raw_means=means,last_model_loss=last,last_actor_loss=actor_loss,actors_sha256=sha(out/'actors.pt'),models_sha256=sha(out/'models.pt'),same_actor_initialization=True,future_clock_plan_analytic=True,no_policy_utility_claim_before_native_evaluation=True,optimizer_not_independently_replayed=True,wall_seconds=time.monotonic()-begin)
    result['model_parameter_counts']={name:sum(p.numel() for p in model.parameters()) for name,model in models.items()}
    result['actor_parameter_counts']={name:sum(p.numel() for p in model.parameters()) for name,model in actors.items()}
    result['actor_objective_model_calls_per_step']={'cm':2,'dynamics_off':2,'direct_q':1}
    result['shared_continuation_value']=True
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
