"""New physical-law/actor fits on old FIT only; reuse valid direct-Q control."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.option_model_policy import SCHEMA as ACTOR_SCHEMA,DYNAMIC,read_options,state_features,initialized_network,numpy_forward
from src.task.CmResidual.empirical_successor_policy import SCHEMA,encoder,inputs,log_weights,physical_kernel,score,expected_value


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists() and torch.cuda.is_available()
    begin=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=json.loads((root/'run_manifest.json').read_text());assert m['experiment_id']=='P-20261002-empirical-successor-policy';source=Path(m['fit_source']);cp=torch.load(source/'fit/models.pt',map_location='cpu',weights_only=False);old_actor=torch.load(source/'fit/actors.pt',map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    parts=[read_options(source/f's{seed}',base) for seed in (603,604)];data={key:np.concatenate([p[key] for p in parts]) for key in parts[0]};assert len(data['reward'])==1536
    current=state_features(data['current'],data['extra'],cp['extra_mean'],cp['extra_std']);future=state_features(data['future'],data['future_extra'],cp['extra_mean'],cp['extra_std']);previous=torch.load(source/'fit/fit_rows.pt',map_location='cpu',weights_only=False)
    assert np.array_equal(current,previous['current']) and np.array_equal(future,previous['future'])
    kernel=physical_kernel(future);gpu=lambda x:torch.from_numpy(x).cuda();x=gpu(current);raw=gpu(data['raw']);known=gpu(data['known']);physical=gpu(future[:,DYNAMIC]);klog=gpu(kernel)
    out.mkdir()
    names=('cm','dynamics_off');models={name:encoder().cuda() for name in names};initial_model={key:v.detach().cpu().clone() for key,v in models['cm'].state_dict().items()}
    assert all(torch.equal(initial_model[k],models['dynamics_off'].state_dict()[k].cpu()) for k in initial_model)
    opt={name:torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False) for name,model in models.items()};generator=torch.Generator(device='cpu').manual_seed(3564);losses={}
    for step in range(1500):
        idx=torch.randint(1536,(256,),generator=generator).cuda()
        for name,model in models.items():
            donor=inputs(x,raw,name);logw=log_weights(model,donor[idx],donor,idx);loss=-torch.logsumexp(logw+klog[idx],-1).mean()
            opt[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10);assert torch.isfinite(norm);opt[name].step();losses[name]=float(loss.detach())
        if step%500==499:print(json.dumps(dict(physical_updates_each=step+1,kernel_observation_loss=losses)),flush=True)
    for model in models.values():
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
    torch.save(dict(schema=SCHEMA,models={name:{k:v.detach().cpu() for k,v in model.state_dict().items()} for name,model in models.items()},model_updates_each=1500,source_fit=str(source)),out/'physical_update1500.pt')
    value=initialized_network('value').cuda().eval();value.load_state_dict(cp['models']['value'])
    for p in value.parameters():p.requires_grad_(False)
    value_before=fingerprint(value.state_dict());actors={name:initialized_network('actor').cuda() for name in names}
    for actor in actors.values():
        assert all(torch.equal(actor.state_dict()[key].cpu(),v) for key,v in old_actor['actor_initial_parameters'].items())
    actor_opt={name:torch.optim.Adam(actor.parameters(),lr=3e-4,foreach=False) for name,actor in actors.items()};generator=torch.Generator(device='cpu').manual_seed(3556);actor_loss={};first_gradient={}
    for step in range(1000):
        idx=torch.randint(1536,(256,),generator=generator).cuda()
        for name,actor in actors.items():
            proposal=actor(x[idx]);probability=score(models[name],value,x[idx],proposal,known[idx],x,raw,physical,name,idx);loss=-probability.mean()+.05*proposal.square().mean()
            actor_opt[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),10);assert torch.isfinite(norm)
            if step==0:first_gradient[name]=float(norm)
            actor_opt[name].step();actor_loss[name]=float(loss.detach())
        if step%500==499:print(json.dumps(dict(actor_updates_each=step+1,loss=actor_loss)),flush=True)
        if step in (499,999):
            torch.save(dict(actor_updates_each=step+1,actors={name:{k:v.detach().cpu() for k,v in actor.state_dict().items()} for name,actor in actors.items()},optimizers={name:optimizer.state_dict() for name,optimizer in actor_opt.items()},generator_state=generator.get_state(),physical_checkpoint='physical_update1500.pt'),out/f'actor_step{step+1:04d}.pt')
    assert fingerprint(value.state_dict())==value_before
    model_states={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in models.items()};actor_states={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in actors.items()};actor_states['direct_q']=old_actor['actors']['direct_q']
    actor_outputs={};actor_errors={};changes={};encoder_outputs={};probabilities={};log_probabilities={};expected={};mean_scores={};kernel_scores={}
    with torch.no_grad():
        for name in names:
            donor=inputs(x,raw,name);encoder_outputs[name]=models[name](donor).cpu().numpy();weights=[];logweights=[];pv=[];mv=[];kv=[]
            for start in range(0,1536,32):
                idx=torch.arange(start,min(start+32,1536),device=x.device);logw=log_weights(models[name],donor[idx],donor,idx);w=logw.exp();weights.append(w.cpu().numpy());logweights.append(logw.cpu().numpy());pv.append(expected_value(value,physical,known[idx],torch.tanh(raw[idx]),w).cpu().numpy());kv.append((-torch.logsumexp(logw+klog[idx],-1)).cpu().numpy())
                mean_state=x[idx].clone();mean_state[:,51:72]=known[idx];mean_state[:,DYNAMIC]=w@physical;mv.append(value(torch.cat((mean_state,torch.tanh(raw[idx])),-1)).flatten().cpu().numpy())
            probabilities[name]=np.concatenate(weights);log_probabilities[name]=np.concatenate(logweights);expected[name]=np.concatenate(pv);mean_scores[name]=np.concatenate(mv);kernel_scores[name]=np.concatenate(kv)
        for name,state in actor_states.items():
            net=initialized_network('actor').cuda().eval();net.load_state_dict(state);actual=net(x).cpu().numpy();rebuilt=numpy_forward(state,current,'tanh');actor_errors[name]=float(np.abs(actual-rebuilt).max());assert actor_errors[name]<=2e-5 and np.isfinite(actual).all() and np.abs(actual).max()<=1+1e-6;actor_outputs[name]=actual
            changes[name]=max(float((state[k]-old_actor['actor_initial_parameters'][k]).abs().max()) for k in state)
    assert all(torch.equal(actor_states['direct_q'][k],old_actor['actors']['direct_q'][k]) for k in actor_states['direct_q'])
    torch.save(dict(schema=SCHEMA,models=model_states,support_current=current,support_future=future,support_raw=data['raw'],extra_mean=cp['extra_mean'],extra_std=cp['extra_std'],model_updates_each=1500,initial_parameters=initial_model,own_episode_excluded_in_model_and_actor_fit=True),out/'models.pt')
    torch.save(dict(schema=ACTOR_SCHEMA,training_physical_model_schema=SCHEMA,actors=actor_states,extra_mean=cp['extra_mean'],extra_std=cp['extra_std'],actor_updates_each=1000,common_initial_fingerprint=fingerprint(old_actor['actor_initial_parameters']),actor_initial_parameters=old_actor['actor_initial_parameters'],source_policy_sha256=m['policy_sha256'],deploy_model_or_future_input=False,direct_q_reused_without_optimizer_repeat=True,pretraining_only=True),out/'actors.pt')
    torch.save(dict(data=data,current=current,future=future,kernel_logscore=kernel,encoder_gpu_outputs=encoder_outputs,probabilities=probabilities,log_probabilities=log_probabilities,expected_value=expected,mean_value=mean_scores,kernel_scores=kernel_scores,actor_gpu_outputs=actor_outputs),out/'fit_rows.pt')
    result=dict(run_status='COMPLETED',new_physical_optimizer_steps=3000,new_actor_optimizer_steps=2000,reused_direct_q_actor_steps=1000,source_prior_model_steps=6000,source_prior_actor_steps=3000,fit_episodes_reused=1536,no_new_fit_physics=True,direct_q_weights_exact_source=True,same_actor_initialization=True,actor_first_gradient_norm=first_gradient,actor_parameter_max_change=changes,actor_forward_numpy_maximum=actor_errors,last_kernel_loss=losses,last_actor_loss=actor_loss,models_sha256=sha(out/'models.pt'),actors_sha256=sha(out/'actors.pt'),physical_support_rows=1536,value_model_unchanged=True,no_return_in_physical_model_loss=True,no_optimizer_replay_claim=True,no_policy_utility_claim_before_native=True,model_parameter_counts={k:sum(p.numel() for p in net.parameters()) for k,net in models.items()},wall_seconds=time.monotonic()-begin)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
