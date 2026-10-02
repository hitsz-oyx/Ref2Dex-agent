"""Measured-return task fits and actual actor learning from reused physical encoders."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.option_model_policy import SCHEMA as ACTOR_SCHEMA,read_options,state_features,initialized_network,numpy_forward
from src.task.CmResidual.physical_encoder_critic import SCHEMA,ENCODER_KEYS,transferred_critic


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists() and torch.cuda.is_available();begin=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=json.loads((root/'run_manifest.json').read_text());assert m['experiment_id']=='P-20261002-physical-encoder-critic';source=Path(m['fit_source']);old=torch.load(source/'fit/models.pt',map_location='cpu',weights_only=False);old_actor=torch.load(source/'fit/actors.pt',map_location='cpu',weights_only=False);base=torch.load(m['base_checkpoint'],map_location='cpu',weights_only=False)
    parts=[read_options(source/f's{seed}',base) for seed in (603,604)];data={k:np.concatenate([part[k] for part in parts]) for k in parts[0]};previous=torch.load(source/'fit/fit_rows.pt',map_location='cpu',weights_only=False)
    current=state_features(data['current'],data['extra'],old['extra_mean'],old['extra_std']);assert len(current)==1536 and np.array_equal(current,previous['current'])
    gpu=lambda a:torch.from_numpy(a).cuda();x=gpu(current);action=torch.tanh(gpu(data['raw']));y=gpu(data['reward']);qinputs=torch.cat((x,action),-1);out.mkdir();names=('cm','dynamics_off')
    critics={name:transferred_critic(old['models'][name]).cuda() for name in names};initial={name:{k:v.detach().cpu().clone() for k,v in net.state_dict().items()} for name,net in critics.items()};common_head=initialized_network('value').state_dict()
    for name in names:
        for key in ENCODER_KEYS:assert torch.equal(initial[name][key],old['models'][name][key])
        for key in ('4.weight','4.bias'):assert torch.equal(initial[name][key],common_head[key])
    optimizer={name:torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False) for name,net in critics.items()};generator=torch.Generator(device='cpu').manual_seed(3554);losses={};first_critic_gradient={}
    for step in range(1500):
        idx=torch.randint(1536,(256,),generator=generator).cuda()
        for name,net in critics.items():
            pred=net(qinputs[idx]).flatten();loss=torch.nn.functional.binary_cross_entropy(pred,y[idx]);optimizer[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10);assert torch.isfinite(norm)
            if step==0:first_critic_gradient[name]=float(norm)
            optimizer[name].step();losses[name]=float(loss.detach())
        if step%500==499:print(json.dumps(dict(task_critic_updates_each=step+1,measured_return_loss=losses)),flush=True)
    torch.save(dict(schema=SCHEMA,task_critic_updates_each=1500,critics={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in critics.items()},optimizers={name:op.state_dict() for name,op in optimizer.items()},generator_state=generator.get_state()),out/'critic_step1500.pt')
    for net in critics.values():
        net.eval()
        for parameter in net.parameters():parameter.requires_grad_(False)
    frozen={name:fingerprint(net.state_dict()) for name,net in critics.items()};actors={name:initialized_network('actor').cuda() for name in names}
    for actor in actors.values():assert all(torch.equal(actor.state_dict()[key].cpu(),v) for key,v in old_actor['actor_initial_parameters'].items())
    actor_optimizer={name:torch.optim.Adam(net.parameters(),lr=3e-4,foreach=False) for name,net in actors.items()};generator=torch.Generator(device='cpu').manual_seed(3556);actor_losses={};first_actor_gradient={}
    for step in range(1000):
        idx=torch.randint(1536,(256,),generator=generator).cuda()
        for name,actor in actors.items():
            raw=actor(x[idx]);pred=critics[name](torch.cat((x[idx],torch.tanh(raw)),-1)).flatten();loss=-pred.mean()+.05*raw.square().mean();actor_optimizer[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),10);assert torch.isfinite(norm)
            if step==0:first_actor_gradient[name]=float(norm)
            actor_optimizer[name].step();actor_losses[name]=float(loss.detach())
        if step%500==499:print(json.dumps(dict(actor_updates_each=step+1,loss=actor_losses)),flush=True)
        if step in (499,999):torch.save(dict(actor_updates_each=step+1,actors={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in actors.items()},optimizers={name:op.state_dict() for name,op in actor_optimizer.items()},generator_state=generator.get_state(),critic_checkpoint='critic_step1500.pt'),out/f'actor_step{step+1:04d}.pt')
    assert all(fingerprint(critics[name].state_dict())==frozen[name] for name in names)
    critic_states={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in critics.items()};actor_states={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in actors.items()};actor_states['direct_q']=old_actor['actors']['direct_q'];forwards={};errors={};actor_outputs={};actor_errors={};critic_changes={};actor_changes={}
    with torch.no_grad():
        for name in names:
            pred=critics[name](qinputs).cpu().numpy();rebuilt=numpy_forward(critic_states[name],qinputs.cpu().numpy(),'sigmoid');errors[name]=float(np.abs(pred-rebuilt).max());assert errors[name]<=2e-5 and np.isfinite(pred).all();forwards[name]=pred;critic_changes[name]=max(float((critic_states[name][key]-initial[name][key]).abs().max()) for key in initial[name]);assert critic_changes[name]>0
        for name,parameters in actor_states.items():
            net=initialized_network('actor').cuda().eval();net.load_state_dict(parameters);pred=net(x).cpu().numpy();rebuilt=numpy_forward(parameters,current,'tanh');actor_errors[name]=float(np.abs(pred-rebuilt).max());assert actor_errors[name]<=2e-5 and np.isfinite(pred).all() and np.abs(pred).max()<=1+1e-6;actor_outputs[name]=pred;actor_changes[name]=max(float((parameters[key]-old_actor['actor_initial_parameters'][key]).abs().max()) for key in parameters)
    torch.save(dict(schema=SCHEMA,critics=critic_states,initial_parameters=initial,extra_mean=old['extra_mean'],extra_std=old['extra_std'],task_critic_updates_each=1500,measured_returns_only=True,physical_decoder_discarded=True,all_critic_parameters_trainable=True),out/'models.pt')
    torch.save(dict(schema=ACTOR_SCHEMA,training_critic_schema=SCHEMA,actors=actor_states,extra_mean=old['extra_mean'],extra_std=old['extra_std'],actor_updates_each=1000,common_initial_fingerprint=fingerprint(old_actor['actor_initial_parameters']),actor_initial_parameters=old_actor['actor_initial_parameters'],source_policy_sha256=m['policy_sha256'],deploy_model_or_future_input=False,direct_q_reused_without_optimizer_repeat=True),out/'actors.pt')
    torch.save(dict(data=data,current=current,task_inputs=qinputs.cpu().numpy(),critic_gpu_outputs=forwards,actor_gpu_outputs=actor_outputs),out/'fit_rows.pt')
    result=dict(run_status='COMPLETED',new_task_critic_optimizer_steps=3000,new_actor_optimizer_steps=2000,reused_physical_model_steps=3000,reused_direct_q_actor_steps=1000,source_prior_model_steps=6000,source_prior_actor_steps=3000,fit_episodes_reused=1536,no_new_fit_physics=True,initial_physical_encoders_exact=True,common_fresh_task_head=True,measured_returns_only=True,no_future_prediction_in_actor_objective=True,all_task_critic_parameters_trainable=True,critic_first_gradient_norm=first_critic_gradient,actor_first_gradient_norm=first_actor_gradient,critic_parameter_max_change=critic_changes,actor_parameter_max_change=actor_changes,task_critic_forward_numpy_maximum=errors,actor_forward_numpy_maximum=actor_errors,last_task_loss=losses,last_actor_loss=actor_losses,models_sha256=sha(out/'models.pt'),actors_sha256=sha(out/'actors.pt'),optimizer_not_independently_replayed=True,wall_seconds=time.monotonic()-begin)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
