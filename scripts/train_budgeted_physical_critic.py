"""Joint measured physics/return critic fitting at an explicit collection budget."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.paired_evaluation import fingerprint
from src.task.CmResidual.option_model_policy import SCHEMA as ACTOR_SCHEMA,DYNAMIC,state_features,initialized_network,numpy_forward
from src.task.CmResidual.budgeted_physical_q import SCHEMA,read_panel,initialized_critic,numpy_head
NAMES=('cm','dynamics_off','cold_q','budget_q')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists() and torch.cuda.is_available();begin=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((root/'run_manifest.json').read_text());assert manifest['experiment_id']=='P-20261003-budgeted-physical-critic';base=torch.load(manifest['base_checkpoint'],map_location='cpu',weights_only=False)
    data={str(seed):read_panel(root/f's{seed}',base) for seed in (651,652,653,654)};assert not data['651']['label_available'].any() and np.isnan(data['651']['reward']).all() and not data['652']['label_available'].any() and np.isnan(data['652']['reward']).all() and data['653']['label_available'].all() and data['654']['label_available'].all()
    extra=np.concatenate([data[s][k] for s in ('651','652','653') for k in ('extra','future_extra')]);mean=extra.mean(0);std=extra.std(0).clip(.001)
    current={s:state_features(d['current'],d['extra'],mean,std) for s,d in data.items()};future={s:state_features(d['future'],d['future_extra'],mean,std) for s,d in data.items()}
    px=np.concatenate([current[s] for s in ('651','652','653')]);pa=np.concatenate([data[s]['raw'] for s in ('651','652','653')]);delta=np.concatenate([future[s][:,DYNAMIC]-current[s][:,DYNAMIC] for s in ('651','652','653')]);ym=delta.mean(0);ys=delta.std(0).clip(.001)
    xs={name:np.concatenate([current[s] for s in (('653','654') if name=='budget_q' else ('653',))]) for name in NAMES};ra={name:np.concatenate([data[s]['raw'] for s in (('653','654') if name=='budget_q' else ('653',))]) for name in NAMES};ys_task={name:np.concatenate([data[s]['reward'] for s in (('653','654') if name=='budget_q' else ('653',))]) for name in NAMES}
    gpu=lambda x:torch.from_numpy(x).cuda();xphys=gpu(px);aphys=torch.tanh(gpu(pa));yphys=gpu((delta-ym)/ys);x={name:gpu(xs[name]) for name in NAMES};action={name:torch.tanh(gpu(ra[name])) for name in NAMES};target={name:gpu(ys_task[name]) for name in NAMES};out.mkdir()
    critics={name:initialized_critic().cuda() for name in NAMES};initial={k:v.detach().cpu().clone() for k,v in critics['cm'].state_dict().items()};assert all(all(torch.equal(initial[k],critics[name].state_dict()[k].cpu()) for k in initial) for name in NAMES)
    optim={name:torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-4,foreach=False) for name,net in critics.items()};phygen=torch.Generator(device='cpu').manual_seed(3652);taskgen=torch.Generator(device='cpu').manual_seed(3654);budgetgen=torch.Generator(device='cpu').manual_seed(3654);losses={};first_gradient={}
    for step in range(1500):
        pi=torch.randint(2304,(256,),generator=phygen).cuda();ti=torch.randint(768,(256,),generator=taskgen).cuda();bi=torch.randint(1536,(256,),generator=budgetgen).cuda()
        for name,net in critics.items():
            idx=bi if name=='budget_q' else ti;pred=net.task_value(torch.cat((x[name][idx],action[name][idx]),-1));taskloss=torch.nn.functional.binary_cross_entropy(pred,target[name][idx]);phyaction=aphys[pi] if name=='cm' else torch.zeros_like(aphys[pi]);physical=net.physical_value(torch.cat((xphys[pi],phyaction),-1));phyloss=(physical-yphys[pi]).square().mean();loss=taskloss+( .05 if name in ('cm','dynamics_off') else 0.)*phyloss
            optim[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10);assert torch.isfinite(norm)
            if step==0:first_gradient[name]=float(norm)
            optim[name].step();losses[name]=dict(task=float(taskloss.detach()),physical=float(phyloss.detach()),total=float(loss.detach()))
        if step%500==499:print(json.dumps(dict(joint_critic_updates_each=step+1,loss=losses)),flush=True)
    torch.save(dict(critic_updates_each=1500,critics={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in critics.items()},optimizers={name:opt.state_dict() for name,opt in optim.items()},generators={name:g.get_state() for name,g in [('physical',phygen),('common_task',taskgen),('budget_task',budgetgen)]}),out/'critic_step1500.pt')
    for net in critics.values():
        net.eval()
        for parameter in net.parameters():parameter.requires_grad_(False)
    frozen={name:fingerprint(net.state_dict()) for name,net in critics.items()};actors={name:initialized_network('actor').cuda() for name in NAMES};actor_initial={k:v.detach().cpu().clone() for k,v in actors['cm'].state_dict().items()};assert all(all(torch.equal(actor_initial[k],actors[name].state_dict()[k].cpu()) for k in actor_initial) for name in NAMES)
    actor_opt={name:torch.optim.Adam(net.parameters(),lr=3e-4,foreach=False) for name,net in actors.items()};common_gen=torch.Generator(device='cpu').manual_seed(3556);budget_gen=torch.Generator(device='cpu').manual_seed(3556);actor_losses={};first_actor_gradient={}
    for step in range(1000):
        ti=torch.randint(768,(256,),generator=common_gen).cuda();bi=torch.randint(1536,(256,),generator=budget_gen).cuda()
        for name,net in actors.items():
            idx=bi if name=='budget_q' else ti;raw=net(x[name][idx]);prob=critics[name].task_value(torch.cat((x[name][idx],torch.tanh(raw)),-1));loss=-prob.mean()+.05*raw.square().mean();actor_opt[name].zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),10);assert torch.isfinite(norm)
            if step==0:first_actor_gradient[name]=float(norm)
            actor_opt[name].step();actor_losses[name]=float(loss.detach())
        if step%500==499:print(json.dumps(dict(actor_updates_each=step+1,loss=actor_losses)),flush=True)
        if step in (499,999):torch.save(dict(actor_updates_each=step+1,actors={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in actors.items()},optimizers={name:opt.state_dict() for name,opt in actor_opt.items()},generators={'common':common_gen.get_state(),'budget':budget_gen.get_state()},critic_checkpoint='critic_step1500.pt'),out/f'actor_step{step+1:04d}.pt')
    assert all(fingerprint(critics[name].state_dict())==frozen[name] for name in NAMES);cstates={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in critics.items()};astates={name:{k:v.detach().cpu() for k,v in net.state_dict().items()} for name,net in actors.items()};task_outputs={};physical_outputs={};actor_outputs={};errors={};changes={}
    with torch.no_grad():
        for name in NAMES:
            inp=torch.cat((x[name],action[name]),-1);actual=critics[name].task_value(inp).cpu().numpy();independent=numpy_head(cstates[name],inp.cpu().numpy(),'task').flatten();errors[name+'_task']=float(np.abs(actual-independent).max());assert errors[name+'_task']<=2e-5;task_outputs[name]=actual
            pinput=torch.cat((xphys,aphys if name=='cm' else torch.zeros_like(aphys)),-1);actual=critics[name].physical_value(pinput).cpu().numpy();independent=numpy_head(cstates[name],pinput.cpu().numpy(),'physical');errors[name+'_physical']=float(np.abs(actual-independent).max());assert errors[name+'_physical']<=2e-5;physical_outputs[name]=actual
            actual=actors[name](x[name]).cpu().numpy();independent=numpy_forward(astates[name],xs[name],'tanh');errors[name+'_actor']=float(np.abs(actual-independent).max());assert errors[name+'_actor']<=2e-5 and np.abs(actual).max()<=1+1e-6;actor_outputs[name]=actual
            changes[name]=dict(critic=max(float((cstates[name][k]-initial[k]).abs().max()) for k in initial),actor=max(float((astates[name][k]-actor_initial[k]).abs().max()) for k in actor_initial))
    torch.save(dict(schema=SCHEMA,critics=cstates,initial_parameters=initial,extra_mean=mean,extra_std=std,delta_mean=ym,delta_std=ys,critic_updates_each=1500,joint_physical_supervision_only_on_off=True,never_composes_predicted_future_values=True),out/'models.pt')
    for suffix,control in [('', 'cold_q'),('_budget','budget_q')]:
        torch.save(dict(schema=ACTOR_SCHEMA,training_critic_schema=SCHEMA,actors={'cm':astates['cm'],'dynamics_off':astates['dynamics_off'],'direct_q':astates[control]},control_name=control,extra_mean=mean,extra_std=std,actor_updates_each=1000,common_initial_fingerprint=fingerprint(actor_initial),actor_initial_parameters=actor_initial,source_policy_sha256=manifest['policy_sha256'],deploy_model_or_future_input=False),out/f'actors{suffix}.pt')
    torch.save(dict(data=data,current=current,future=future,physical_current=px,physical_raw=pa,physical_delta=delta,task_current=xs,task_raw=ra,task_labels=ys_task,task_gpu_outputs=task_outputs,physical_gpu_outputs=physical_outputs,actor_gpu_outputs=actor_outputs),out/'fit_rows.pt')
    result=dict(run_status='COMPLETED',actual_joint_critic_optimizer_steps=6000,physical_auxiliary_updates_subset=3000,actual_actor_optimizer_steps=4000,short_physical_episodes=1536,common_complete_episodes=768,extra_complete_episodes_budget_q_only=768,env_control_ticks_cm_off=310272,env_control_ticks_budget_q=310272,no_terminal_labels_for_short=True,common_initial_parameters=True,first_critic_gradient_norm=first_gradient,first_actor_gradient_norm=first_actor_gradient,parameter_changes=changes,all_final_numpy_maximum=errors,last_loss=losses,models_sha256=sha(out/'models.pt'),actor_sha256={'common':sha(out/'actors.pt'),'budget':sha(out/'actors_budget.pt')},optimizer_not_independently_replayed=True,wall_seconds=time.monotonic()-begin)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
