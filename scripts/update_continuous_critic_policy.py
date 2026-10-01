"""One audited fresh panel: four fixed PPO epochs for three matched policies."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.task.CmResidual.continuous_critic_cm import initialized_models,actor_loss,critic_loss,SCHEMA
from src.task.CmResidual.paired_evaluation import fingerprint
from scripts.run_contact_response_probe import sha

VARIANTS=('cm','state_only','none')

def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return type(v)((k,cpu(x)) for k,x in v.items())
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return copy.deepcopy(v)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or ROOT not in out.parents or not torch.cuda.is_available():raise ValueError('unique GPU update required')
    begin=time.monotonic();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=torch.load(a.previous,map_location='cpu',weights_only=False);update=old['updates']+1
    audit=json.loads((a.panel/'panel_audit.json').read_text());result=json.loads((a.panel/'results.json').read_text());rows=json.loads((a.panel/'rows.json').read_text())
    if old['schema']!=SCHEMA or old['seed']!=762 or not 1<=update<=20 or result['continuous_updates']!=update-1 or result['deterministic'] or rows[0]['seed']!=546+update or audit['run_status']!='COMPLETED':raise ValueError('fixed fresh on-policy update order')
    if sha(a.previous)!=result['continuous_checkpoint_sha256'] or sha(a.panel/'trace.pt')!=result['trace_sha256']:raise ValueError('audited behavior provenance')
    initial=torch.load(a.panel/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(a.panel/'trace.pt',map_location='cpu',weights_only=False)
    reward=torch.tensor([r['physical105'] for r in rows],dtype=torch.float32)
    if [r['environment'] for r in rows]!=list(range(768)):raise ValueError('full physical row order')
    device=torch.device('cuda:0');records={};payload={};summaries={}
    for group,variant in enumerate(VARIANTS,1):
        ids=(initial['policy_group']==group).nonzero().flatten()
        if len(ids)!=192:raise ValueError('matched interaction budget')
        # Flatten all ACTUAL 202 steps. No future measured state enters inputs.
        x=trace['normalized_context'][:,ids].reshape(-1,70).to(device)
        request=trace['request'][:,ids].reshape(-1,12).to(device)
        logp=trace['request_logprob'][:,ids].reshape(-1).to(device)
        action=trace['executed_action12'][:,ids].reshape(-1,12).to(device)
        transition=trace['physical_transition'][:,ids].reshape(-1,6).to(device)
        ret=reward[ids].expand(202,-1).reshape(-1).to(device)
        behavior_v=trace['critic_value'][:,ids].reshape(-1).to(device)
        advantage=ret-behavior_v;adv_mean=advantage.mean();adv_std=advantage.std(unbiased=False)
        advantage=(advantage-adv_mean)/(adv_std+1e-8)
        actor,critic=initialized_models();actor.load_state_dict(old['variants'][variant]['actor']);critic.load_state_dict(old['variants'][variant]['critic'])
        actor=actor.to(device).train();critic=critic.to(device).train()
        params=list(actor.parameters())+list(critic.parameters());named=list(actor.named_parameters())+ [('critic.'+k,v) for k,v in critic.named_parameters()]
        optimizer=torch.optim.Adam(params,lr=3e-4,foreach=False)
        if 'optimizer' in old['variants'][variant]:optimizer.load_state_dict(old['variants'][variant]['optimizer'])
        generator=torch.Generator(device=device).manual_seed(762+20000+update)
        batches=0;losses=[];first=None
        for epoch in range(4):
            order=torch.randperm(len(x),device=device,generator=generator)
            for start in range(0,len(x),1024):
                ix=order[start:start+1024]
                if batches==0:
                    first=dict(indices=cpu(ix),actor_before=cpu(actor.state_dict()),critic_before=cpu(critic.state_dict()),optimizer_before=cpu(optimizer.state_dict()),input=cpu(x[ix]),request=cpu(request[ix]),old_logp=cpu(logp[ix]),action=cpu(action[ix]),transition=cpu(transition[ix]),return_target=cpu(ret[ix]),advantage=cpu(advantage[ix]),advantage_mean=float(adv_mean),advantage_std=float(adv_std))
                mean,logstd=actor(x[ix]);value,prediction=critic(x[ix],action[ix],variant)
                al=actor_loss(mean,logstd,request[ix],logp[ix],advantage[ix]);cl=critic_loss(value,prediction,ret[ix],transition[ix],variant);loss=al+cl
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if batches==0:first.update(mean=cpu(mean),logstd=cpu(logstd),value=cpu(value),prediction=cpu(prediction),actor_loss=float(al),critic_loss=float(cl),gradient={k:cpu(v.grad) for k,v in named})
                norm=torch.nn.utils.clip_grad_norm_(params,.5)
                if not torch.isfinite(norm):raise ValueError('finite joint PPO gradient')
                optimizer.step()
                if batches==0:first.update(gradient_norm=float(norm),actor_after=cpu(actor.state_dict()),critic_after=cpu(critic.state_dict()),optimizer_after=cpu(optimizer.state_dict()))
                batches+=1;losses.append([float(al),float(cl),float(norm)])
        if batches!=152 or any(not torch.isfinite(v).all() for v in params):raise ValueError('four complete fixed epochs')
        if fingerprint(actor.state_dict())==old['variants'][variant]['actor_fingerprint']:raise ValueError('actor did not update')
        records[variant]=first
        payload[variant]=dict(actor=cpu(actor.state_dict()),critic=cpu(critic.state_dict()),optimizer=cpu(optimizer.state_dict()),actor_fingerprint=fingerprint(actor.state_dict()),critic_fingerprint=fingerprint(critic.state_dict()))
        summaries[variant]=dict(episodes=192,transitions=len(x),epochs=4,minibatches=batches,physical105_count=int(reward[ids].sum()),advantage_mean=float(adv_mean),advantage_std=float(adv_std),first_losses=losses[0],last_losses=losses[-1],actor_changed=True)
    out.mkdir();torch.save(dict(schema=SCHEMA,seed=762,updates=update,variants=payload,official_actor_weights_used=False,previous_sha256=sha(a.previous),panel_trace_sha256=sha(a.panel/'trace.pt')),out/'policy_heads.pt')
    torch.save(dict(update=update,seed=546+update,panel=str(a.panel.resolve()),previous=str(a.previous.resolve()),previous_sha256=sha(a.previous),panel_sha256={f:sha(a.panel/f) for f in ('initial.pt','trace.pt','rows.json','panel_audit.json')},records=records,optimizer_hyperparameters=dict(lr=3e-4,betas=[.9,.999],eps=1e-8,gradclip=.5)),out/'update_packet.pt')
    r=dict(run_status='COMPLETED',update=update,variants=summaries,actual_task_return_only=True,no_predicted_reward=True,checkpoint_sha256=sha(out/'policy_heads.pt'),wall_seconds=time.monotonic()-begin)
    (out/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

if __name__=='__main__':main()
