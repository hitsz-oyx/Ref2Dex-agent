# Separately retained u17 audit correction. Original auditor and all scalar limits unchanged.
"""Independent NumPy audit of fixed first minibatch per head in every panel.

This audits all gradients and the first Adam step, NOT every optimizer step.
Full panel source hashes, physical returns, schedules and final weights retained.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
DIAG=None
VARIANT=None
BRANCH_CERTIFICATES=[]
import torch
from scripts.run_contact_response_probe import sha

def array(v):return v.numpy().astype(np.float64)
def mlp(x,w,keys):
    acts=[x]
    for i,k in enumerate(keys):
        y=acts[-1]@array(w[k+'.weight']).T+array(w[k+'.bias'])
        acts.append(np.maximum(y,0) if i<len(keys)-1 else y)
    return acts[-1],acts
def backward(d,w,keys,acts):
    g={}
    for i in range(len(keys)-1,-1,-1):
        k=keys[i];g[k+'.weight']=d.T@acts[i];g[k+'.bias']=d.sum(0)
        d=d@array(w[k+'.weight'])
        if i:
            mask=acts[i]>0
            if keys==['network.0','network.2','network.4'] and i==1:
                actual=DIAG[VARIANT]['actor.network.0'].numpy().astype(np.float64)
                pre=acts[0]@array(w['network.0.weight']).T+array(w['network.0.bias'])
                # Standard FP32 dot-product rounding enclosure, including bias.
                unit=2**-24;gamma=(2*acts[0].shape[1]+1)*unit/(1-(2*acts[0].shape[1]+1)*unit)
                bound=gamma*(np.abs(acts[0])@np.abs(array(w['network.0.weight'])).T+np.abs(array(w['network.0.bias'])))+1e-12
                if not np.all(np.abs(pre-actual)<=bound):raise ValueError('GPU preactivation outside independent FP32 rounding enclosure')
                flips=np.argwhere((pre>0)!=(actual>0))
                for row,col in flips:
                    if abs(pre[row,col])>bound[row,col] or abs(actual[row,col])>bound[row,col]:raise ValueError('non-roundoff branch difference')
                    BRANCH_CERTIFICATES.append(dict(variant=VARIANT,sample=int(row),unit=int(col),numpy64=float(pre[row,col]),gpu32=float(actual[row,col]),rounding_bound=float(bound[row,col]),upstream_derivative=float(d[row,col])))
                mask=actual>0
            d=d*mask
    return g,d

def main():
    global DIAG,VARIANT
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.directory;begin=time.monotonic();torch.set_num_threads(2)
    if a.output.exists() or ROOT not in a.output.resolve().parents:raise ValueError('new own correction output')
    diagnosis=json.loads((a.diagnosis/'results.json').read_text())
    if not diagnosis['engineering_only'] or not diagnosis['no_optimizer_or_physics_updates']:raise ValueError('only frozen engineering replay')
    for path,h in diagnosis['source_sha256'].items():
        if sha(Path(path))!=h:raise ValueError('diagnosed source changed')
    for v,q in diagnosis['results'].items():
        if q['saved_forward_maximum'] or q['actor_loss_error'] or q['critic_loss_error'] or any(q['saved_gradient_replay_error'].values()):raise ValueError('saved GPU computation not reproduced exactly')
    DIAG=torch.load(a.diagnosis/'replayed_preactivations.pt',map_location='cpu',weights_only=False)
    if (root/'gradient_audit.json').exists():raise ValueError('preserve previous audit')
    pack=torch.load(root/'update_packet.pt',map_location='cpu',weights_only=False);panel=Path(pack['panel']);prev=Path(pack['previous'])
    if sha(prev)!=pack['previous_sha256'] or any(sha(panel/f)!=h for f,h in pack['panel_sha256'].items()):raise ValueError('fixed behavior/physical source changed')
    before=torch.load(prev,map_location='cpu',weights_only=False);after=torch.load(root/'policy_heads.pt',map_location='cpu',weights_only=False)
    trace=torch.load(panel/'trace.pt',map_location='cpu',weights_only=False);initial=torch.load(panel/'initial.pt',map_location='cpu',weights_only=False);rows=json.loads((panel/'rows.json').read_text())
    reward=np.array([r['physical105'] for r in rows],dtype=np.float64)
    if pack['seed']!=546+pack['update'] or after['updates']!=pack['update'] or before['updates']!=pack['update']-1 or len(rows)!=768:raise ValueError('fixed update identity')
    maximum=dict(input=0.,advantage=0.,forward=0.,loss=0.,gradient=0.,adam=0.,norm_relative=0.)
    for group,variant in enumerate(('cm','state_only','none'),1):
        VARIANT=variant
        rec=pack['records'][variant];ids=np.flatnonzero(initial['policy_group'].numpy()==group);ix=rec['indices'].numpy();n=len(ix)
        if n!=1024:raise ValueError('fixed first complete minibatch')
        source={k:trace[k][:,ids].reshape(-1,trace[k].shape[-1]).numpy() for k in ('normalized_context','request','executed_action12','physical_transition')}
        for source_key,packet_key in (('normalized_context','input'),('request','request'),('executed_action12','action'),('physical_transition','transition')):
            maximum['input']=max(maximum['input'],float(np.max(np.abs(source[source_key][ix]-rec[packet_key].numpy()))))
        returns=np.broadcast_to(reward[ids],(202,192)).ravel();oldv=trace['critic_value'][:,ids].numpy().ravel().astype(np.float64)
        advantage=returns-oldv;mean=advantage.mean();std=advantage.std();normalized=(advantage-mean)/(std+1e-8)
        maximum['advantage']=max(maximum['advantage'],float(np.max(np.abs(normalized[ix]-array(rec['advantage'])))))
        if not np.array_equal(returns[ix],rec['return_target'].numpy()) or not np.array_equal(trace['request_logprob'][:,ids].numpy().ravel()[ix],rec['old_logp'].numpy()):raise ValueError('actual terminal physical return / frozen behavior likelihood')
        for kind in ('actor','critic'):
            if any(not torch.equal(v,before['variants'][variant][kind][k]) for k,v in rec[kind+'_before'].items()):raise ValueError('first minibatch initial weights')
        aw=rec['actor_before'];cw=rec['critic_before'];x=array(rec['input']);request=array(rec['request']);adv=array(rec['advantage']);oldlp=array(rec['old_logp'])
        mu,aa=mlp(x,aw,['network.0','network.2','network.4']);ls=np.clip(array(aw['log_std']),-5,0);iv=np.exp(-2*ls);diff=request-mu
        lp=(-.5*diff**2*iv-ls-.5*np.log(2*np.pi)).sum(-1);ratio=np.exp(lp-oldlp)
        active=((adv>=0)&(ratio<=1.2))|((adv<0)&(ratio>=.8));dlp=-adv*ratio*active/n
        ag,_=backward(dlp[:,None]*diff*iv,aw,['network.0','network.2','network.4'],aa)
        ag['log_std']=((dlp[:,None]*(diff**2*iv-1)).sum(0)-.001)*((array(aw['log_std'])>=-5)&(array(aw['log_std'])<=0))
        al=-np.minimum(ratio*adv,np.clip(ratio,.8,1.2)*adv).mean()-.001*(ls+.5*np.log(2*np.pi*np.e)).sum()
        # Encoder's last layer also has ReLU, unlike generic final MLP output.
        latent,ca=mlp(x,cw,['encoder.0','encoder.2']);latent=np.maximum(latent,0)
        value=latent@array(cw['value.weight']).T+array(cw['value.bias']);value=value.ravel()
        action=array(rec['action']) if variant=='cm' else np.zeros((n,12));din=np.concatenate((latent,action),-1)
        pred,da=mlp(din,cw,['dynamics.0','dynamics.2']);target=array(rec['transition']);ret=array(rec['return_target']);weight=0 if variant=='none' else .05
        dv=(value-ret)/n;dp=2*weight*(pred-target)/(n*6)
        cg,dl=backward(dp,cw,['dynamics.0','dynamics.2'],da)
        cg['value.weight']=dv[None]@latent;cg['value.bias']=np.array([dv.sum()]);dl=dl[:,:64]+dv[:,None]*array(cw['value.weight'])
        eg,_=backward(dl*(ca[-1]>0),cw,['encoder.0','encoder.2'],ca);cg.update(eg)
        cl=.5*np.mean((value-ret)**2)+weight*np.mean((pred-target)**2)
        maximum['forward']=max(maximum['forward'],float(np.abs(mu-array(rec['mean'])).max()),float(np.abs(value-array(rec['value'])).max()),float(np.abs(pred-array(rec['prediction'])).max()))
        maximum['loss']=max(maximum['loss'],abs(al-rec['actor_loss']),abs(cl-rec['critic_loss']))
        gradients={**ag,**{'critic.'+k:v for k,v in cg.items()}}
        norm=np.sqrt(sum(np.sum(v*v) for v in gradients.values()));maximum['norm_relative']=max(maximum['norm_relative'],abs(norm-rec['gradient_norm'])/(1+norm))
        opt=rec['optimizer_before'];params=opt['param_groups'][0]['params'];names=list(aw)+['critic.'+k for k in cw]
        if set(gradients)!=set(names) or len(params)!=len(names):raise ValueError('all policy and critic gradients')
        factor=min(1.,.5/(rec['gradient_norm']+1e-6))
        for name,param in zip(names,params):
            grad=array(rec['gradient'][name]);maximum['gradient']=max(maximum['gradient'],float(np.max(np.abs(grad-gradients[name]))))
            state=opt['state'].get(param,{});m=array(state['exp_avg']) if state else np.zeros_like(grad);v=array(state['exp_avg_sq']) if state else np.zeros_like(grad);t=float(state['step'])+1 if state else 1
            g=grad*factor;m=.9*m+.1*g;v=.999*v+.001*g*g
            kind='critic' if name.startswith('critic.') else 'actor';key=name[7:] if kind=='critic' else name
            expected=array(rec[kind+'_before'][key])-3e-4*m/(1-.9**t)/(np.sqrt(v)/np.sqrt(1-.999**t)+1e-8)
            maximum['adam']=max(maximum['adam'],float(np.abs(expected-array(rec[kind+'_after'][key])).max()))
        summary=json.loads((root/'results.json').read_text())['variants'][variant]
        if summary['epochs']!=4 or summary['minibatches']!=152 or summary['episodes']!=192 or summary['transitions']!=38784:raise ValueError('fixed full training schedule')
    limits=dict(input=0.,advantage=2e-5,forward=2e-5,loss=2e-5,gradient=2e-5,adam=2e-5,norm_relative=2e-6)
    if any(maximum[k]>limits[k] for k in maximum):raise ValueError(('independent PPO / auxiliary / Adam drift',maximum))
    report=dict(run_status='COMPLETED',update=pack['update'],seed=pack['seed'],maximum_error=maximum,scope='first predetermined minibatch per variant per panel; remaining steps NOT independently reconstructed',all_actor_and_critic_first_step_gradients=True,first_step_adam=True,actual_physical_returns=True,source_sha256={str(root/f):sha(root/f) for f in ('update_packet.pt','policy_heads.pt','results.json')},wall_seconds=time.monotonic()-begin)
    if len(BRANCH_CERTIFICATES)!=1 or BRANCH_CERTIFICATES[0]['variant']!='none' or pack['update']!=17:raise ValueError('only diagnosed u17 single branch correction')
    report.update(original_audit_retained_failed=True,scalar_tolerances_unchanged=limits,branch_certificates=BRANCH_CERTIFICATES,gpu_saved_gradients_exactly_reproduced=True,no_optimizer_or_physics_updates=True,correction_script_sha256=sha(Path(__file__)),diagnosis_sha256={str(a.diagnosis/f):sha(a.diagnosis/f) for f in ['results.json','replayed_preactivations.pt']})
    a.output.mkdir();(a.output/'gradient_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
