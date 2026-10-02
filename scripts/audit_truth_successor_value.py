"""Independent raw current/successor task state, all model forwards and risk gates."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha
import numpy as np
import torch

def decode(d,base):
    i=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);n=lambda k:i[k].numpy();v=lambda k:t[k].numpy();arm=n('policy_group');motion=n('motion');stops=n('phase_stop')[motion]
    q=np.concatenate((n('base_q')[None],v('native_q')[:-1]),0);dq=np.concatenate((n('initial_dof_vel')[None],v('native_dq')[:-1]),0);obj=np.concatenate((n('object_root')[None],v('object_root')[:-1]),0);pair=np.concatenate((n('initial_contact')[None],v('contact')[:-1]),0)
    clock=np.broadcast_to(np.arange(202)[:,None],(202,768));planned=np.minimum(clock+1,stops[None]);ref=n('native_reference_q')[motion[None],planned]
    raw=np.concatenate((q,dq,obj,pair,ref,(planned/stops[None])[...,None]),-1).astype(np.float32)
    if not np.allclose(raw,v('context'),atol=1e-6,rtol=0):raise ValueError('independent actual PREcontext')
    z=np.clip((raw-base['observation_mean'].numpy())/base['observation_std'].numpy(),-10,10)
    if not np.allclose(z,v('normalized_context'),atol=2e-6,rtol=0):raise ValueError('independent P0 normalization')
    window=(v('progress')>=stops[None]-74)&(v('progress')<=stops[None]+30);valid=(v('object_root')[...,2]-n('initial_height')[None]>=np.float32(.03))&(v('clearance')>=np.float32(.02));bad=window&~valid
    y=np.array([r['physical105'] for r in json.loads((d/'rows.json').read_text())],dtype=np.float32);assert np.array_equal(y.astype(bool),~bad.any(0)) and np.all(window.sum(0)==105)
    post=np.cumsum(bad,0)>0;pre=np.concatenate((np.zeros_like(post[:1]),post[:-1]),0);end=stops+30;current_done=pre|(clock>=end[None]);next_done=post|(clock+1>=end[None]);take=(arm[None]>0)&~current_done&(clock<201);ticks,envs=np.nonzero(take)
    current=np.concatenate((z[ticks,envs],(~pre[ticks,envs]).astype(np.float32)[:,None],((end[envs]-ticks).clip(0).astype(np.float32)/np.float32(202))[:,None]),-1)
    following=np.concatenate((z[ticks+1,envs],(~post[ticks,envs]).astype(np.float32)[:,None],((end[envs]-ticks-1).clip(0).astype(np.float32)/np.float32(202))[:,None]),-1)
    independent=np.array([0,1,2,3,4,5,6,8,10,12,14,15]);command=(v('target')[:,:,independent]-q[:,:,independent])/np.abs(n('pd_scale')[independent])
    if not np.allclose(command[:,arm>0],v('executed_action12')[:,arm>0],atol=1e-5,rtol=0):raise ValueError('causal executed PD command')
    cluster=np.full(768,-1,dtype=np.int64)
    for g in (1,2,3):cluster[np.flatnonzero(arm==g)]=np.arange(192)
    return dict(current=current,next=following,command=v('executed_action12')[ticks,envs],target=y[envs],known=next_done[ticks,envs],known_value=(~post[ticks,envs]).astype(np.float32),tick=ticks,environment=envs,motion=motion[envs],timebin=np.minimum(ticks*16//202,15),cluster=cluster[envs])

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);root=a.directory.resolve();m=json.loads((root/'run_manifest.json').read_text());r=json.loads((root/'fit/results.json').read_text());assert r['run_status']=='COMPLETED'
    model=root/'fit/models.pt';prediction=root/'fit/predictions.pt';assert sha(model)==r['model_sha256'] and sha(prediction)==r['prediction_sha256'];c=torch.load(model,map_location='cpu',weights_only=False);p=torch.load(prediction,map_location='cpu',weights_only=False);base=torch.load(Path(m['base_checkpoint']),map_location='cpu',weights_only=False)
    fit=decode(root/'s547',base);test=decode(root/'s595',base)
    for k in ('target','cluster','tick','environment'):
        if not np.array_equal(test[k],p[k].numpy()):raise ValueError('full causal eligible row/target '+k)
    if not np.array_equal(test['known'],p['next_known'].numpy()):raise ValueError('PRE vs NEXT history')
    table=np.full((3,16),np.float32(fit['target'].mean()),dtype=np.float32)
    for mo in range(3):
        for b in range(16):
            select=(fit['motion']==mo)&(fit['timebin']==b)
            if select.any():table[mo,b]=np.float32(fit['target'][select].sum()/len(fit['target'][select]))
    if not np.array_equal(table,c['phase_template'].numpy()):raise ValueError('FIT-only strongcontrol')
    maxerr={}
    for variant,state in c['parameters'].items():
        x=test['next'] if variant=='oracle_successor' else test['current'];action=test['command'] if variant=='direct_q' else np.zeros_like(test['command']);z=np.concatenate((x,action),-1).astype(np.float64)
        for layer in ('0','2'):z=np.maximum(z@state[layer+'.weight'].numpy().astype(np.float64).T+state[layer+'.bias'].numpy().astype(np.float64),0)
        z=z@state['4.weight'].numpy().astype(np.float64).T+state['4.bias'].numpy().astype(np.float64);v=(1/(1+np.exp(-z.clip(-700,700)))).ravel()
        if variant=='oracle_successor':v=np.where(test['known'],test['known_value'],v)
        maxerr[variant]=float(np.abs(v-p['predictions'][variant].numpy()).max())
        if maxerr[variant]>2e-5:raise ValueError(('full independent value forward',variant,maxerr[variant]))
    assert np.array_equal(table[test['motion'],test['timebin']],p['predictions']['motion_time'].numpy())
    risk={k:(value.double().numpy()-test['target'].astype(np.float64))**2 for k,value in p['predictions'].items()};brier={k:float(v.mean()) for k,v in risk.items()};rng=np.random.default_rng(3533);idx=rng.integers(0,192,size=(2000,192));weights=np.bincount(test['cluster'],minlength=192);gates={};interval={}
    for control in ('current_value','direct_q','motion_time'):
        diff=np.bincount(test['cluster'],weights=risk['oracle_successor']-risk[control],minlength=192);ci=np.quantile(diff[idx].sum(1)/weights[idx].sum(1),[.025,.975]).tolist();interval[control]=ci;gates[control]=dict(gain1percent=brier['oracle_successor']<=.99*brier[control],paired_upper95_negative=ci[1]<0)
        if abs(brier[control]-r['brier'][control])>1e-12 or not np.allclose(ci,r['paired_difference_interval95'][control],atol=1e-12,rtol=0):raise ValueError('paired risk/statistics')
    assert gates==r['gates'];label='PROMISING' if all(all(g.values()) for g in gates.values()) else 'UNPROMISING';assert label==r['label']
    result=dict(run_status='COMPLETED',label=label,all_test_current_and_true_successor_task_states_rebuilt=True,all_post_boundary_labels_causal=True,full_model_numpy_forward_maximum=maxerr,fit_template_exact=True,all_clustered_risk_gates_rebuilt=True,optimizer_not_replayed=True,privileged_oracle_not_deployable=True,source_result_sha256=sha(root/'fit/results.json'),script_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-start)
    (root/'value_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
