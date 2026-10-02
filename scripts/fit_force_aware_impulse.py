"""Matched physical residual forecasts; fresh fixed native test and cluster gates."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from src.task.CmResidual.force_aware_impulse import SCHEMA,impulse_target,force_prior,initialized_model

def read_panel(path,sources):
    r=json.loads((path/'results.json').read_text());audit=json.loads((path/'panel_audit.json').read_text());assert r['run_status']==audit['run_status']=='COMPLETED' and not r['deterministic']
    for f in ['initial.pt','trace.pt','results.json','panel_audit.json','physical_metadata.json']:sources[str((path/f).resolve())]=sha(path/f)
    assert sha(path/'initial.pt')==r['initial_sha256'] and sha(path/'trace.pt')==r['trace_sha256']
    i=torch.load(path/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(path/'trace.pt',map_location='cpu',weights_only=False)
    phy=json.loads((path/'physical_metadata.json').read_text());g=np.array(phy['gravity']);mass=np.array([q['mass'] for q in phy['object_body_properties']]);weight=mass*np.linalg.norm(g)
    assert len(mass)==768 and (mass>0).all() and np.max(np.abs(g-[0,0,-9.81]))<1e-5
    group=i['policy_group'].numpy();motion=i['motion'].numpy();stop=i['phase_stop'].numpy()[motion];lift=i['lift_start'].numpy()[motion]
    pre=t['context'].numpy()[...,36:49];post=t['object_root'].numpy();progress=t['progress'].numpy()
    clear=np.concatenate([np.full((1,768),np.nan),t['clearance'].numpy()[:-1]],0)
    eligible=(progress>=stop[None]-74)&(progress<=stop[None]+30)&(pre[...,2]-i['initial_height'].numpy()[None]>=.03)&(clear>=.02)&(group[None]>0)
    eligible[0]=False;tick,env=np.where(eligible);v=pre[tick,env,7:10].astype(np.float64)
    objforce=t['object_force'].numpy()[tick-1,env].astype(np.float64);handforce=t['hand_force'].numpy()[tick-1,env].reshape(-1,15).astype(np.float64)
    forces=np.concatenate([objforce,handforce],-1)/weight[env,None];prior=force_prior(objforce,v,mass[env],g)
    label=impulse_target(v,post[tick,env,7:10].astype(np.float64),g,1/30)
    phase=np.where(progress[tick,env]<lift[env],0,np.where(progress[tick,env]<stop[env]-74,1,np.where(progress[tick,env]<=stop[env],2,3)));cells=motion[env]*4+phase
    slots=np.full(768,-1,int)
    for arm in (1,2,3):
        ids=np.flatnonzero(group==arm);assert len(ids)==192;slots[ids]=np.arange(192)
    full=[t['request_noise'][:,np.flatnonzero(group==arm)] for arm in (1,2,3)];assert torch.equal(full[0],full[1]) and torch.equal(full[0],full[2])
    result=dict(state=t['normalized_context'].numpy()[tick,env],extra=np.concatenate([forces,prior],-1).astype(np.float32),prior=prior.astype(np.float32),action=t['executed_action12'].numpy()[tick,env],y=label.astype(np.float32),cells=cells,cluster=slots[env],tick=tick,environment=env,policy_group=group[env],eligible_episodes=len(np.unique(env)),eligible_transitions=len(env))
    assert all(np.isfinite(result[k]).all() for k in ['state','extra','prior','action','y'])
    return result

def features(data,extra_mean,extra_std,templates,variant):
    state=templates[data['cells']] if variant=='global_action' else data['state']
    extra=np.clip((data['extra']-extra_mean)/extra_std,-10,10);action=np.zeros_like(data['action']) if variant=='state_only' else data['action']
    return np.concatenate([state,extra,action],-1).astype(np.float32)

def numpy_model(x,state):
    h=x.astype(np.float64)
    for layer in (0,2,4):
        h=h@state[f'{layer}.weight'].numpy().astype(np.float64).T+state[f'{layer}.bias'].numpy().astype(np.float64)
        if layer!=4:h=np.maximum(h,0)
    return h

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();assert ROOT in out.parents and not out.exists();out.mkdir();torch.set_num_threads(2);assert torch.cuda.is_available()
    begin=time.monotonic();sources={};parts=[]
    for f in [Path(__file__),ROOT/'src/task/CmResidual/force_aware_impulse.py',ROOT/'docs/experiments/probes/P-20261002-force-aware-impulse.md']:sources[str(f)]=sha(f)
    for seed in range(547,567):
        d=read_panel(a.source/f's{seed}',sources);assert json.loads((a.source/f's{seed}/results.json').read_text())['continuous_updates']==seed-547;parts.append(d);print(json.dumps(dict(fit_seed=seed,eligible=d['eligible_transitions'])),flush=True)
    fit={key:np.concatenate([d[key] for d in parts]) for key in ['state','extra','prior','action','y','cells']};del parts
    test=read_panel(a.test/'s575',sources);assert json.loads((a.test/'s575/results.json').read_text())['continuous_updates']==20
    if len(fit['y'])<10000 or len(test['y'])<1024 or test['eligible_episodes']<32:
        result=dict(run_status='COMPLETED',label='UNCLEAR',reason='fixed current-state eligibility minimum not met',fit_count=len(fit['y']),test_count=len(test['y']),test_episodes=test['eligible_episodes'],no_optimizer_steps=True,source_sha256=sources,wall_seconds=time.monotonic()-begin)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');return
    extra_mean=fit['extra'].mean(0);extra_std=fit['extra'].std(0).clip(.001);residual=fit['y']-fit['prior'];ymean=residual.mean(0);ystd=residual.std(0).clip(.001)
    templates=np.zeros((12,70),np.float32);counts=np.bincount(fit['cells'],minlength=12)
    for cell in range(12):
        if counts[cell]:templates[cell]=fit['state'][fit['cells']==cell].mean(0)
    target=torch.from_numpy((residual-ymean)/ystd).cuda();models={};predictions={};audit_error={};last_loss={};saved_features={}
    for variant in ['cm','state_only','global_action']:
        x=features(fit,extra_mean,extra_std,templates,variant);tx=torch.from_numpy(x).cuda();model=initialized_model().cuda();opt=torch.optim.Adam(model.parameters(),lr=3e-4);gen=torch.Generator(device='cuda').manual_seed(2192)
        for step in range(2000):
            if time.monotonic()-begin>1650:raise TimeoutError('fixed full fit budget')
            ids=torch.randint(len(tx),(4096,),device='cuda',generator=gen);loss=(model(tx[ids])-target[ids]).square().mean();assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm);opt.step()
            if step%500==499:print(json.dumps(dict(variant=variant,step=step+1,normalized_residual_loss=float(loss.detach()))),flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()};models[variant]=state;last_loss[variant]=float(loss.detach());z=features(test,extra_mean,extra_std,templates,variant);saved_features[variant]=torch.from_numpy(z)
        with torch.no_grad():standard=np.concatenate([model(torch.from_numpy(block).cuda()).cpu().numpy() for block in np.array_split(z,max(1,(len(z)+4095)//4096))])
        predictions[variant]=test['prior']+ymean+ystd*standard
        sample=np.arange(0,len(z),max(1,len(z)//512))[:512];audit_error[variant]=float(np.abs(numpy_model(z[sample],state)-standard[sample]).max());assert audit_error[variant]<2e-5
        del tx,model,opt,x;torch.cuda.empty_cache()
    predictions['physical_prior']=test['prior'];names=['cm','state_only','global_action','physical_prior'];episode=np.zeros((192,5),np.float64)
    np.add.at(episode[:,0],test['cluster'],1)
    for j,name in enumerate(names,1):
        error=(predictions[name].astype(np.float64)-test['y'].astype(np.float64))**2;row=error.mean(1);np.add.at(episode[:,j],test['cluster'],row)
    rates=episode[:,1:].sum(0)/episode[:,0].sum();rng=np.random.RandomState(2193);boot=episode[rng.randint(0,192,size=(2000,192))].sum(1);assert (boot[:,0]>0).all();values=boot[:,1:]/boot[:,0,None];cis={name:np.quantile(values[:,0]-values[:,index],[.025,.975]).tolist() for index,name in enumerate(names[1:],1)}
    gates={name:dict(mse_gain1percent=bool(rates[0]<=.99*rates[index]),paired_upper95_negative=bool(cis[name][1]<0)) for index,name in enumerate(names[1:],1)}
    cp=dict(schema=SCHEMA,models=models,extra_mean=torch.from_numpy(extra_mean),extra_std=torch.from_numpy(extra_std),residual_mean=torch.from_numpy(ymean),residual_std=torch.from_numpy(ystd),global_templates=torch.from_numpy(templates),fit_cell_counts=counts.tolist(),updates_each=2000,fit_only_statistics=True,source_sha256=sources)
    torch.save(cp,out/'impulse_models.pt');torch.save(dict(data={k:torch.from_numpy(v) for k,v in test.items() if isinstance(v,np.ndarray)},features=saved_features,predictions={k:torch.from_numpy(v) for k,v in predictions.items()}),out/'test_predictions.pt')
    (out/'episode_errors.json').write_text(json.dumps(episode.tolist())+'\n')
    for filename,h in sources.items():assert sha(Path(filename))==h,filename
    result=dict(run_status='COMPLETED',label='PROMISING' if all(all(q.values()) for q in gates.values()) else 'UNPROMISING',mse=dict(zip(names,map(float,rates))),paired_difference_interval95=cis,gates=gates,fit_transitions=len(fit['y']),test_transitions=len(test['y']),test_episodes=test['eligible_episodes'],common_stream_clusters=192,last_training_loss=last_loss,independent_numpy_standard_forward_max_error=audit_error,source_sha256=sources,model_sha256=sha(out/'impulse_models.pt'),predictions_sha256=sha(out/'test_predictions.pt'),wall_seconds=time.monotonic()-begin,no_policy_updates=True,cm_utility_unproved=True,formal_validation=False,not_exact_pairwise_contact_impulse=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

if __name__=='__main__':main()
