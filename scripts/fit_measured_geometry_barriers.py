"""Fixed actual-geometry barrier prediction screen and strong matched controls."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_static_hold_feasibility import rotation
from scripts.fit_force_aware_impulse import numpy_model
from src.task.CmResidual.measured_geometry_barriers import SCHEMA,relative_geometry,initialized_model
VARIANTS=['cm','state_only','geometry_off_action','global_action']

def read_panel(path,sources):
    r=json.loads((path/'results.json').read_text());audit=json.loads((path/'panel_audit.json').read_text());assert r['run_status']==audit['run_status']=='COMPLETED' and not r['deterministic'] and r['continuous_updates']==20
    for f in ['initial.pt','trace.pt','results.json','panel_audit.json','physical_metadata.json']:sources[str((path/f).resolve())]=sha(path/f)
    assert sha(path/'initial.pt')==r['initial_sha256'] and sha(path/'trace.pt')==r['trace_sha256']
    i=torch.load(path/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(path/'trace.pt',map_location='cpu',weights_only=False);meta=json.loads((path/'physical_metadata.json').read_text())
    names=[meta['native_body_names'][k] for k in meta['contact_body_ids']];assert names==['index_intermediate','middle_intermediate','pinky_intermediate','ring_intermediate','thumb_distal']
    group=i['policy_group'].numpy();motion=i['motion'].numpy();stop=i['phase_stop'].numpy()[motion];lift=i['lift_start'].numpy()[motion];progress=t['progress'].numpy()
    mask=(progress>=stop[None]-74)&(progress<=stop[None]+30)&(group[None]>0);mask[:2]=False;tick,env=np.where(mask)
    # Context BEFORE action must be previous native POST root. No post-action poses.
    root=t['object_root'].numpy();pre=root[tick-1,env].astype(np.float64);older=root[tick-2,env].astype(np.float64);assert np.max(np.abs(pre-t['context'].numpy()[tick,env,36:49]))<1e-7
    pos,rot=relative_geometry(pre,t['hand_body_position'].numpy()[tick-1,env],t['hand_body_quaternion'].numpy()[tick-1,env]);previous,_=relative_geometry(older,t['hand_body_position'].numpy()[tick-2,env],t['hand_body_quaternion'].numpy()[tick-2,env])
    force=np.concatenate([t['object_force'].numpy()[tick-1,env,None],t['hand_force'].numpy()[tick-1,env]],axis=1).astype(np.float64);mass=np.array([p['mass'] for p in meta['object_body_properties']]);weight=mass*np.linalg.norm(meta['gravity']);assert (weight>0).all()
    force=np.einsum('nij,nbj->nbi',rotation(pre[:,3:7]).swapaxes(-1,-2),force)/weight[env,None,None]
    clr=t['clearance'].numpy();prior=np.stack([pre[:,2]-older[:,2],clr[tick-1,env]-clr[tick-2,env]],-1)/.005
    y=np.stack([root[tick,env,2]-pre[:,2],clr[tick,env]-clr[tick-1,env]],-1)/.005
    extra=np.concatenate([pos.reshape(-1,15),rot,force.reshape(-1,18),prior,(pos-previous).reshape(-1,15)],-1).astype(np.float32);assert extra.shape[-1]==80
    phase=np.where(progress[tick,env]<lift[env],0,np.where(progress[tick,env]<stop[env]-74,1,np.where(progress[tick,env]<=stop[env],2,3)));cells=motion[env]*4+phase
    slots=np.full(768,-1,int)
    for arm in (1,2,3):slots[np.flatnonzero(group==arm)]=np.arange(192)
    full=[t['request_noise'][:,np.flatnonzero(group==arm)] for arm in (1,2,3)];assert torch.equal(full[0],full[1]) and torch.equal(full[0],full[2])
    result=dict(state=t['normalized_context'].numpy()[tick,env],extra=extra,prior=prior.astype(np.float32),action=t['executed_action12'].numpy()[tick,env],y=y.astype(np.float32),cells=cells,cluster=slots[env],tick=tick,environment=env,policy_group=group[env],motion=motion[env],eligible_episodes=len(np.unique(env)))
    assert all(np.isfinite(result[k]).all() for k in ['state','extra','prior','action','y'])
    return result

def features(data,mean,std,templates,extra_templates,variant):
    state=templates[data['cells']] if variant=='global_action' else data['state'];extra=data['extra'].copy()
    if variant=='geometry_off_action':
        for segment in [slice(0,45),slice(65,80)]:extra[:,segment]=extra_templates[data['cells'],segment]
    action=np.zeros_like(data['action']) if variant=='state_only' else data['action']
    return np.concatenate([state,np.clip((extra-mean)/std,-10,10),action],-1).astype(np.float32)

def scores(data,predictions):
    names=VARIANTS+['physical_persistence'];ep=np.zeros((192,6),np.float64);np.add.at(ep[:,0],data['cluster'],1)
    for j,name in enumerate(names,1):np.add.at(ep[:,j],data['cluster'],((predictions[name].astype(np.float64)-data['y'])**2).mean(1))
    rates=ep[:,1:].sum(0)/ep[:,0].sum();rng=np.random.RandomState(3213);b=ep[rng.randint(0,192,(2000,192))].sum(1);values=b[:,1:]/b[:,0,None];ci={name:np.quantile(values[:,0]-values[:,j],[.025,.975]).tolist() for j,name in enumerate(names[1:],1)}
    gates={name:dict(gain1percent=bool(rates[0]<=.99*rates[j]),paired_upper95_negative=bool(ci[name][1]<0)) for j,name in enumerate(names[1:],1)}
    return rates,ci,gates,ep

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();assert ROOT in out.parents and not out.exists();out.mkdir();torch.set_num_threads(2);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    begin=time.monotonic();sources={}
    for f in [Path(__file__),ROOT/'src/task/CmResidual/measured_geometry_barriers.py',ROOT/'docs/experiments/probes/P-20261002-measured-geometry-barriers.md']:sources[str(f)]=sha(f)
    parts=[read_panel(a.source/f's{seed}',sources) for seed in (578,579)];fit={k:np.concatenate([d[k] for d in parts]) for k in ['state','extra','prior','action','y','cells']};del parts;test=read_panel(a.source/'s580',sources)
    mean=fit['extra'].mean(0);std=fit['extra'].std(0).clip(.001);residual=fit['y']-fit['prior'];ym=residual.mean(0);ys=residual.std(0).clip(.001);templates=np.zeros((12,70),np.float32);et=np.zeros((12,80),np.float32);counts=np.bincount(fit['cells'],minlength=12)
    for cell in range(12):
        if counts[cell]:templates[cell]=fit['state'][fit['cells']==cell].mean(0);et[cell]=fit['extra'][fit['cells']==cell].mean(0)
    if len(fit['y'])<10000 or len(test['y'])<1024 or test['eligible_episodes']<32:
        (out/'results.json').write_text(json.dumps(dict(run_status='COMPLETED',label='UNCLEAR',reason='predeclared eligibility minimum',no_optimizer_steps=True,source_sha256=sources),indent=2)+'\n');return
    target=torch.from_numpy((residual-ym)/ys).cuda();models={};predictions={};saved={};errors={};last_loss={}
    for variant in VARIANTS:
        tx=torch.from_numpy(features(fit,mean,std,templates,et,variant)).cuda();model=initialized_model().cuda();opt=torch.optim.Adam(model.parameters(),lr=3e-4);gen=torch.Generator(device='cuda').manual_seed(3212)
        for step in range(1500):
            if time.monotonic()-begin>400:raise TimeoutError('fixed geometry fit budget')
            ids=torch.randint(len(tx),(4096,),device='cuda',generator=gen);loss=(model(tx[ids])-target[ids]).square().mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm);opt.step()
            if step%500==499:print(json.dumps(dict(variant=variant,updates=step+1,loss=float(loss.detach()))),flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()};models[variant]=state;last_loss[variant]=float(loss.detach());z=features(test,mean,std,templates,et,variant);saved[variant]=torch.from_numpy(z)
        with torch.no_grad():pred=np.concatenate([model(torch.from_numpy(x).cuda()).cpu().numpy() for x in np.array_split(z,16)])
        sample=np.linspace(0,len(z)-1,512).astype(int);errors[variant]=float(np.abs(numpy_model(z[sample],state)-pred[sample]).max());assert errors[variant]<2e-5
        predictions[variant]=test['prior']+ym+ys*pred;del tx,model,opt;torch.cuda.empty_cache()
    predictions['physical_persistence']=test['prior'];rates,ci,gates,ep=scores(test,predictions)
    torch.save(dict(schema=SCHEMA,models=models,extra_mean=torch.from_numpy(mean),extra_std=torch.from_numpy(std),residual_mean=torch.from_numpy(ym),residual_std=torch.from_numpy(ys),global_templates=torch.from_numpy(templates),extra_templates=torch.from_numpy(et),fit_cell_counts=counts.tolist(),updates_each=1500,fit_only_statistics=True,source_sha256=sources),out/'barrier_models.pt')
    torch.save(dict(data={k:torch.from_numpy(v) for k,v in test.items() if isinstance(v,np.ndarray)},features=saved,predictions={k:torch.from_numpy(v) for k,v in predictions.items()}),out/'test_predictions.pt');(out/'episode_errors.json').write_text(json.dumps(ep.tolist())+'\n')
    for f,h in sources.items():assert sha(Path(f))==h,f
    result=dict(run_status='COMPLETED',label='PROMISING' if all(all(g.values()) for g in gates.values()) else 'UNPROMISING',mse=dict(zip(VARIANTS+['physical_persistence'],map(float,rates))),paired_difference_interval95=ci,gates=gates,fit_transitions=len(fit['y']),test_transitions=len(test['y']),test_episodes=test['eligible_episodes'],common_noise_clusters=192,source_sha256=sources,model_sha256=sha(out/'barrier_models.pt'),predictions_sha256=sha(out/'test_predictions.pt'),sample_numpy_standard_forward_max_error=errors,last_training_loss=last_loss,wall_seconds=time.monotonic()-begin,no_policy_updates=True,cm_utility_unproved=True,formal_validation=False,actual_sdk_measured_poses=True,not_contact_points=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)

if __name__=='__main__':main()
