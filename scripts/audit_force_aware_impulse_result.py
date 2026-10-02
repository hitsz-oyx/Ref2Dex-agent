"""Independent raw-time/units/FIT-stats/full-forward/cluster/gate audit.

Before observing fitted results: labels/prior obey FP32 rounding enclosures;
normalized network forward allowance2e-5*(1+abs(output)); physical-output
rounding adds3u*(abs(prior)+abs(mean)+abs(std*output)), u=2^-24.
These are this NEW forecast's unit-aware checks, not altered native tolerances.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch,yaml
from scripts.run_contact_response_probe import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.directory.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists();out.mkdir();begin=time.monotonic();torch.set_num_threads(2);u=2**-24
    m=json.loads((run/'run_manifest.json').read_text());r=json.loads((run/'fit/results.json').read_text());assert m['run_status']==r['run_status']=='COMPLETED'
    for f,h in {**m['input_sha256'],**r['source_sha256']}.items():assert sha(Path(f))==h,f
    if r['label']=='UNCLEAR':
        (out/'results.json').write_text(json.dumps(dict(run_status='COMPLETED',label='UNCLEAR',reason='original fixed eligibility stop retained; no model/gain claim'))+'\n');return
    assert sha(run/'fit/impulse_models.pt')==r['model_sha256'] and sha(run/'fit/test_predictions.pt')==r['predictions_sha256']
    cp=torch.load(run/'fit/impulse_models.pt',map_location='cpu',weights_only=False);packet=torch.load(run/'fit/test_predictions.pt',map_location='cpu',weights_only=False);assert cp['updates_each']==2000
    cmd=m['phases'][0]['command'];cfgpath=Path(cmd[cmd.index('--cfg_env')+1]);cfg=yaml.safe_load(cfgpath.read_text());assert cfg['env']['controlFrequencyInv']==2 and 'dt' not in cfg['sim']
    configpy=ROOT/'third_party/DExplore/dexplore/utils/config.py';basepy=ROOT/'third_party/DExplore/dexplore/env/tasks/base_task.py';taskpy=ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py'
    assert 'SIM_TIMESTEP = 1.0 / 60.0' in configpy.read_text() and 'sim_params.dt = SIM_TIMESTEP' in configpy.read_text() and 'for i in range(self.control_freq_inv):' in basepy.read_text() and 'asset_options.linear_damping = 0.01' in taskpy.read_text()
    def raw(path):
        i=torch.load(path/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(path/'trace.pt',map_location='cpu',weights_only=False);p=json.loads((path/'physical_metadata.json').read_text())
        gravity=float(-p['gravity'][2]);assert abs(gravity-9.81)<1e-5 and p['gravity'][:2]==[0.,0.];mass=np.array([v['mass'] for v in p['object_body_properties']]);assert (mass>0).all()
        post=t['object_root'].numpy();pre=np.concatenate([i['object_root'].numpy()[None],post[:-1]],0);group=i['policy_group'].numpy();mo=i['motion'].numpy();stop=i['phase_stop'].numpy()[mo];lift=i['lift_start'].numpy()[mo];prog=t['progress'].numpy()
        clearance=np.concatenate([np.full((1,768),np.nan),t['clearance'].numpy()[:-1]],0);mask=(prog>=stop[None]-74)&(prog<=stop[None]+30)&(pre[...,2]-i['initial_height'].numpy()[None]>=np.float32(.03))&(clearance>=.02)&(group[None]>0);mask[0]=False;tick,env=np.where(mask)
        current_v=pre[tick,env,7:10].astype(np.float64);next_v=post[tick,env,7:10].astype(np.float64);label=(next_v-current_v)/(gravity*(2/60));label[:,2]+=1
        of=t['object_force'].numpy()[tick-1,env].astype(np.float64);hf=t['hand_force'].numpy()[tick-1,env].astype(np.float64).reshape(-1,15)
        normalized_force=np.concatenate([of,hf],-1)/(mass[env,None]*gravity);prior=of/(mass[env,None]*gravity)-.01*current_v/gravity
        cells=mo[env]*4+np.where(prog[tick,env]<lift[env],0,np.where(prog[tick,env]<stop[env]-74,1,np.where(prog[tick,env]<=stop[env],2,3)))
        stream=np.full(768,-1,int)
        for arm in (1,2,3):stream[np.flatnonzero(group==arm)]=np.arange(192)
        return dict(state=t['normalized_context'].numpy()[tick,env],extra=np.concatenate([normalized_force,prior],-1).astype(np.float32),prior=prior.astype(np.float32),action=t['executed_action12'].numpy()[tick,env],y=label.astype(np.float32),cells=cells,cluster=stream[env],tick=tick,environment=env,policy_group=group[env]),label,prior
    allfit=[]
    for seed in range(547,567):allfit.append(raw(Path(m['source_run'])/f's{seed}')[0])
    fit={key:np.concatenate([v[key] for v in allfit]) for key in ['state','extra','prior','action','y','cells']};del allfit
    emean=fit['extra'].mean(0);estd=fit['extra'].std(0).clip(.001);residual=fit['y']-fit['prior'];ymean=residual.mean(0);ystd=residual.std(0).clip(.001)
    for key,value in [('extra_mean',emean),('extra_std',estd),('residual_mean',ymean),('residual_std',ystd)]:assert np.array_equal(cp[key].numpy(),value),key
    template=np.zeros((12,70),np.float32);counts=np.bincount(fit['cells'],minlength=12)
    for cell in range(12):
        if counts[cell]:template[cell]=fit['state'][fit['cells']==cell].mean(0)
    assert np.array_equal(template,cp['global_templates'].numpy()) and counts.tolist()==cp['fit_cell_counts'] and len(fit['y'])==r['fit_transitions'];del fit,residual
    test,label64,prior64=raw(run/'s575');saved={k:v.numpy() for k,v in packet['data'].items()};assert all(np.array_equal(saved[k],test[k]) for k in test), 'raw identity/causality mismatch'
    assert np.all(np.abs(saved['y']-label64)<=u*np.abs(label64)+1e-10) and np.all(np.abs(saved['prior']-prior64)<=u*np.abs(prior64)+1e-10)
    extra=np.clip((test['extra']-emean)/estd,-10,10);predictions={k:v.numpy().astype(np.float64) for k,v in packet['predictions'].items()};max_forward_error=0.;max_error_fraction=0.
    for variant in ['cm','state_only','global_action']:
        state=template[test['cells']] if variant=='global_action' else test['state'];action=np.zeros_like(test['action']) if variant=='state_only' else test['action'];x=np.concatenate([state,extra,action],-1).astype(np.float32);assert np.array_equal(x,packet['features'][variant].numpy())
        weights=cp['models'][variant]
        for start in range(0,len(x),4096):
            h=x[start:start+4096].astype(np.float64)
            for layer in (0,2,4):
                h=h@weights[f'{layer}.weight'].numpy().astype(np.float64).T+weights[f'{layer}.bias'].numpy().astype(np.float64)
                if layer!=4:h=np.maximum(h,0)
            prior=test['prior'][start:start+len(h)].astype(np.float64);mean=ymean.astype(np.float64);std=ystd.astype(np.float64);pred=prior+mean+std*h
            error=np.abs(pred-predictions[variant][start:start+len(h)]);bound=std*2e-5*(1+np.abs(h))+3*u*(np.abs(prior)+np.abs(mean)+np.abs(std*h))+1e-10
            assert np.all(error<=bound),'full unit-aware neural forward audit'
            max_forward_error=max(max_forward_error,float(error.max()));max_error_fraction=max(max_error_fraction,float((error/bound).max()))
    assert np.array_equal(predictions['physical_prior'],test['prior'].astype(np.float64))
    names=['cm','state_only','global_action','physical_prior'];episode=np.zeros((192,5));np.add.at(episode[:,0],test['cluster'],1)
    for column,name in enumerate(names,1):
        per_row=np.sum((predictions[name]-test['y'].astype(np.float64))**2,axis=1)/3;np.add.at(episode[:,column],test['cluster'],per_row)
    original=np.array(json.loads((run/'fit/episode_errors.json').read_text()));assert np.max(np.abs(episode-original))<1e-9
    values=episode[:,1:].sum(0)/episode[:,0].sum();rng=np.random.RandomState(2193);b=episode[rng.randint(0,192,(2000,192))].sum(1);sample=b[:,1:]/b[:,0,None];interval={name:np.quantile(sample[:,0]-sample[:,j],[.025,.975]).tolist() for j,name in enumerate(names[1:],1)}
    gates={name:dict(mse_gain1percent=bool(values[0]<=.99*values[j]),paired_upper95_negative=bool(interval[name][1]<0)) for j,name in enumerate(names[1:],1)};label='PROMISING' if all(all(v.values()) for v in gates.values()) else 'UNPROMISING'
    assert r['label']==label and r['gates']==gates and all(abs(r['mse'][name]-values[j])<1e-12 for j,name in enumerate(names)) and all(np.max(np.abs(np.array(r['paired_difference_interval95'][name])-interval[name]))<1e-12 for name in interval)
    result=dict(run_status='COMPLETED',label=label,all_fit_statistics_and_templates_exact=True,all_current_force_inputs_reconstructed_from_prior_tick=True,physical_labels_and_prior_within_fp32_rounding_enclosures=True,all_test_model_predictions_replayed=True,full_physical_forward_max_error=max_forward_error,largest_fraction_of_predeclared_error_bound=max_error_fraction,fit_transitions=r['fit_transitions'],test_transitions=len(test['y']),common_stream_clusters=192,mse=dict(zip(names,map(float,values))),gates=gates,source_manifest_sha256=sha(run/'run_manifest.json'),source_result_sha256=sha(run/'fit/results.json'),script_sha256=sha(Path(__file__)),clock_source_sha256={str(f):sha(f) for f in [cfgpath,configpy,basepy,taskpy]},wall_seconds=time.monotonic()-begin,cpu_reason='independent NumPy raw-force/conservation/normalization/model arithmetic and bootstrap; no fitting/deployment',policy_utility_unproved=True,formal_validation=False)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
