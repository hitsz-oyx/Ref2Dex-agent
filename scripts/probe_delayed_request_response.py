"""Fixed Gaussian-innovation moments, observed closed-loop outcomes, no fitting."""
import argparse,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import yaml
from scripts.run_contact_response_probe import sha
def ar(t):return t.numpy().astype(np.float64)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();run=a.run.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('unique own probe')
    m=json.loads((run/'run_manifest.json').read_text());assert m['run_status']=='COMPLETED' and m['training_seeds']==list(range(547,567))
    cmd=m['phases'][0]['command'];cfgpath=Path(cmd[cmd.index('--cfg_env')+1]);cfg=yaml.safe_load(cfgpath.read_text())
    configpy=ROOT/'third_party/DExplore/dexplore/utils/config.py';basepy=ROOT/'third_party/DExplore/dexplore/env/tasks/base_task.py'
    assert 'SIM_TIMESTEP = 1.0 / 60.0' in configpy.read_text() and 'sim_params.dt = SIM_TIMESTEP' in configpy.read_text()
    assert 'for i in range(self.control_freq_inv):' in basepy.read_text() and cfg['env']['controlFrequencyInv']==2 and 'dt' not in cfg['sim']
    dt=2/60;torch.set_num_threads(2);begin=time.monotonic();sources={}
    for f in [cfgpath,configpy,basepy,Path(__file__),ROOT/'docs/experiments/probes/P-20261002-delayed-request-response.md',run/'run_manifest.json']:sources[str(f.resolve())]=sha(f)
    out.mkdir();manifest=dict(run_status='STARTED',experiment_id='P-20261002-delayed-request-response',training_only=True,no_models_or_native_rollouts=True,wall_limit_seconds=120,storage_limit_bytes=5<<20,cpu_reason='file/physical-label audits and moment/cluster-bootstrap statistics; no neural computation')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');records=[];panels=[];noise_error=position_error=0.
    try:
        for seed in range(547,567):
            if time.monotonic()-begin>120:raise TimeoutError('fixed probe budget')
            d=run/f's{seed}';native=json.loads((d/'results.json').read_text());audit=json.loads((d/'panel_audit.json').read_text())
            assert native['run_status']==audit['run_status']=='COMPLETED' and not native['deterministic'] and native['continuous_updates']==seed-547
            assert sha(d/'initial.pt')==native['initial_sha256'] and sha(d/'trace.pt')==native['trace_sha256']
            for f in ['initial.pt','trace.pt','results.json','panel_audit.json']:sources[str((d/f).resolve())]=sha(d/f)
            initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
            checkpoint=Path(m['panel_checkpoints'][str(seed)]['path']);assert sha(checkpoint)==native['continuous_checkpoint_sha256'];sources[str(checkpoint.resolve())]=sha(checkpoint)
            cp=torch.load(checkpoint,map_location='cpu',weights_only=False);sigma=np.exp(np.clip(ar(cp['variants']['none']['actor']['log_std']),-5,0))
            ids=np.flatnonzero((initial['motion'].numpy()==2)&(initial['policy_group'].numpy()==3));assert len(ids)==64
            post=ar(trace['object_root'][:,ids]);pre=np.concatenate([ar(initial['object_root'][ids])[None],post[:-1]],axis=0)
            eps=ar(trace['request_noise'][:,ids]);actual=(ar(trace['request'][:,ids])-ar(trace['request_mean'][:,ids]))/sigma
            noise_error=max(noise_error,float(np.abs(actual-eps).max()));assert noise_error<=2e-5
            stored=ar(trace['physical_transition'][:,ids,2])*.005
            position_error=max(position_error,float(np.abs((post[...,2]-pre[...,2])-stored).max()*1000));assert position_error<=1e-5
            current_clearance=np.concatenate([np.full((1,64),np.nan),ar(trace['clearance'][:-1,ids])],axis=0)
            progress=trace['progress'][:194,ids].numpy();stop=float(initial['phase_stop'][2])
            eligible=(progress>=stop-74)&(progress<=stop+30)&(pre[:194,:,2]-ar(initial['initial_height'][ids])[None]>=.03)&(current_clearance[:194]>=.02)
            y1=(post[:194,:,2]-pre[:194,:,2]-dt*pre[:194,:,9])*1000
            y8=(post[7:201,:,2]-pre[:194,:,2]-8*dt*pre[:194,:,9])*1000
            moments=np.stack([eligible.sum(0),(eps[:194,:,2]*y1*eligible).sum(0),(eps[:194,:,2]*y8*eligible).sum(0),(eps[8:202,:,2]*y8*eligible).sum(0)],axis=-1)
            records.append(moments);panels.append(dict(seed=seed,eligible_transitions=int(eligible.sum()),eligible_episodes=int((eligible.sum(0)>0).sum())))
            print(json.dumps(panels[-1]),flush=True);del trace,initial,post,pre,eps,actual,cp
        data=np.stack(records);counts=data[...,0];total=data.sum((0,1));n=int(total[0]);episodes=int((counts>0).sum());assert np.isfinite(data).all()
        result=dict(run_status='COMPLETED',scope='fixed delayed closed-loop request-response moment only; not model or policy utility',training_only=True,final_evaluation_unused=True,eligible_transitions=n,eligible_episodes=episodes,panels=panels,horizons=[1,8],request_coordinate=2,object_world_coordinate=2,action_tick_seconds=dt,noise_maximum_error=noise_error,position_reconstruction_maximum_mm=position_error,no_models_optimizer_or_physics=True,source_sha256=sources,script_sha256=sha(Path(__file__)),one_realized_learner_sequence=True,journal_ready=False)
        if n<1024 or episodes<32:result.update(label='UNCLEAR',reason='fixed eligibility minimum not met')
        else:
            point=total[1:]/total[0];rng=np.random.RandomState(947);boot=np.zeros((2000,4))
            for block in data:boot+=block[rng.randint(0,64,size=(2000,64))].sum(1)
            assert (boot[:,0]>0).all();samples=boot[:,1:]/boot[:,0,None];interval=np.quantile(samples,[.025,.975],axis=0);difference=np.quantile(samples[:,1]-samples[:,0],[.025,.975])
            gates=dict(delayed_moment_at_least_point1mm=bool(point[1]>=.1),paired_delayed_minus_immediate_lower95_positive=bool(difference[0]>0),future_innovation_equivalent_within_point1mm=bool(interval[0,2]>=-.1 and interval[1,2]<=.1))
            label='UNCLEAR' if not gates['future_innovation_equivalent_within_point1mm'] else 'PROMISING' if all(gates.values()) else 'UNPROMISING'
            result.update(label=label,moments_mm=dict(immediate=float(point[0]),delayed=float(point[1]),future_negative_control=float(point[2])),interval95_mm=dict(immediate=interval[:,0].tolist(),delayed=interval[:,1].tolist(),future_negative_control=interval[:,2].tolist(),paired_delayed_minus_immediate=difference.tolist()),gates=gates,bootstrap_scope='2000 episode-cluster resamples within each panel; exploratory conditional realized-sequence uncertainty, not independent training replications')
        for filename,h in sources.items():assert sha(Path(filename))==h,filename
        result['wall_seconds']=time.monotonic()-begin;assert result['wall_seconds']<=120
        (out/'episode_moments.json').write_text(json.dumps(data.tolist())+'\n');(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');manifest.update(run_status='COMPLETED',label=result['label'],source_sha256=sources,wall_seconds=result['wall_seconds']);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','panels']}),flush=True)
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
