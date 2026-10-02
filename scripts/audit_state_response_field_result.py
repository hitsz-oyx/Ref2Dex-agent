"""Independent raw-label, FIT-control, episode-risk and frozen-gate closeout."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert ROOT in out.parents and not out.exists();out.mkdir();begin=time.monotonic();torch.set_num_threads(2)
    run=a.directory.resolve();m=json.loads((run/'run_manifest.json').read_text());r=json.loads((run/'fit/results.json').read_text());assert m['run_status']==r['run_status']=='COMPLETED'
    for f,h in {**m['input_sha256'],**r['source_sha256']}.items():assert sha(Path(f))==h,f
    assert sha(run/'fit/response_field.pt')==r['checkpoint_sha256'] and sha(run/'fit/test_predictions.pt')==r['predictions_sha256']
    cp=torch.load(run/'fit/response_field.pt',map_location='cpu',weights_only=False);packets=torch.load(run/'fit/test_predictions.pt',map_location='cpu',weights_only=False)
    def forward(x,state):
        x=x.astype(np.float64)
        for layer in (0,2,4):
            x=x@state[f'{layer}.weight'].numpy().astype(np.float64).T+state[f'{layer}.bias'].numpy().astype(np.float64)
            if layer!=4:x=np.maximum(x,0)
        return x
    def raw(path):
        i=torch.load(path/'initial.pt',map_location='cpu',weights_only=False);t=torch.load(path/'trace.pt',map_location='cpu',weights_only=False)
        group=i['policy_group'].numpy();motion=i['motion'].numpy();progress=t['progress'].numpy();stop=i['phase_stop'].numpy()[motion];lift=i['lift_start'].numpy()[motion]
        pre=np.concatenate([i['object_root'].numpy()[None],t['object_root'].numpy()[:-1]],0);post=t['object_root'].numpy()
        target=np.concatenate([post[...,:3].astype(np.float64)-pre[...,:3],post[...,7:10].astype(np.float64)-pre[...,7:10]],-1)/np.array([.005]*3+[.05]*3)
        phase=np.zeros_like(progress);phase[progress>=lift[None]]=1;phase[progress>=stop[None]-74]=2;phase[progress>stop[None]]=3;cells=motion[None]*4+phase
        mask=(progress>=stop[None]-74)&(progress<=stop[None]+30);assert np.all(mask.sum(0)==105)
        ans=[]
        for arm in (1,2,3):
            ids=np.flatnonzero(group==arm);valid=mask[:,ids].T
            def pick(v):return v[:,ids].swapaxes(0,1)[valid]
            ans.append(dict(x=pick(t['normalized_context'].numpy()),y=pick(target),eps=pick(t['request_noise'].numpy()),cells=pick(cells)))
        return ans
    fit=raw(Path(m['source_run'])/'s547');global_response=np.zeros((12,72));cell_counts=np.zeros(12,int);sum_response=np.zeros((12,72))
    for d in fit:
        b=forward(d['x'],cp['models']['baseline']);moment=(d['eps'][:,:,None]*(d['y']-b)[:,None,:]).reshape(-1,72)
        for c in range(12):
            mask=d['cells']==c;cell_counts[c]+=int(mask.sum());sum_response[c]+=moment[mask].sum(0)
    used=cell_counts>0;global_response[used]=sum_response[used]/cell_counts[used,None]
    control=cp['global_field'].numpy().astype(np.float64);global_error=float(np.abs(global_response-control).max());assert global_error<2e-5
    energy=float((control**2*cell_counts[:,None]).sum()/(cell_counts.sum()*72));assert abs(energy-r['fit_global_response_energy'])<1e-12
    test=raw(run/'s570');episode=np.zeros((192,2));model_error=label_error=0.
    for key,d in enumerate(test):
        q=packets[str(key)];x=q['x'].numpy();y=q['y'].numpy();eps=q['eps'].numpy();cells=q['cells'].numpy();b=q['baseline'].numpy().astype(np.float64);field=q['field'].numpy().astype(np.float64)
        assert np.array_equal(x,d['x']) and np.array_equal(eps,d['eps']) and np.array_equal(cells,d['cells'])
        label_error=max(label_error,float(np.abs(y-d['y']).max()));assert label_error<1e-5
        for start in range(0,len(x),4096):
            model_error=max(model_error,float(np.abs(forward(x[start:start+4096],cp['models']['baseline'])-b[start:start+4096]).max()),float(np.abs(forward(x[start:start+4096],cp['models']['field'])-field[start:start+4096]).max()))
        assert model_error<2e-5
        # Replay the stored float32 training target algebra, then accumulate double.
        moment=(eps[:,:,None]*(y-q['baseline'].numpy())[:,None,:]).reshape(-1,72).astype(np.float64);g=control[cells]
        risk=(np.sum(field*field-g*g,axis=1)-2*np.sum((field-g)*moment,axis=1))/72
        episode[:,0]+=risk.reshape(192,105).sum(1);episode[:,1]+=105
    saved=np.array(json.loads((run/'fit/episode_risks.json').read_text()));episode_error=float(np.abs(episode-saved).max());assert episode_error<1e-10
    difference=float(episode[:,0].sum()/episode[:,1].sum());rng=np.random.RandomState(1085);boot=episode[rng.randint(0,192,(2000,192))].sum(1);ci=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist()
    gates=dict(nonzero_fit_global_energy=energy>0,risk_gain_at_least_ten_percent=difference<=-.1*energy,paired_upper95_negative=ci[1]<0)
    label='PROMISING' if all(gates.values()) else 'UNPROMISING';assert r['gates']==gates and r['label']==label and abs(r['paired_risk_difference']-difference)<1e-12 and np.max(np.abs(np.array(r['interval95'])-ci))<1e-12
    result=dict(run_status='COMPLETED',label=label,raw_test_transitions=60480,all_test_model_predictions_independently_replayed=True,common_stream_clusters=192,fit_global_max_error=global_error,physical_target_max_error=label_error,full_numpy_forward_max_error=model_error,episode_risk_max_error=episode_error,paired_risk_difference=difference,interval95=ci,fit_global_response_energy=energy,gates=gates,cpu_reason='independent NumPy physics/model arithmetic and statistical audit; no fitted or deployed model',source_manifest_sha256=sha(run/'run_manifest.json'),result_sha256=sha(run/'fit/results.json'),script_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin,formal_validation=False,policy_utility_not_demonstrated=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
