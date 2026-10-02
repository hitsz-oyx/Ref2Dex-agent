"""Independent before-physics command counterfactual; no model/physics fitting."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
INDEPENDENT=[0,1,2,3,4,5,6,8,10,12,14,15]
COUPLING={6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}
SCALES=np.array([.02]*3+[.10]*3+[.15]*6)
THRESHOLDS=np.array([.0001]*3+[.001]*3+[.005]*6)

def array(t):return t.numpy().astype(np.float64)
def project(base,request,q,lower,upper,pd):
    goal=base.copy();delta=np.tanh(request)*SCALES
    goal[...,:6]=np.clip(base[...,:6]+delta[...,:6],q[...,:6]-np.abs(pd[:6]),q[...,:6]+np.abs(pd[:6]))
    for column,parent in enumerate(INDEPENDENT[6:],6):
        children=COUPLING.get(parent,())
        lo=max([lower[parent]]+[lower[j]/ratio for j,ratio in children]);hi=min([upper[parent]]+[upper[j]/ratio for j,ratio in children])
        assert lo<=hi
        goal[...,parent]=np.clip(base[...,parent]+delta[...,column],lo,hi)
        for child,ratio in children:goal[...,child]=goal[...,parent]*ratio
    return goal

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();run=a.run.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('unique own output')
    m=json.loads((run/'run_manifest.json').read_text());assert m['run_status']=='COMPLETED'
    assert m['training_seeds']==list(range(547,567)) and m['evaluation_seeds']==[568,569]
    torch.set_num_threads(2);begin=time.monotonic();names=['cm','state_only','none'];summary={n:dict(n=0,meaningful=0,null_at1e7=0,coordinate_meaningful=np.zeros(12,dtype=np.int64),episode_fractions=[]) for n in names};sources={str(run/'run_manifest.json'):sha(run/'run_manifest.json')};panels=[];maximum=0.
    for filename in ['docs/experiments/probes/P-20261002-request-execution-headroom.md','docs/decisions/D-20261002-after-continuous-policy-route-review.md','scripts/probe_request_execution_headroom.py','src/task/CmResidual/continuous_critic_cm.py','src/task/CmResidual/selective_finger_response.py']:
        sources[str(ROOT/filename)]=sha(ROOT/filename)
    out.mkdir();manifest=dict(run_status='STARTED',experiment_id='P-20261002-request-execution-headroom',source_run=str(run),training_only=True,no_models_or_native_rollouts=True,cpu_reason='file processing, deterministic command audit and statistics; no neural model computation',wall_limit_seconds=120,storage_limit_bytes=5<<20)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        for seed in range(547,567):
            if time.monotonic()-begin>120:raise TimeoutError('fixed screen budget')
            d=run/f's{seed}';native=json.loads((d/'results.json').read_text());audit=json.loads((d/'panel_audit.json').read_text())
            assert native['run_status']==audit['run_status']=='COMPLETED' and not native['deterministic'] and native['continuous_updates']==seed-547
            assert sha(d/'initial.pt')==native['initial_sha256'] and sha(d/'trace.pt')==native['trace_sha256']
            for f in ['initial.pt','trace.pt','results.json','panel_audit.json']:sources[str((d/f).resolve())]=sha(d/f)
            initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
            ids=np.flatnonzero((initial['motion'].numpy()==2)&(initial['policy_group'].numpy()>0));assert len(ids)==192
            q=np.concatenate([array(initial['base_q'][ids])[None],array(trace['native_q'][:-1,ids])],axis=0)
            base=array(trace['base_target'][:,ids]);request=array(trace['request'][:,ids]);mean=array(trace['request_mean'][:,ids]);saved=array(trace['target'][:,ids])
            lower=array(initial['native_lower']);upper=array(initial['native_upper']);pd=array(initial['pd_scale'])
            actual=project(base,request,q,lower,upper,pd);counterfactual=project(base,mean,q,lower,upper,pd)
            error=float(np.abs(actual-saved).max());maximum=max(maximum,error);assert error<=1e-6,error
            difference=np.abs(actual[...,INDEPENDENT]-counterfactual[...,INDEPENDENT]);per_coordinate=difference>=THRESHOLDS
            meaningful=per_coordinate.any(-1);null=(difference<=1e-7).all(-1)
            progress=trace['progress'][:,ids].numpy();stop=float(initial['phase_stop'][2]);window=(progress>=stop-74)&(progress<=stop+30)
            assert np.all(window.sum(0)==105)
            panel={}
            for group,name in enumerate(names,1):
                selected=initial['policy_group'][ids].numpy()==group;assert selected.sum()==64
                use=window[:,selected];value=meaningful[:,selected];n=int(use.sum());assert n==6720
                row=dict(n=n,meaningful=int((value&use).sum()),null_at1e7=int((null[:,selected]&use).sum()),projection_maximum=error)
                panel[name]=row;s=summary[name];s['n']+=n;s['meaningful']+=row['meaningful'];s['null_at1e7']+=row['null_at1e7']
                s['coordinate_meaningful']+=(per_coordinate[:,selected]&use[...,None]).sum((0,1))
                s['episode_fractions']+=((value&use).sum(0)/105).tolist()
            panels.append(dict(seed=seed,arms=panel));print(json.dumps(dict(seed=seed,projection_error=error)),flush=True)
            del trace,initial,q,base,request,mean,saved,actual,counterfactual
        gates={}
        for name,s in summary.items():
            assert s['n']==134400 and len(s['episode_fractions'])==1280
            fractions=np.array(s.pop('episode_fractions'));s['meaningful_fraction']=s['meaningful']/s['n'];s['null_at1e7_fraction']=s['null_at1e7']/s['n'];s['coordinate_meaningful_fraction']=(s.pop('coordinate_meaningful')/s['n']).tolist();s['episode_fraction_quantiles']=np.quantile(fractions,[0,.25,.5,.75,1]).tolist()
            gates[name+'_at_least_half_meaningful']=2*s['meaningful']>=s['n']
        label='PROMISING' if all(gates.values()) else 'UNPROMISING'
        for filename,h in sources.items():assert sha(Path(filename))==h,filename
        result=dict(run_status='COMPLETED',label=label,scope='request innovation execution headroom only, not physical effect or policy utility',summary=summary,gates=gates,panels=panels,primary_motion=2,primary_steps_per_episode=105,total_training_episodes=3840,total_transitions=403200,thresholds=THRESHOLDS.tolist(),independent_coordinates=INDEPENDENT,independent_projection_maximum=maximum,training_only=True,final_evaluation_unused=True,no_new_models_optimizer_or_physics=True,before_physics_counterfactual_only=True,source_sha256=sources,script_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin,journal_ready=False)
        assert result['wall_seconds']<=120
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');manifest.update(run_status='COMPLETED',label=label,source_sha256=sources,wall_seconds=result['wall_seconds'])
        print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','panels']}),flush=True)
    except BaseException as e:manifest.update(run_status='FAILED',error=repr(e));raise
    finally:(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':main()
