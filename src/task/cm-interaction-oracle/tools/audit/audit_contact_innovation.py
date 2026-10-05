#!/usr/bin/env python3
"""No-fit contact feature, nuisance, closed-form weight and metric replay."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run')]
from probe_contact_innovation import build_inputs,evaluate_candidates,summarize,MODES,KINDS
from probe_spatial_action_fidelity import load_inputs
from probe_relative_finger_innovation import nuisance_replay
from geometric_consequence import physics_baseline
from contact_innovation import bounded_design
from probe_interventions import sha


def max_difference(first,second):
    if isinstance(first,dict):return max(max_difference(first[k],second[k]) for k in first)
    if isinstance(first,torch.Tensor):return float((first.cpu()-second.cpu()).abs().max())
    if first is None or isinstance(first,(str,bool)):assert first==second;return 0.
    return abs(first-second)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--forecast-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir/'engineering_replay.json'
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();torch.set_num_threads(2)
    m=json.loads((args.run_dir/'manifest.json').read_text());assert m['run_status']=='COMPLETED' and not m['smoke']
    assert all(sha(ROOT/k)==v for k,v in m['code_sha256'].items())
    assert sha(args.dataset)==m['dataset_sha256'] and sha(args.forecast_run/'diagnostic.pt')==m['forecast_diagnostic_sha256']
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    saved=json.loads((args.run_dir/'result.json').read_text())
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    p,bridge,g,h,source,forecast,_,_=load_inputs(args,dev)
    nuisance=nuisance_replay(p,g,source,dev)
    common,actions,norms,contact_state=build_inputs(p,bridge,g,h,source,forecast,dev)
    replay=dict(common=max_difference(common,d['common']),actions=max_difference(actions,d['actions']),
                norms=max_difference(norms,d['norms']),contact_state=max_difference(contact_state,d['contact_state']))
    mu=dict(Physics=physics_baseline(p['before'].to(dev),8/30),OOFState=source['oof_predictions']['State'].to(dev))
    replay['mu']=max_difference(mu,d['mu'])
    rows=torch.arange(len(common),device=dev);arms=torch.as_tensor(d['arms'],device=dev)
    shuffled=arms[torch.as_tensor(d['permutation'],device=dev)]
    tr,te=(torch.as_tensor(d[k],device=dev) for k in ('train','test'))
    assert all(np.array_equal(d[k],source[k]) for k in ('train','test','arms','clusters'))
    for ids in (d['train'],d['test']):assert set(d['permutation'][ids])==set(ids)
    z=source['z'].to(dev);scale=source['target_scale'].to(dev)
    records={};vectors={}
    for mode in MODES:
        for kind in KINDS:
            name=mode+'_'+kind;key='Contact' if kind=='ContactShuffled' else kind
            arm=shuffled if kind=='ContactShuffled' else arms
            state={k:v.to(dev) if isinstance(v,torch.Tensor) else v for k,v in d['states'][name].items()}
            value=evaluate_candidates(common,actions[key],norms['actions'][key],state)
            pred=(mu[mode]+value[rows,arm]*scale).cpu().numpy()
            candidates=(mu[mode][te,None]+value[te]*scale).cpu().numpy()
            replay[name]=float(np.abs(pred-d['predictions'][name]).max())
            replay[name+'_candidates']=float(np.abs(candidates-d['candidates'][name]).max())
            x=bounded_design(common,actions[key][rows,arm],norms['actions'][key])[tr].double()-state['x_mean']
            target=((z-mu[mode])/scale)[tr].double()-state['y_mean']
            rhs=x.T@target
            records[name]=float((x.T@(x@state['weight']-target)+32*state['weight']).abs().max()/rhs.abs().max().clamp_min(1))
            assert records[name]<1e-8
            vectors[name]=(candidates[:,1:]-candidates[:,:1]).mean(0).tolist()
            if kind=='State':assert float((value-value[:,:1]).abs().max())==0
            if kind=='Contact':
                shuffle=(mu[mode]+value[rows,shuffled]*scale).cpu().numpy()
                replay[name+'_test_shuffle']=float(np.abs(shuffle-d['predictions'][name+'_test_shuffled']).max())
    independent=summarize(source,d['predictions'],d['candidates'],scale.cpu().numpy(),d['test'])
    metrics={k:max_difference(independent[k],saved[k]) for k in ('metrics','comparisons','contrasts','gates')}
    assert independent['status']==saved['status'] and max(replay.values())<1e-5 and max(metrics.values())<1e-6
    out.write_text(json.dumps(dict(status='PASS',replay_errors=replay,nuisance_replay=nuisance,normal_equation_relative_errors=records,
        metric_errors=metrics,diagnostic_sha256=sha(args.run_dir/'diagnostic.pt'),audit_sha256=sha(Path(__file__).resolve()),
        elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    (args.run_dir/'per_arm_contrasts.json').write_text(json.dumps(dict(predicted=vectors,GT_adjusted=source['GT_contrasts'].tolist(),
        scope='Exposed marginal E12/I14 contrasts; local/current-wrist sampled proximity proxy, no contact certification or independent GT'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[mode+'_'+kind for mode in MODES for kind in KINDS]
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    for ax,key in zip(axes,('E_mse','I_mse')):
        ax.barh(names,[saved['metrics'][name][key] for name in names],color=['tab:blue']*5+['tab:orange']*5)
        ax.axvline(saved['metrics']['FrozenState'][key],color='black',linestyle='--',label='Prior frozen State')
        ax.invert_yaxis();ax.set_xlabel(key+' / fixed source scale');ax.legend()
    fig.suptitle('Contact geometry / matched nuisance / bounded ridge32')
    fig.tight_layout();fig.savefig(args.run_dir/'contact_innovation.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='PASS',max_replay=max(replay.values()),max_normal=max(records.values()),elapsed=time.monotonic()-started)))


if __name__=='__main__':main()
