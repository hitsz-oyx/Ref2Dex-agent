#!/usr/bin/env python3
"""No-fit replay of anatomical actions, nuisance folds and centered heads."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run')]
from probe_relative_finger_innovation import SPECS,prepare_inputs,nuisance_replay
from probe_spatial_action_fidelity import load_inputs
from relative_finger_consequence import RelativeFingerHead,evaluate_all
from probe_interventions import sha
from conditional_consequence import contrast_scores


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
    result=json.loads((args.run_dir/'result.json').read_text())
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    p,bridge,g,h,source,forecast,_,_=load_inputs(args,dev)
    nuisance=nuisance_replay(p,g,source,dev)
    common,actions,norm=prepare_inputs(p,bridge,g,h,source,forecast,dev)
    replay=dict(common=float((common.cpu()-d['common']).abs().max()))
    replay.update({'action_'+k:float((v.cpu()-d['actions'][k]).abs().max()) for k,v in actions.items()})
    replay.update({'extra_norm_'+k:float((v.cpu()-d['extra_norm'][k]).abs().max()) for k,v in norm.items()})
    for part in ('train','test'):assert np.array_equal(source[part],d[part]) and set(d['permutation'][d[part]])==set(d[part])
    mu=source['oof_predictions']['State'].to(dev);scale=source['target_scale'].to(dev)
    actual=torch.as_tensor(d['arms'],device=dev);rows=torch.arange(len(actual),device=dev)
    shuffled=actual[torch.as_tensor(d['permutation'],device=dev)];test=torch.as_tensor(d['test'],device=dev)
    means={};vectors={};contrast_errors={};metric_errors={}
    for name,(kind,centered) in SPECS.items():
        model=RelativeFingerHead(common.shape[1],centered).to(dev);model.load_state_dict(d['states'][name]);model.eval()
        value=evaluate_all(model,common,actions[kind]);arm=shuffled if name=='PredFingerShuffled' else actual
        prediction=(mu+value[rows,arm]*scale).cpu().numpy()
        candidates=(mu[test,None]+value[test]*scale).cpu().numpy()
        replay[name]=float(np.max(np.abs(prediction-d['predictions'][name])))
        replay[name+'_candidates']=float(np.max(np.abs(candidates-d['candidates'][name])))
        means[name]=float(value.mean(1).abs().max())
        if centered:assert means[name]<1e-6
        if name=='PredFingerCentered':
            shuffle=(mu+value[rows,shuffled]*scale).cpu().numpy()
            replay[name+'_test_shuffle']=float(np.max(np.abs(shuffle-d['predictions']['PredFinger_test_shuffled'])))
        error=((prediction[d['test']]-source['z'].numpy()[d['test']])/scale.cpu().numpy())**2
        metrics=dict(E_mse=float(error[:,:12].mean()),I_mse=float(error[:,12:].mean()))
        metric_errors[name]=max(abs(v-result['metrics'][name][k]) for k,v in metrics.items())
        effect=(candidates[:,1:]-candidates[:,:1]).mean(0);vectors[name]=effect.tolist()
        for key,axes in (('E',np.arange(12)),('I',np.arange(12,26))):
            c=contrast_scores(source['GT_contrasts'],effect,scale.cpu().numpy(),axes)
            contrast_errors[name+'_'+key]=max(abs(c[k]-result['contrasts'][name][key]['raw'][k]) for k in ('normalized_mse','gain_vs_zero','correlation','signed_agreement','amplitude_ratio'))
    assert max(replay.values())<1e-5 and max(metric_errors.values())<1e-6 and max(contrast_errors.values())<1e-6
    out.write_text(json.dumps(dict(status='PASS',replay_errors=replay,nuisance_replay=nuisance,
        candidate_mean_max_abs=means,metric_errors=metric_errors,contrast_errors=contrast_errors,
        audit_sha256=sha(Path(__file__).resolve()),diagnostic_sha256=sha(args.run_dir/'diagnostic.pt'),
        elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
    (args.run_dir/'per_arm_contrasts.json').write_text(json.dumps(dict(predicted=vectors,GT_adjusted=source['GT_contrasts'].tolist(),
        units='E: m/rad/m/s/rad/s; I: log net-force/proximity(m)/proxy; candidate contrasts are exploratory, not per-state counterfactual GT'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5));names=list(SPECS)
    for ax,key in zip(axes,('E_mse','I_mse')):
        ax.barh(names,[result['metrics'][n][key] for n in names]);ax.invert_yaxis()
        ax.axvline(result['metrics']['State'][key],color='black',linestyle='--',label='Frozen State')
        ax.set_xlabel(key+' / fixed source scale');ax.legend()
    fig.suptitle('Full anatomical finger flow / OOF-state innovation / fixed300 updates')
    fig.tight_layout();fig.savefig(args.run_dir/'relative_finger_innovation.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='PASS',replay_max=max(replay.values()),nuisance_max=max(nuisance.values()),elapsed=time.monotonic()-started)))


if __name__=='__main__':main()
