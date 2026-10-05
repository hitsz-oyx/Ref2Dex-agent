#!/usr/bin/env python3
"""Replay ref11 measured geometry/current-state/weights without new fitting."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run')]
from probe_oracle_hand_flow import load_inputs,flow_permutation,summarize,paired_observable_scores,SPECS
from oracle_hand_flow import OracleFlowHead,evaluate,exact_observable_groups
from geometric_consequence import persistence_baseline
from intervention import physical_targets
from probe_interventions import sha


def difference(first,second):
    if isinstance(first,dict):return max(difference(first[k],second[k]) for k in first)
    if isinstance(first,torch.Tensor):return float((first.cpu()-second.cpu()).abs().max())
    if first is None or isinstance(first,(str,bool)):assert first==second;return 0.
    return abs(first-second)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--split-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir/'engineering_replay.json'
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();torch.set_num_threads(2)
    m=json.loads((args.run_dir/'manifest.json').read_text());assert m['run_status']=='COMPLETED' and not m['smoke']
    assert all(sha(ROOT/k)==v for k,v in m['code_sha256'].items())
    assert sha(args.dataset)==m['dataset_sha256'] and sha(args.split_run/'diagnostic.pt')==m['split_diagnostic_sha256']
    assert all(sha(Path(k))==v for k,v in m['hand_visual_sha256'].items())
    assert sha(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')==m['urdf_sha256']
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    r=json.loads((args.run_dir/'result.json').read_text())
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    p,train,test,clusters,geometry,actions,checks,h,norms=load_inputs(args,dev)
    replay=dict(h=difference(h,d['h']),geometry=difference(geometry,d['geometry']),actions=difference(actions,d['actions']),norms=difference(norms,d['norms']))
    z=physical_targets(p['before'],p['trajectory'][:,7]).to(dev);tr=torch.as_tensor(train,device=dev)
    mean=z[tr].mean(0);scale=z[tr].std(0,unbiased=False).clamp_min(.001)
    replay.update(z=difference(z,d['z']),target_mean=difference(mean,d['target_mean']),target_scale=difference(scale,d['target_scale']))
    permutation,groups=flow_permutation(p,train,test)
    assert np.array_equal(permutation,d['permutation']) and np.array_equal(groups,d['groups'])
    assert all(np.array_equal(first,second) for first,second in ((train,d['train']),(test,d['test']),(clusters,d['clusters'])))
    perm=torch.as_tensor(permutation,device=dev);predictions={}
    for name,kind in SPECS.items():
        model=OracleFlowHead(h.shape[1]).to(dev);model.load_state_dict(d['states'][name]);model.eval()
        flow=actions[kind][perm] if name.endswith('Shuffled') else actions[kind]
        predictions[name]=(evaluate(model,h,flow)*scale+mean).cpu().numpy()
        replay[name]=float(np.abs(predictions[name]-d['predictions'][name]).max())
        if name in ('Endpoint','Chunk'):
            key=name+'_test_shuffled';predictions[key]=(evaluate(model,h,actions[kind][perm])*scale+mean).cpu().numpy()
            replay[key]=float(np.abs(predictions[key]-d['predictions'][key]).max())
    predictions['TrainMean']=mean.expand_as(z).cpu().numpy();predictions['Persistence']=persistence_baseline(p['before']).numpy()
    independent=summarize(z.cpu().numpy(),scale.cpu().numpy(),predictions,test,clusters)
    metric_errors={key:difference(independent[key],r[key]) for key in ('metrics','comparisons','gates','prediction_links')}
    pairs,support=exact_observable_groups(p,test);assert np.array_equal(pairs,d['observable_pairs'])
    paired=paired_observable_scores(pairs,z.cpu().numpy(),predictions,scale.cpu().numpy(),clusters)
    metric_errors['observable_contrasts']=difference(paired,r['observable_contrasts'])
    assert independent['status']==r['status'] and max(replay.values())<1e-5 and max(metric_errors.values())<1e-6
    report=dict(status='PASS',replay_errors=replay,metric_errors=metric_errors,geometry_checks=checks,contrast_support=support,
                diagnostic_sha256=sha(args.run_dir/'diagnostic.pt'),audit_sha256=sha(Path(__file__).resolve()),elapsed_seconds=time.monotonic()-started)
    out.write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['State','Endpoint','Chunk','EndpointShuffled','ChunkShuffled','TrainMean','Persistence']
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key in zip(axes,('E_mse','I_mse')):
        ax.barh(names,[r['metrics'][name][key] for name in names]);ax.invert_yaxis();ax.set_xlabel(key+' / source scale')
    fig.suptitle('Ref11 post-treatment oracle raw hand flow: endpoint / temporal chunks')
    fig.tight_layout();fig.savefig(args.run_dir/'oracle_hand_flow.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(status='PASS',max_replay=max(replay.values()),elapsed=time.monotonic()-started)))


if __name__=='__main__':main()
