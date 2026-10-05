#!/usr/bin/env python3
"""No-fit, post-treatment FK diagnostic: wrist versus finger endpoint error."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/cm-interaction-oracle'
sys.path[:0]=[str(ROOT),str(TASK/'src'),str(TASK/'tools/run')]
from geometric_consequence import NominalSurfaceActions
from execution_geometry import endpoint_flows
from consequence_sufficiency import cluster_gain
from probe_interventions import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();out=args.run_dir/'endpoint_decomposition.json'
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((args.run_dir/'manifest.json').read_text())
    assert manifest['run_status']=='COMPLETED' and not manifest['smoke']
    assert manifest['projection']=='bounded_only_preserve_continuous'
    assert sha(args.dataset)==manifest['dataset_sha256']
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    original=torch.load(Path(d['original_run'])/'diagnostic.pt',map_location='cpu',weights_only=False)
    assert sha(Path(d['original_run'])/'diagnostic.pt')==manifest['original_diagnostic_sha256']
    p=torch.load(args.dataset,map_location='cpu',weights_only=False)
    torch.set_num_threads(2);dev=torch.device('cuda:0')
    bridge=NominalSurfaceActions(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',dev)
    g={k:v.to(dev) for k,v in original['source_geometry'].items()}
    rows=torch.arange(len(d['arms']),device=dev);arms=torch.as_tensor(d['arms'],device=dev)
    real=p['native_q'][:,7].to(dev);pred=d['full_q']['Ha'].to(dev)[rows,arms]
    pred_wrist_real_finger=torch.cat((pred[:,:6],real[:,6:]),-1)
    real_wrist_pred_finger=torch.cat((real[:,:6],pred[:,6:]),-1)
    # These hybrid inputs read q8: diagnosis only, never a causal feature.
    actual_flow,_=endpoint_flows(p,bridge,g,real[:,None])
    errors={};metrics={};test=d['test'];clusters=d['clusters'][test]
    for name,q in (('Predicted',pred),('PredWrist_RealFinger',pred_wrist_real_finger),
                   ('RealWrist_PredFinger',real_wrist_pred_finger),('Nominal',g['targets'][rows,arms])):
        flow,_=endpoint_flows(p,bridge,g,q[:,None])
        error=(flow-actual_flow).square().mean((1,2,3)).cpu().numpy()[test]
        errors[name]=error
        ranked=np.sort(error)[::-1];total=error.sum()
        metrics[name]=dict(rmse_mm=float(np.sqrt(error.mean())*1000),
            pointwise_euclidean_rmse_mm=float(np.sqrt(3*error.mean())*1000),
            top1_mse_share=float(ranked[:1].sum()/total),top5_mse_share=float(ranked[:5].sum()/total),
            median_window_component_rmse_mm=float(np.sqrt(np.median(error))*1000))
    assert abs(metrics['Predicted']['rmse_mm']-json.loads((args.run_dir/'result.json').read_text())['execution_metrics']['Ha']['surface_endpoint_rmse_mm'])<1e-5
    comparisons={name:cluster_gain(errors['Predicted'],errors[name],clusters,247)
                 for name in ('PredWrist_RealFinger','RealWrist_PredFinger')}
    report=dict(status='DIAGNOSTIC',metrics=metrics,comparisons=comparisons,
        diagnostic_sha256=sha(args.run_dir/'diagnostic.pt'),code_sha256=sha(Path(__file__).resolve()),
        scope='Post-result post-treatment FK ablations. No fit, row exclusion, selector, gate change, or physical counterfactual claim. RMSE primary averages squared XYZ components; Euclidean metric shown separately.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
