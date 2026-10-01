#!/usr/bin/env python3
"""Report all secondary control comparisons; never alter the failed primary gate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args()
    root=args.directory;destination=root/'posthoc_controls.json'
    if destination.exists():raise ValueError('retain existing secondary analysis')
    if json.loads((root/'closeout_audit.json').read_text())['status']!='PASS':raise ValueError('unaudited primary')
    p=torch.load(root/'predictions.pt',map_location='cpu',weights_only=False)
    risk=p['risk_mm2'].numpy();boot=p['bootstrap_mm2'].numpy();seed=p['seed_risk_mm2'].numpy()
    pairs=(('factual_minus_zero',3,0),('compute_minus_zero',3,1),('factual_minus_global',2,0),
           ('compute_minus_global',2,1),('compute_minus_factual',0,1),('global_minus_zero',3,2))
    results={label:dict(point_mm2=float((risk[:,a]-risk[:,b]).mean()),upper95_mm2=float(np.quantile(boot[:,a]-boot[:,b],.95)),
        central95_mm2=np.quantile(boot[:,a]-boot[:,b],[.025,.975]).tolist(),
        optimization_seed_points_mm2=(seed[:,:,a]-seed[:,:,b]).mean(0).tolist()) for label,a,b in pairs}
    result=dict(status='POSTHOC_EXPLORATORY',primary_label_unchanged='UNPROMISING',all_six_control_pairs_reported=True,
        comparisons=results,source_sha256={str(path.resolve()):sha(path) for path in
            (Path(__file__),root/'predictions.pt',root/'results.json',root/'closeout_audit.json')})
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
