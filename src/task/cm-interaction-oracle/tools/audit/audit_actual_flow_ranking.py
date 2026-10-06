#!/usr/bin/env python3
"""No-fit reconstruction of nested OOF features and held predictions."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'src/task/cm-interaction-oracle/src'), str(ROOT/'src/task/cm-interaction-oracle/tools/run'), str(Path(__file__).parent)]
from package_rolling_oracle_asset import Snapshot, sha
from probe_actual_flow_ranking import load_data, summarize
from actual_flow_ranking import ARMS, Head, h_inputs, anchor_folds, rows_for, partition, predict, readout_input
from geometric_consequence import standardize_fit, normalize


def max_error(a,b):
    if isinstance(a,dict):
        if set(a) != set(b):
            raise ValueError('normalizer keys differ')
        return max(max_error(a[k],b[k]) for k in a)
    return float((a-b).abs().max())


def reconstruct(data, artifact, fit_panels, checks):
    ids = rows_for(fit_panels,'cuda:0')
    h, hn = h_inputs(data,ids)
    tn = standardize_fit(data['ei'],ids)
    checks.append(max_error(hn,artifact['h_norms']))
    checks.append(max_error(tn,artifact['target_norm']))
    model = Head(h.shape[1]+720,26).to('cuda:0')
    model.load_state_dict(artifact['weights']); model.eval()
    raw = predict(model,torch.cat((h,data['flow']),-1))*tn['scale']+tn['mean']
    return raw, h, model, tn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset-run',type=Path,required=True)
    ap.add_argument('--dense-run',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    args = ap.parse_args()
    output = args.run_dir/'replay.json'
    if output.exists():
        raise ValueError('refusing existing replay')
    start = time.monotonic(); torch.set_num_threads(2)
    manifest = json.loads((args.run_dir/'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED' or manifest['smoke']:
        raise ValueError('completed scientific run required')
    for path, expected in manifest['input_sha256'].items():
        # Card may acquire a Result after the run; its frozen bytes are archived.
        target = args.run_dir/'protocol.md' if path.endswith('P-20261006-actual-flow-y-ranking.md') else Path(path)
        if sha(target) != expected:
            raise ValueError('immutable input/code mismatch: '+str(target))
    if sha(args.run_dir/'models.pt') != manifest['models_sha256'] or sha(args.run_dir/'result.json') != manifest['result_sha256']:
        raise ValueError('saved output hash mismatch')
    snapshot = Snapshot(); cpu = load_data(args.asset_run,args.dense_run,snapshot)
    data = {k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in cpu.items()}
    saved = torch.load(args.run_dir/'models.pt',map_location='cuda:0',weights_only=False)
    if saved['panel_keys'] != cpu['panel_keys']:
        raise ValueError('panel identity mismatch')
    folds = anchor_folds(cpu['panel_keys'],4,271)
    if not np.array_equal(folds,saved['outer_folds']):
        raise ValueError('outer split mismatch')
    checks = []; preds = {a:torch.full_like(cpu['y'],float('nan')) for a in ARMS}
    shuf = {a:torch.full_like(cpu['y'],float('nan')) for a in ARMS}
    rng = np.random.default_rng(272)
    perm = torch.tensor(np.stack([rng.permutation(7) for _ in cpu['panel_keys']]),device='cuda:0')
    if not torch.equal(perm,saved['permutations']):
        raise ValueError('shuffle schedule mismatch')
    sf = data['flow'].reshape(-1,7,720).gather(1,perm[:,:,None].expand(-1,-1,720)).reshape(-1,720)
    with torch.no_grad():
        for f, artifact in enumerate(saved['folds']):
            if time.monotonic()-start > 100:
                raise TimeoutError('120second replay budget')
            source, held = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            partition(cpu['panel_keys'],source,held)
            if not np.array_equal(source,artifact['fit_panels']) or not np.array_equal(held,artifact['hold_panels']):
                raise ValueError('saved outer indices drift')
            inner = anchor_folds([cpu['panel_keys'][i] for i in source],3,272)
            feat = torch.full_like(data['ei'],float('nan'))
            for j, item in enumerate(artifact['inner']):
                fitp, holdp = source[inner != j], source[inner == j]
                partition(cpu['panel_keys'],fitp,holdp)
                if not np.array_equal(fitp,item['fit_panels']) or not np.array_equal(holdp,item['hold_panels']):
                    raise ValueError('saved nested indices drift')
                values,_,_,_ = reconstruct(data,item,fitp,checks)
                ids = rows_for(holdp,'cuda:0'); feat[ids] = values[ids]
            full,h,pm,tn = reconstruct(data,artifact['physical'],source,checks)
            tr,te = rows_for(source,'cuda:0'),rows_for(held,'cuda:0')
            feat[te] = full[te]
            pn = standardize_fit(feat,tr); gn = standardize_fit(data['ei'],tr)
            yn = standardize_fit(data['y'],tr,floor=.05)
            for value,stored in ((pn,artifact['predicted_norm']),(gn,artifact['gt_norm']),(yn,artifact['y_norm'])):
                checks.append(max_error(value,stored))
            _,hn = h_inputs(data,tr); checks.append(max_error(hn,artifact['h_norms']))
            sp = predict(pm,torch.cat((h,sf),-1))*tn['scale']+tn['mean']
            for arm in ARMS:
                ei = normalize(data['ei'],gn) if arm == 'GT_EI' else normalize(feat,pn)
                x = readout_input(h,data['flow'],ei,arm)
                model = Head(x.shape[1],8).to('cuda:0')
                model.load_state_dict(artifact['heads'][arm]['weights']); model.eval()
                p = predict(model,x)*yn['scale']+yn['mean']
                si = ei if arm == 'GT_EI' else normalize(sp,pn)
                s = predict(model,readout_input(h,sf,si,arm))*yn['scale']+yn['mean']
                preds[arm][te.cpu()] = p[te].cpu(); shuf[arm][te.cpu()] = s[te].cpu()
    prediction_error = max(float((preds[a]-saved['predictions'][a].cpu()).abs().max()) for a in ARMS)
    shuffle_error = max(float((shuf[a]-saved['shuffled'][a].cpu()).abs().max()) for a in ARMS)
    result = summarize(cpu,preds,shuf,folds)
    old = json.loads((args.run_dir/'result.json').read_text())
    exact = result == old
    report = dict(status='PASS' if max(checks+[prediction_error,shuffle_error]) <= 1e-6 and exact else 'FAIL',
                  normalizer_max_error=max(checks),prediction_max_error=prediction_error,
                  shuffle_max_error=shuffle_error,exact_summary_replay=exact,
                  elapsed_seconds=time.monotonic()-start, no_fitting_or_simulation=True)
    output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    if report['status'] != 'PASS':
        raise ValueError('saved weight replay mismatch')


if __name__ == '__main__':
    main()
