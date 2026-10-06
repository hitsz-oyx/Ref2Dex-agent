#!/usr/bin/env python3
"""Nested held-anchor direct/bottleneck/hybrid oracle-Y ranking Probe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT), str(ROOT/'src/task/cm-interaction-oracle/src'), str(ROOT/'src/task/cm-interaction-oracle/tools/audit')]
from package_rolling_oracle_asset import Snapshot, sha
from actual_flow_ranking import (ARMS, Head, join_panels, anchor_folds, rows_for, partition,
    h_inputs, physical_fit, fit, predict, readout_input, ranking, bootstrap_difference)
from geometric_consequence import standardize_fit, normalize


def load_data(asset, dense, snapshot):
    manifest = snapshot.read(asset/'manifest.json')
    report = snapshot.read(dense/'report.json')
    if manifest['status'] != 'COMPLETED' or report['status'] != 'PASS':
        raise ValueError('completed assets/PASS FK required')
    packets = [snapshot.read(g['asset'], g['sha256']) for g in manifest['groups']]
    derived = snapshot.read(dense/'derived_dense_flows.pt', report['artifact_sha256'])
    # A PASS report from another asset snapshot must not be accepted.
    for g in manifest['groups']:
        if report['input_sha256'].get(str(Path(g['asset']).resolve())) != g['sha256']:
            raise ValueError('FK provenance does not match input asset')
    return join_panels(packets, derived)


def summarize(data, predictions, shuffled, folds):
    y = data['y'].cpu()
    metrics = {a:ranking(y, predictions[a]) for a in ARMS}
    shuffle_metrics = {a:ranking(y, shuffled[a]) for a in ARMS}
    degradation = {a:bootstrap_difference(y, predictions[a], shuffled[a], data['panel_keys']) for a in ARMS}
    unique = {a:bootstrap_difference(y, predictions[a], predictions['Direct'], data['panel_keys']) for a in ('Bottleneck', 'Hybrid')}
    coverage = len(data['panel_keys']) >= 100 and metrics['H']['strict_pairs'] >= 100
    gates = {a:coverage and metrics[a]['pairwise_accuracy'] is not None and metrics[a]['pairwise_accuracy'] >= .70
             and metrics[a]['median_top1_regret'] <= .125 and degradation[a]['difference'] > 0
             for a in ('Direct', 'Bottleneck', 'Hybrid')}
    per_fold = {str(f):{a:ranking(y[rows_for(np.flatnonzero(folds == f), 'cpu')], predictions[a][rows_for(np.flatnonzero(folds == f), 'cpu')]) for a in ARMS} for f in range(4)}
    per_anchor = {}
    for key in sorted(set(tuple(k[:2]) for k in data['panel_keys'])):
        ids = rows_for([i for i,k in enumerate(data['panel_keys']) if tuple(k[:2]) == key], 'cpu')
        per_anchor[str(key)] = {a:ranking(y[ids], predictions[a][ids]) for a in ARMS}
    return dict(status='UNCLEAR' if not coverage else ('PROMISING' if any(gates.values()) else 'UNPROMISING'),
                panels=len(data['panel_keys']), anchors=len(per_anchor), gates=gates, metrics=metrics,
                shuffle_metrics=shuffle_metrics, shuffle_degradation=degradation, unique_comparisons=unique,
                unique_contribution={a:'PROMISING' if v['difference'] is not None and v['difference'] >= .03 and v['lower95'] > 0 else 'UNCLEAR' for a,v in unique.items()},
                per_fold=per_fold, per_anchor=per_anchor,
                limits='Post-treatment actual-flow oracle, exposed cohort, four shared solver groups; no learned rolling Z or policy-training gain.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset-run', type=Path, required=True)
    ap.add_argument('--dense-run', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); torch.set_num_threads(2)
    snapshot = Snapshot()
    code = [Path(__file__), ROOT/'src/task/cm-interaction-oracle/src/actual_flow_ranking.py',
            ROOT/'src/task/cm-interaction-oracle/src/oracle_hand_flow.py',
            ROOT/'src/task/cm-interaction-oracle/src/geometric_consequence.py',
            ROOT/'src/task/cm-interaction-oracle/src/ranking_tolerance.py',
            ROOT/'src/task/cm-interaction-oracle/tools/audit/package_rolling_oracle_asset.py']
    for path in code:
        snapshot.hashes[str(path.resolve())] = sha(path)
    protocol = ROOT/'src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-actual-flow-y-ranking.md'
    snapshot.hashes[str(protocol)] = sha(protocol)
    (args.run_dir/'protocol.md').write_bytes(protocol.read_bytes())
    manifest = dict(status='STARTED', git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                    run_id=args.run_dir.name, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    input_sha256=snapshot.hashes, updates=1 if args.smoke else 400,
                    budget_seconds=120 if args.smoke else 600, smoke=args.smoke)
    (args.run_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    try:
        if not torch.cuda.is_available():
            raise ValueError('GPU required')
        cpu = load_data(args.asset_run, args.dense_run, snapshot)
        data = {k:v.to('cuda:0') if isinstance(v, torch.Tensor) else v for k,v in cpu.items()}
        keys = data['panel_keys']; outer = anchor_folds(keys, 4, 271)
        predictions = {a:torch.full_like(cpu['y'], float('nan')) for a in ARMS}
        shuffled = {a:torch.full_like(cpu['y'], float('nan')) for a in ARMS}
        saved = []
        rng = np.random.default_rng(272)
        permutations = torch.tensor(np.stack([rng.permutation(7) for _ in keys]), device='cuda:0')
        flow_shuffle = data['flow'].reshape(-1,7,720).gather(1, permutations[:,:,None].expand(-1,-1,720)).reshape(-1,720)
        for fold in range(4):
            if time.monotonic()-start > manifest['budget_seconds']-20:
                raise TimeoutError('bounded fit deadline')
            source = np.flatnonzero(outer != fold); held = np.flatnonzero(outer == fold)
            record = partition(keys, source, held)
            tr, te = rows_for(source, 'cuda:0'), rows_for(held, 'cuda:0')
            inner = anchor_folds([keys[i] for i in source], 3, 272)
            features = torch.full_like(data['ei'], float('nan')); nested = []
            for f in range(3):
                fit_panels, hold_panels = source[inner != f], source[inner == f]
                ids = rows_for(fit_panels, 'cuda:0'); hold = rows_for(hold_panels, 'cuda:0')
                part = partition(keys, fit_panels, hold_panels)
                out, artifact = physical_fit(data, ids, manifest['updates'])
                features[hold] = out[hold]
                nested.append(dict(**part, **artifact))
            full, physical = physical_fit(data, tr, manifest['updates'])
            features[te] = full[te]
            if not torch.isfinite(features).all():
                raise ValueError('incomplete OOF consequence coverage')
            h, hn = h_inputs(data, tr)
            pn = standardize_fit(features, tr); yn = standardize_fit(data['y'], tr, floor=.05)
            gt_norm = standardize_fit(data['ei'], tr)
            physical_model = Head(h.shape[1]+720,26).to('cuda:0')
            physical_model.load_state_dict(physical['weights']); physical_model.eval()
            with torch.no_grad():
                shuffle_physical = predict(physical_model, torch.cat((h,flow_shuffle),-1))*physical['target_norm']['scale']+physical['target_norm']['mean']
            # full and task H preprocessing share precisely the same fit rows.
            heads = {}
            for arm in ARMS:
                ftr = normalize(data['ei'],gt_norm) if arm == 'GT_EI' else normalize(features,pn)
                x = readout_input(h,data['flow'],ftr,arm)
                model, stats = fit(x, normalize(data['y'],yn), tr, manifest['updates'])
                with torch.no_grad():
                    raw = predict(model,x)*yn['scale']+yn['mean']
                    sx = readout_input(h,flow_shuffle,normalize(shuffle_physical,pn) if arm != 'GT_EI' else ftr,arm)
                    sr = predict(model,sx)*yn['scale']+yn['mean']
                predictions[arm][te.cpu()] = raw[te].cpu(); shuffled[arm][te.cpu()] = sr[te].cpu()
                heads[arm] = dict(weights=model.state_dict(),stats=stats)
            saved.append(dict(**record, inner=nested, physical=physical, h_norms=hn,
                              predicted_norm=pn, gt_norm=gt_norm, y_norm=yn, heads=heads))
            print(json.dumps(dict(fold=fold, held_panels=len(held), elapsed_seconds=time.monotonic()-start)),flush=True)
        for value in list(predictions.values())+list(shuffled.values()):
            if not torch.isfinite(value).all():
                raise ValueError('nonfinite held predictions')
        result = summarize(cpu, predictions, shuffled, outer)
        if args.smoke:
            result['status'] = 'ENGINEERING_SMOKE'
        torch.save(dict(folds=saved, panel_keys=keys, outer_folds=outer, permutations=permutations.cpu(),
                        predictions=predictions, shuffled=shuffled), args.run_dir/'models.pt')
        if (args.run_dir/'models.pt').stat().st_size > 150*1024**2:
            raise ValueError('150MiB predictor cap exceeded')
        for path, expected in snapshot.hashes.items():
            if sha(Path(path)) != expected:
                raise ValueError('input drift: '+path)
        (args.run_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        manifest.update(status='COMPLETED', models_sha256=sha(args.run_dir/'models.pt'),
                        result_sha256=sha(args.run_dir/'result.json'))
        print(json.dumps({k:result[k] for k in ('status','panels','anchors','gates')},indent=2))
    except Exception as exc:
        manifest.update(status='FAILED', error=str(exc)); raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-start
        (args.run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
