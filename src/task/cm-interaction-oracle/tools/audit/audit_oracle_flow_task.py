#!/usr/bin/env python3
"""No-fit GPU reconstruction/replay of the ref12 strict source-OOF task chain."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[5]; TASK = ROOT/'src/task/cm-interaction-oracle'
sys.path[:0] = [str(ROOT), str(TASK/'src'), str(TASK/'tools/run')]
from oracle_hand_flow import measured_flow_inputs, current_state, OracleFlowHead, evaluate
from oracle_flow_task import validate_folds, assemble_crossfit, task_input, TaskHead, evaluate_task, task_summary, TASK_SPECS
from geometric_consequence import NominalSurfaceActions, standardize_fit, normalize
from consequence_sufficiency import readouts
from conditional_consequence import environment_folds
from probe_interventions import sha
from probe_early_hold import dropout_auc


def difference(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        return max(difference(a[k], b[k]) for k in a)
    if isinstance(a, torch.Tensor):
        return float((a.cpu()-b.cpu()).abs().max())
    if isinstance(a, np.ndarray):
        return float(np.max(np.abs(a-b)))
    if a is None or isinstance(a, (str, bool)):
        assert a == b; return 0.
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max(difference(x, y) for x, y in zip(a, b))
    return abs(a-b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--oracle-flow-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(); out = args.run_dir/'engineering_replay.json'
    if out.exists(): raise FileExistsError(out)
    started = time.monotonic(); torch.set_num_threads(2)
    m = json.loads((args.run_dir/'manifest.json').read_text())
    r = json.loads((args.run_dir/'result.json').read_text())
    assert m['run_status'] == 'COMPLETED' and not m['smoke']
    assert sha(args.dataset) == m['dataset_sha256']
    assert sha(args.oracle_flow_run/'diagnostic.pt') == m['oracle_flow_diagnostic_sha256']
    assert sha(args.oracle_flow_run/'manifest.json') == m['oracle_flow_manifest_sha256']
    assert all(sha(ROOT/k) == v for k, v in m['code_sha256'].items())
    assert all(sha(Path(k)) == v for k, v in m['hand_visual_sha256'].items())
    urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    assert sha(urdf) == m['urdf_sha256']
    p = torch.load(args.dataset, map_location='cpu', weights_only=False)
    d = torch.load(args.run_dir/'diagnostic.pt', map_location='cpu', weights_only=False)
    parent = torch.load(args.oracle_flow_run/'diagnostic.pt', map_location='cpu', weights_only=False)
    train, test, clusters = d['train'], d['test'], d['clusters']
    assert np.array_equal(train, parent['train']) and np.array_equal(test, parent['test'])
    assert np.array_equal(clusters, p['episode_id'].numpy()%168)
    folds = environment_folds(train, p['motion_id'].numpy(), clusters, seed=261)
    assert np.array_equal(folds, d['folds']); partitions = validate_folds(train, test, clusters, folds)
    replay = dict(partitions=difference(partitions, d['partitions']))
    dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
    bridge = NominalSurfaceActions(urdf, dev)
    geometry, actions, checks = measured_flow_inputs(p, bridge)
    tr = torch.as_tensor(train, device=dev)
    h, norm = current_state(p, geometry, tr)
    e, i, _, y, _ = readouts(p); z = torch.cat((e, i), -1).to(dev)
    zn = standardize_fit(z, tr); yn = standardize_fit(y.to(dev), tr)
    replay.update(h=difference(h, d['h']), geometry=difference(geometry, d['geometry']), flow=difference(actions['Chunk'], d['flow']),
                  normalizers=difference(norm, d['normalizers']), z=difference(z, d['z']), y=difference(y, d['y']),
                  target_normalizer=difference(zn, d['target_normalizer']), task_target_normalizer=difference(yn, d['task_target_normalizer']))
    # Independently check both physical continuation labels without changing their denominator.
    late_height = p['trajectory'][:, 8:, 2]-p['rest_height'][:, None]
    assert torch.equal(y[:, 6], (late_height < .02).any(1).float())
    assert torch.equal(y[:, 7], (late_height >= .03).float().mean(1))
    full = {}; models = {}
    for name in ('State', 'Chunk'):
        model = OracleFlowHead(h.shape[1]).to(dev); model.load_state_dict(parent['states'][name]); model.eval()
        full[name] = evaluate(model, h, actions[name])*zn['scale']+zn['mean']; models[name] = model
        replay['full_'+name] = difference(full[name], d['full_predictions'][name])
    fold_predictions = {'State': {}, 'Chunk': {}}
    for prefix, part in partitions.items():
        ids = torch.as_tensor(part['fit'], device=dev)
        hf, nf = current_state(p, geometry, ids); sn = standardize_fit(z, ids)
        saved = d['fold_data'][prefix]
        replay[prefix+'_H'] = difference(hf, saved['h'])
        replay[prefix+'_normalizers'] = difference(nf, saved['normalizers'])
        replay[prefix+'_target_normalizer'] = difference(sn, saved['target_normalizer'])
        assert np.array_equal(part['fit'], saved['fit']) and np.array_equal(part['hold'], saved['hold'])
        for name in ('State', 'Chunk'):
            model = OracleFlowHead(hf.shape[1]).to(dev); model.load_state_dict(d['consequence_states'][prefix+'_'+name]); model.eval()
            value = evaluate(model, hf, actions[name])*sn['scale']+sn['mean']
            fold_predictions[name][prefix] = value
            replay[prefix+'_'+name] = difference(value, d['fold_predictions'][name][prefix])
    crossfit = {name: assemble_crossfit(train, test, partitions, fold_predictions[name], full[name]) for name in full}
    replay['crossfit'] = difference(crossfit, d['crossfit'])
    physical = {name: normalize(value, zn) for name, value in crossfit.items()}; physical['GT'] = normalize(z, zn)
    replay['physical'] = difference(physical, d['physical'])
    perm = torch.as_tensor(d['permutation'], device=dev)
    assert np.array_equal(d['permutation'], parent['permutation'])
    shuffled = evaluate(models['Chunk'], h, actions['Chunk'][perm])*zn['scale']+zn['mean']
    replay['flow_shuffled_consequence'] = difference(shuffled, d['flow_shuffled_consequence'])
    physical_shuffled = dict(physical, Chunk=normalize(shuffled, zn))
    predictions = {}; task_models = {}
    for name in TASK_SPECS:
        x = task_input(h, actions['Chunk'], physical, name)
        model = TaskHead(x.shape[1]).to(dev); model.load_state_dict(d['task_states'][name]); model.eval(); task_models[name] = model
        predictions[name] = (evaluate_task(model, x)*yn['scale']+yn['mean']).cpu().numpy()
        replay['task_'+name] = difference(predictions[name], d['predictions'][name])
    for name in ('Flow', 'PredEI', 'Flow_PredEI'):
        x = task_input(h, actions['Chunk'][perm], physical_shuffled, name)
        key = name+'_test_shuffled'
        predictions[key] = (evaluate_task(task_models[name], x)*yn['scale']+yn['mean']).cpu().numpy()
        replay[key] = difference(predictions[key], d['predictions'][key])
    predictions['TrainMean'] = yn['mean'].expand_as(y.to(dev)).cpu().numpy()
    fresh = task_summary(y.numpy(), yn['scale'].cpu().numpy(), predictions, train, test, clusters)
    for name, raw in predictions.items(): fresh['metrics'][name]['physical_failure_auc'] = dropout_auc(raw[test, 6], y.numpy()[test, 6])
    metric_errors = {key: difference(fresh[key], r[key]) for key in ('metrics', 'comparisons', 'oracle_retention', 'gates', 'status', 'unique_added_value')}
    assert max(replay.values()) < 1e-5 and max(metric_errors.values()) < 1e-6
    report = dict(status='PASS', replay_errors=replay, metric_errors=metric_errors, fold_coverage=dict(source=len(train), test=len(test), env_overlap=0),
                  geometry_checks=checks, diagnostic_sha256=sha(args.run_dir/'diagnostic.pt'), audit_sha256=sha(Path(__file__).resolve()),
                  elapsed_seconds=time.monotonic()-started, new_fits=0)
    out.write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(TASK_SPECS); fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key in zip(axes, ('primary_mse', 'physical_failure_mse')):
        ax.barh(names, [r['metrics'][name][key] for name in names]); ax.invert_yaxis(); ax.set_xlabel(key+' / source target scale')
    fig.suptitle('Ref12 post-treatment oracle flow → environment-OOF E/I → Y')
    fig.tight_layout(); fig.savefig(args.run_dir/'oracle_flow_task.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(status='PASS', max_replay=max(replay.values()), elapsed=time.monotonic()-started)))


if __name__ == '__main__': main()
