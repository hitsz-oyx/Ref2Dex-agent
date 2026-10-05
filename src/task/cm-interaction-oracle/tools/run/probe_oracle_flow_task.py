#!/usr/bin/env python3
"""Ref12 oracle actual-flow to environment-OOF consequences to task Y Probe."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT/'src/task/cm-interaction-oracle'
sys.path[:0] = [str(ROOT), str(TASK/'src'), str(TASK/'tools/run')]
from oracle_hand_flow import current_state, measured_flow_inputs, OracleFlowHead, evaluate, fit
from oracle_flow_task import validate_folds, assemble_crossfit, task_input, fit_task, TASK_SPECS, task_summary, evaluate_task
from conditional_consequence import environment_folds
from consequence_sufficiency import readouts, HEADS
from geometric_consequence import NominalSurfaceActions, standardize_fit, normalize
from probe_oracle_hand_flow import EXPECTED_SHA
from probe_geometric_consequence import cpu_tree
from probe_interventions import sha
from probe_early_hold import dropout_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--oracle-flow-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); torch.set_num_threads(2)
    def save(name, value): (args.run_dir/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    parent = json.loads((args.oracle_flow_run/'manifest.json').read_text())
    paths = [Path(__file__).resolve(), TASK/'src/oracle_flow_task.py', TASK/'tools/run/probe_early_hold.py']
    code = dict(parent['code_sha256'])
    code.update({str(path.relative_to(ROOT)): sha(path) for path in paths})
    manifest = dict(run_status='STARTED', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                    command=sys.argv, created_at=datetime.now(timezone.utc).isoformat(), physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    dataset_sha256=sha(args.dataset), oracle_flow_run=str(args.oracle_flow_run),
                    oracle_flow_manifest_sha256=sha(args.oracle_flow_run/'manifest.json'),
                    oracle_flow_diagnostic_sha256=sha(args.oracle_flow_run/'diagnostic.pt'),
                    code_sha256=code, hand_visual_sha256=parent['hand_visual_sha256'], urdf_sha256=parent['urdf_sha256'],
                    seeds=[255, 256, 259, 260, 261], consequence_updates=1 if args.smoke else 300,
                    task_updates=1 if args.smoke else 500, new_consequence_fits=6, new_task_fits=6,
                    reused_full_consequence_models=['State', 'Chunk'], source_OOF=True, smoke=args.smoke,
                    new_simulation=False, execution_predictor_used=False, paired_collection=False, torch_version=torch.__version__)
    save('manifest.json', manifest)
    try:
        assert manifest['dataset_sha256'] == EXPECTED_SHA and parent['run_status'] == 'COMPLETED' and not parent['smoke']
        assert parent['updates'] == 300 and parent['git_commit'] == '452d7e5dc7d8ad39a1769e3aed13724a4c398aad'
        assert all(sha(ROOT/k) == v for k, v in code.items())
        assert all(sha(Path(k)) == v for k, v in parent['hand_visual_sha256'].items())
        urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
        assert sha(urdf) == parent['urdf_sha256']
        p = torch.load(args.dataset, map_location='cpu', weights_only=False)
        old = torch.load(args.oracle_flow_run/'diagnostic.pt', map_location='cpu', weights_only=False)
        train, test, clusters = old['train'], old['test'], old['clusters']
        assert np.array_equal(clusters, p['episode_id'].numpy()%168)
        folds = environment_folds(train, p['motion_id'].numpy(), clusters, seed=261)
        partitions = validate_folds(train, test, clusters, folds)
        e, i, _, y, details = readouts(p); z = torch.cat((e, i), -1)
        assert torch.equal(z, old['z']) and details['risk'].all()
        dev = torch.device('cuda:0'); torch.cuda.set_device(dev); torch.cuda.reset_peak_memory_stats(dev)
        bridge = NominalSurfaceActions(urdf, dev)
        geometry, actions, checks = measured_flow_inputs(p, bridge)
        tr = torch.as_tensor(train, device=dev); te = torch.as_tensor(test, device=dev)
        h, norm = current_state(p, geometry, tr)
        assert torch.equal(h.cpu(), old['h']) and torch.equal(geometry.cpu(), old['geometry'])
        assert torch.equal(actions['Chunk'].cpu(), old['actions']['Chunk'])
        zg = z.to(dev); z_norm = standardize_fit(zg, tr)
        assert torch.equal(z_norm['mean'].cpu(), old['target_mean']) and torch.equal(z_norm['scale'].cpu(), old['target_scale'])
        yg = y.to(dev); y_norm = standardize_fit(yg, tr); yn = normalize(yg, y_norm)
        permutation = old['permutation']; perm = torch.as_tensor(permutation, device=dev)
        assert set(permutation[train]) == set(train) and set(permutation[test]) == set(test)
        full_predictions = {}; full_models = {}; inherited_errors = {}
        for name in ('State', 'Chunk'):
            model = OracleFlowHead(h.shape[1]).to(dev); model.load_state_dict(old['states'][name]); model.eval()
            value = evaluate(model, h, actions[name])*z_norm['scale']+z_norm['mean']
            inherited_errors[name] = float(np.abs(value.cpu().numpy()-old['predictions'][name]).max())
            assert inherited_errors[name] < 1e-5
            full_predictions[name] = value; full_models[name] = model
        flow_shuffled = evaluate(full_models['Chunk'], h, actions['Chunk'][perm])*z_norm['scale']+z_norm['mean']
        inherited_errors['Chunk_test_shuffled'] = float(np.abs(flow_shuffled.cpu().numpy()-old['predictions']['Chunk_test_shuffled']).max())
        assert inherited_errors['Chunk_test_shuffled'] < 1e-5
        records, states, fold_data, fold_predictions = {}, {}, {}, {'State': {}, 'Chunk': {}}
        for prefix, part in partitions.items():
            ids = torch.as_tensor(part['fit'], device=dev)
            hf, nf = current_state(p, geometry, ids)
            target_norm = standardize_fit(zg, ids); target = normalize(zg, target_norm)
            fold_data[prefix] = dict(**part, h=hf.cpu(), normalizers=cpu_tree(nf), target_normalizer=cpu_tree(target_norm))
            for name in ('State', 'Chunk'):
                model, value, record = fit(hf, actions[name], target, ids, manifest['consequence_updates'])
                raw = value*target_norm['scale']+target_norm['mean']
                key = prefix+'_'+name; records[key] = record; states[key] = cpu_tree(model.state_dict())
                fold_predictions[name][prefix] = raw
            print(json.dumps(dict(completed=prefix, fit=len(part['fit']), hold=len(part['hold']))), flush=True)
        crossfit = {name: assemble_crossfit(train, test, partitions, fold_predictions[name], full_predictions[name])
                    for name in ('State', 'Chunk')}
        physical = {name: normalize(value, z_norm) for name, value in crossfit.items()}
        physical['GT'] = normalize(zg, z_norm)
        physical_shuffled = dict(physical, Chunk=normalize(flow_shuffled, z_norm))
        predictions, task_states, task_models = {}, {}, {}
        for name in TASK_SPECS:
            x = task_input(h, actions['Chunk'], physical, name)
            model, pred, record = fit_task(x, yn, tr, manifest['task_updates'])
            records['task_'+name] = record; task_states[name] = cpu_tree(model.state_dict()); task_models[name] = model
            predictions[name] = (pred*y_norm['scale']+y_norm['mean']).cpu().numpy()
            print(json.dumps(dict(completed='task_'+name, source_all8_mse=record['source_all8_mse'])), flush=True)
        assert len({records['task_'+name]['initial_hash'] for name in TASK_SPECS}) == 1
        for name in ('Flow', 'PredEI', 'Flow_PredEI'):
            x = task_input(h, actions['Chunk'][perm], physical_shuffled, name)
            pred = evaluate_task(task_models[name], x)
            predictions[name+'_test_shuffled'] = (pred*y_norm['scale']+y_norm['mean']).cpu().numpy()
        predictions['TrainMean'] = y_norm['mean'].expand_as(yg).cpu().numpy()
        result = task_summary(y.numpy(), y_norm['scale'].cpu().numpy(), predictions, train, test, clusters)
        quality = {}
        for name, value in crossfit.items():
            error = ((value-zg)/z_norm['scale']).square()
            in_sample = ((full_predictions[name]-zg)/z_norm['scale']).square()
            quality[name] = {subset: dict(E_mse=float(err[:, :12].mean()), I_mse=float(err[:, 12:].mean()))
                             for subset, err in (('source_OOF', error[tr]), ('test_full_source', error[te]),
                                                 ('source_full_in_sample_diagnostic_only', in_sample[tr]))}
        result.update(consequence_quality=quality, target_heads=list(HEADS), geometry_checks=checks,
                      inherited_replay_errors=inherited_errors, source_windows=len(train), test_windows=len(test),
                      held_environments=len(np.unique(clusters[test])), engineering_only=args.smoke,
                      elapsed_seconds=time.monotonic()-started, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(dev))
        for name, raw in predictions.items():
            result['metrics'][name]['physical_failure_auc'] = dropout_auc(raw[test, 6], y.numpy()[test, 6])
        if args.smoke: result['status'] = 'UNCLEAR'
        save('result.json', result); save('fit_records.json', records)
        torch.save(dict(train=train, test=test, clusters=clusters, folds=folds, partitions=partitions,
                        h=h.cpu(), normalizers=cpu_tree(norm), geometry=geometry.cpu(), flow=actions['Chunk'].cpu(),
                        z=z, y=y, target_normalizer=cpu_tree(z_norm), task_target_normalizer=cpu_tree(y_norm),
                        fold_data=fold_data, consequence_states=states,
                        fold_predictions=cpu_tree(fold_predictions), crossfit=cpu_tree(crossfit),
                        full_predictions=cpu_tree(full_predictions), flow_shuffled_consequence=flow_shuffled.cpu(),
                        physical=cpu_tree(physical), task_states=task_states, predictions=predictions,
                        permutation=permutation), args.run_dir/'diagnostic.pt')
        assert all(sha(ROOT/k) == v for k, v in code.items()) and sha(args.dataset) == EXPECTED_SHA
        assert sha(args.oracle_flow_run/'diagnostic.pt') == manifest['oracle_flow_diagnostic_sha256']
        assert sha(args.oracle_flow_run/'manifest.json') == manifest['oracle_flow_manifest_sha256']
        assert sum(f.stat().st_size for f in args.run_dir.rglob('*') if f.is_file()) < 30*1024**2
        manifest['run_status'] = 'COMPLETED'
        print(json.dumps(dict(status=result['status'], gates=result['gates'], unique_added_value=result['unique_added_value'],
                              oracle_retention=result['oracle_retention'], metrics=result['metrics'])), flush=True)
    except BaseException as error:
        manifest.update(run_status='FAILED', failure=f'{type(error).__name__}: {error}'); raise
    finally:
        manifest.update(completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
        save('manifest.json', manifest)


if __name__ == '__main__': main()
