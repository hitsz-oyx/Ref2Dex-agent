"""Independent NumPy PD/feature/label replay of competent-source and R records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15]
COUPLING = ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05),
            (13, 12, 1.05), (16, 15, .6), (17, 15, .8))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_features(data, ticks, envs, future):
    q, dq, hand, obj, vel = [data[k][ticks, envs] for k in ('q', 'dq', 'hand', 'obj', 'velocity')]
    rotation, center = obj[..., :3, :3], obj[..., :3, 3]
    local = np.einsum('npj,njk->npk', hand - center[:, None], rotation)
    state = np.concatenate((q, dq, local.reshape(-1, 33), obj[:, :3].reshape(-1, 12), vel), -1)
    tau = np.einsum('ntpj,njk->ntpk', future - hand[:, None], rotation).reshape(-1, 24, 33)
    return state, tau


def independent_pd(action, current_q, controller):
    action = action.copy()
    action[..., 6:] = (1 + action[..., 6:]) / 2
    target = np.asarray(controller['offset'], np.float32) + np.asarray(controller['scale'], np.float32) * action
    target[..., :6] += current_q[..., :6]
    for distal, parent, ratio in COUPLING:
        target[..., distal] = target[..., parent] * ratio
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--fit', type=Path)
    p.add_argument('--execution', type=Path)
    args = p.parse_args()
    source_manifest = json.loads((args.source / 'manifest.json').read_text())
    if source_manifest['status'] != 'COMPLETED' or not source_manifest['training_allowed']:
        raise ValueError('completed training-eligible source required')
    if sha(args.source / 'trajectory.npz') != source_manifest['trajectory_sha256']:
        raise ValueError('source data identity mismatch')
    with np.load(args.source / 'trajectory.npz', allow_pickle=False) as f:
        source = {k: f[k].copy() for k in f.files}
    checks = dict(teacher_origin=source_manifest['source_origin'],
                  teacher_checkpoint_sha256=source_manifest['checkpoint_sha256'])
    errors = {}

    def check_native(data, manifest, label):
        controls, n = data['action'].shape[:2]
        if controls != 542 or any(len(data[k]) != 543 for k in ('q', 'dq', 'hand', 'obj', 'velocity')):
            raise ValueError('control/terminal-state boundary mismatch')
        if any(not np.isfinite(v).all() for v in data.values()):
            raise ValueError('nonfinite source record')
        if not np.array_equal(data['action'], data['applied']):
            raise ValueError('applied labels differ from requested action')
        reproduced = independent_pd(data['action'], data['q'][:-1], manifest['native_controller'])
        errors[label + '_pd'] = float(np.max(np.abs(reproduced - data['pd_targets'])))
        if errors[label + '_pd'] > 1e-6:
            raise ValueError('actual native PD target replay mismatch')
        clipped = np.max(np.abs(data['intended'] - data['action']), axis=-1) > 1e-6
        if not np.array_equal(clipped, data['clipped']):
            raise ValueError('clipping labels differ')
        from consequence_evaluator.gate1 import episode_outcome
        outcomes = [episode_outcome(dict(object_pose=data['obj'][:, i],
            object_velocity=data['velocity'][:, i], surface_gap=data['surface_gap'][:, i],
            support_gap=data['support_gap'][:, i], table_footprint=data['table_footprint'][:, i])) for i in range(n)]
        folder = args.source if label == 'source' else args.execution
        saved = json.loads((folder / 'result.json').read_text())['outcomes']
        for actual, expected in zip(outcomes, saved):
            for key in actual:
                if actual[key] != expected[key]:
                    raise ValueError('outcome replay mismatch: ' + key)
        checks[label + '_episodes'] = n

    sys.path[:0] = [str(TASK / 'src'), str(ROOT / 'src/task/consequence-evaluator/src')]
    check_native(source, source_manifest, 'source')
    folder = args.source
    if args.fit or args.execution:
        import torch
        from trajectory_policy.pd_retargeter import load_model
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        # CPU is used for deterministic audit of a tiny retained MLP; no refitting.
        # Batched neural replay uses GPU when available (normal execution here).
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        checks['model_replay_device'] = device
        if args.fit:
            folder = args.fit
            manifest = json.loads((args.fit / 'manifest.json').read_text())
            result = json.loads((args.fit / 'result.json').read_text())
            if manifest['status'] != 'COMPLETED':
                raise ValueError('complete fit required')
            with np.load(args.fit / 'predictions.npz', allow_pickle=False) as f:
                saved_predictions = {k: f[k] for k in f.files}
            arrays = {}
            for name, rows in manifest['groups'].items():
                query = np.arange(519)
                ticks = np.repeat(query, len(rows))
                envs = np.tile(rows, len(query))
                future = np.stack([source['hand'][t + 1:t + 25, e] for t, e in zip(ticks, envs)])
                state, tau = independent_features(source, ticks, envs, future)
                target = source['pd_targets'][ticks, envs][:, ACTIVE].copy()
                target[:, :6] -= source['q'][ticks, envs, :6]
                arrays[name] = state, tau, target
            for arm in ('tau', 'state_only'):
                packet = torch.load(args.fit / (arm + '.pt'), map_location=device, weights_only=False)
                if sha(args.fit / (arm + '.pt')) != result['checkpoint_sha256'][arm]:
                    raise ValueError('selected checkpoint identity mismatch')
                model = load_model(packet, device)
                train = arrays['train']
                for name, value in zip(('state', 'tau', 'target'), train):
                    mean = value.astype(np.float64).mean(0)
                    floor = np.array([.001] * 3 + [.01] * 9) if name == 'target' else .001
                    scale = np.maximum(value.astype(np.float64).std(0), floor)
                    errors[arm + '_' + name + '_mean'] = float(np.max(np.abs(mean - getattr(model, name + '_mean').cpu().numpy())))
                    errors[arm + '_' + name + '_scale'] = float(np.max(np.abs(scale - getattr(model, name + '_scale').cpu().numpy())))
                    if max(errors[arm + '_' + name + '_mean'], errors[arm + '_' + name + '_scale']) > 2e-5:
                        raise ValueError('train-only normalization differs')
                for split, (state, tau, target) in arrays.items():
                    outputs = []
                    with torch.no_grad():
                        for i in range(0, len(state), 512):
                            outputs.append(model(torch.as_tensor(state[i:i + 512], device=device),
                                                 torch.as_tensor(tau[i:i + 512], device=device)).cpu().numpy())
                    prediction = np.concatenate(outputs)
                    key = arm + '_' + split
                    errors[key + '_prediction'] = float(np.max(np.abs(prediction - saved_predictions[key])))
                    mse = float(np.mean(((prediction - target) / model.target_scale.cpu().numpy()) ** 2))
                    errors[key + '_mse'] = abs(mse - result['reports'][arm][split]['normalized_mse'])
                    if errors[key + '_prediction'] > 2e-5 or errors[key + '_mse'] > 2e-5:
                        raise ValueError('independent feature/label/prediction replay mismatch')
                monitor = [json.loads(row) for row in (args.fit / 'monitor.jsonl').read_text().splitlines()]
                selected = min(monitor, key=lambda row: row[arm])['update']
                if selected != packet['selected_update']:
                    raise ValueError('validation-only checkpoint selection differs')
            checks['split'] = manifest['groups']
        if args.execution:
            folder = args.execution
            manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest['status'] != 'COMPLETED':
                raise ValueError('completed execution required')
            with np.load(folder / 'trajectory.npz', allow_pickle=False) as f:
                execution = {k: f[k].copy() for k in f.files}
            check_native(execution, manifest, 'execution')
            paths = [Path(path) for path in manifest['input_sha256'] if path.endswith('/tau.pt')]
            if len(paths) != 1:
                raise ValueError('one actual-PD model required')
            model = load_model(torch.load(paths[0], map_location=device, weights_only=False), device)
            for tick in range(542):
                ids = np.arange(16, 24)
                future = np.moveaxis(source['hand'][np.minimum(tick + np.arange(1, 25), 542)][:, 24:32], 0, 1)
                s, tau = independent_features(execution, np.full(8, tick), ids, future)
                for name, value in [('state', s), ('tau', tau)]:
                    error = np.max(np.abs(value - execution[name + '_features'][tick]))
                    errors[name + '_features'] = max(errors.get(name + '_features', 0.), float(error))
                with torch.no_grad():
                    prediction = model(torch.as_tensor(s, device=device), torch.as_tensor(tau, device=device)).cpu().numpy()
                errors['execution_prediction'] = max(errors.get('execution_prediction', 0.),
                    float(np.max(np.abs(prediction - execution['encoded_target'][tick]))))
                requested = np.zeros((8, 18), np.float32)
                requested[:, ACTIVE] = prediction
                requested[:, :6] += execution['q'][tick, ids, :6]
                for distal, parent, ratio in COUPLING:
                    requested[:, distal] = requested[:, parent] * ratio
                errors['execution_requested_pd'] = max(errors.get('execution_requested_pd', 0.),
                    float(np.max(np.abs(requested - execution['requested_pd'][tick, ids]))))
            if any(errors[k] > 2e-5 for k in ('state_features', 'tau_features', 'execution_prediction', 'execution_requested_pd')):
                raise ValueError('actual R features / output / requested PD mismatch')
    report = dict(**{'pass': True}, checks=checks, max_errors=errors,
        audit_source_sha256=sha(__file__), scope='independent retained data/PD/feature/label replay; not scientific validation')
    (folder / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
