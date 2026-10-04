#!/usr/bin/env python3
"""Ref5 K=1: frozen spatial Cmv2 E into a matched, episode-split G bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_ref4_g_bridge import Bridge, _split, _standardize
from probe_cmv2_unified_ei import read_obj
from src.task.CmResidual.surface_execution import area_hand_samples
from src.task.CmResidual.v118_planner import TorchInspireKinematics
from src.task.CmResidual.dexplore_cm_geometry import (
    native_joint_limits, dexplore_action_to_native_targets, dexplore_root_pose,
)
from src.task.ObjectInteractionCmv2.model import ObjectInteractionCmv2V13Model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pose_effect(effect):
    """13D legacy E to 6D translation/axis-angle in the decision object frame."""
    quat = nn.functional.normalize(effect[..., 3:7], dim=-1)
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    norm = quat[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, quat[..., 3:4])
    aa = quat[..., :3] * torch.where(norm > 1e-7, angle / norm.clamp_min(1e-7), 2.0)
    return torch.cat((effect[..., :3], aa), -1)


def subset_rows(source, per_episode):
    keys = torch.stack((source['source_run'], source['episode_id']), -1)
    selected = []
    for key in torch.unique(keys, dim=0):
        idx = (keys == key).all(-1).nonzero().flatten()
        take = torch.linspace(0, len(idx) - 1, min(per_episode, len(idx))).round().long()
        selected.append(idx[take])
    return torch.cat(selected).sort().values


@torch.no_grad()
def predict_effect(source, checkpoint, device, batch_size, geometry_seed):
    assets = ROOT / 'third_party/DExplore/dexplore/data/assets'
    urdf = assets / 'inspire_hand_new/inspire_hand_right.urdf'
    obj_path = assets / 'mjcf/objects/airplane/airplane.obj'
    vertices, normals = read_obj(obj_path)
    ids = np.sort(np.random.default_rng(geometry_seed).choice(len(vertices), 256, replace=False))
    obj, objn = torch.tensor(vertices[ids], device=device), torch.tensor(normals[ids], device=device)
    surface = area_hand_samples(urdf, count=512, seed=geometry_seed)
    hp = torch.tensor(surface['points'], device=device)
    hn = torch.tensor(surface['normals'], device=device)
    links = torch.tensor(surface['links'], device=device, dtype=torch.long)
    fk = TorchInspireKinematics(urdf, device)
    lower, upper = native_joint_limits(urdf, device)
    cfg = SimpleNamespace(hidden_width=64, num_tokens=8, use_residual=False,
                          interaction_mode='swept', feature_scale_m=0.02,
                          knn_k=16, interaction_radius_m=0.02, frame_dt_s=1 / 30)
    model = ObjectInteractionCmv2V13Model(cfg).to(device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = saved.get('model', saved)
    if any(key.startswith('base.') for key in state):
        state = {key[5:]: val for key, val in state.items() if key.startswith('base.')}
    model.load_state_dict(state, strict=True)
    model.eval()
    predictions = []
    n = len(source['state'])
    for start in range(0, n, batch_size):
        current = source['state'][start:start + batch_size].to(device)
        action = source['action'][start:start + batch_size].to(device)
        q = current[:, :18]
        pose = dexplore_root_pose(current[:, 36:49])
        native_links = fk.forward(q[:, None])[:, 0]
        target_q = dexplore_action_to_native_targets(action, q, lower, upper)
        future_links = fk.forward(target_q[:, None])[:, 0]

        def hand(link_poses, points):
            return torch.einsum('bpij,pj->bpi', link_poses[:, links, :3, :3], points)

        world = hand(native_links, hp) + native_links[:, links, :3, 3]
        world_next = hand(future_links, hp) + future_links[:, links, :3, 3]
        rotation = pose[:, :3, :3].transpose(-1, -2)
        local = lambda value: torch.einsum('bij,bpj->bpi', rotation, value)
        count = len(q)
        batch = {'obj_points': obj[None].expand(count, -1, -1),
                 'obj_normals': objn[None].expand(count, -1, -1),
                 'hand_points': local(world - pose[:, None, :3, 3]),
                 'hand_normals': local(hand(native_links, hn)),
                 'hand_flow': local(world_next - world),
                 'hand_valid_mask': torch.ones(count, len(hp), device=device, dtype=torch.bool),
                 'delta_time_s': torch.full((count,), 1 / 30, device=device)}
        prediction = model(batch)['delta_xi_root'].cpu()[:, None]
        if not torch.isfinite(prediction).all():
            raise ValueError('nonfinite Cmv2 E')
        predictions.append(prediction)
    return torch.cat(predictions), {'urdf_sha256': sha256(urdf),
                                   'object_sha256': sha256(obj_path),
                                   'geometry_seed': geometry_seed, 'object_anchors': 256,
                                   'hand_surface_points': 512,
                                   'input_permission': 'current state + current action only; nominal FK sweep',
                                   'root_contract': 'identity hand root, verified on native trace; no future state input'}


def fit_bridge(history, consequence, target, train, device, epochs, batch_size, seed):
    torch.manual_seed(seed)
    model = Bridge({'H': history.shape[-1], 'E': consequence.shape[-1]}, ['H', 'E']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    ym, ys = target[train].mean().to(device), target[train].std(unbiased=False).clamp_min(1e-6).to(device)
    idx_train = train.nonzero().flatten()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for epoch in range(epochs):
        model.train()
        order = idx_train[torch.randperm(len(idx_train), generator=generator)]
        total = 0.0
        for idx in order.split(batch_size):
            pred = model({'H': history[idx].to(device), 'E': consequence[idx].to(device)})
            loss = nn.functional.mse_loss(pred, (target[idx].to(device) - ym) / ys)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        losses.append(total / len(idx_train))
    return model, ym, ys, losses


@torch.no_grad()
def evaluate(model, history, consequence, target, test, group_keys, groups, device, ym, ys, batch_size):
    model.eval()
    pred = torch.cat([(model({'H': history[idx].to(device), 'E': consequence[idx].to(device)}) * ys + ym).cpu()
                      for idx in test.nonzero().flatten().split(batch_size)])
    error = (pred - target[test]).abs()
    test_keys = group_keys[test]
    group_mae = [float(error[(test_keys == torch.tensor(g)).all(-1)].mean()) for g in groups]
    return {'test_mae': float(error.mean()), 'episode_balanced_mae': float(np.mean(group_mae)),
            'test_group_mae': group_mae, 'parameter_count': sum(p.numel() for p in model.parameters())}, pred


def comparison(base, arm, seed):
    delta = np.array(base['test_group_mae']) - np.array(arm['test_group_mae'])
    sampled = np.random.default_rng(seed).choice(delta, (2000, len(delta)), replace=True).mean(1)
    return {'absolute_mae_reduction': float(delta.mean()),
            'relative_mae_reduction': float(delta.mean() / base['episode_balanced_mae']),
            'descriptive_episode_ci95': np.quantile(sampled, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'tmp/e260_all4_h16_histfix.pt')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'tmp/P-20261004-cmv2-unified-ei-k1-v7.best1.pt')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--rows-per-episode', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--inference-batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=201)
    parser.add_argument('--split-seed', type=int, default=20261004)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    start = time.monotonic()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This model-compute probe requires an available GPU')
    torch.set_num_threads(2)
    source = torch.load(args.input, map_location='cpu', weights_only=False)
    idx = subset_rows(source, args.rows_per_episode)
    data = {key: value[idx] if torch.is_tensor(value) and len(value) == len(source['state']) else value
            for key, value in source.items()}
    del source
    raw = {'_namespace': data.get('source_namespace', torch.zeros_like(data['source_run'])),
           '_run': data['source_run'], '_episode': data['episode_id']}
    train, test, train_groups, test_groups = _split(raw, args.split_seed)
    keys = torch.stack([raw[k] for k in ('_namespace', '_run', '_episode')], -1)
    gt = pose_effect(data['effect'][:, :1])
    pred, geometry = predict_effect(data, args.checkpoint, device, args.inference_batch_size, args.split_seed)
    history_raw = torch.cat([data[k].float() for k in ('history_state', 'history_previous_action', 'history_context', 'history_progress')], -1)
    history = _standardize(history_raw, train)
    mean, scale = gt[train].mean((0, 1)), gt[train].std((0, 1), unbiased=False).clamp_min(1e-6)
    egt, epred = (gt - mean) / scale, (pred - mean) / scale
    arms = {'H': torch.zeros_like(egt), 'HE_GT': egt, 'HE_CM_fit': epred,
            'HA': _standardize(data['action'].float()[:, None], train)}
    target = data['return_to_go'].float()
    results, predictions, checkpoints = {}, {}, {}
    for name, consequence in arms.items():
        model, ym, ys, losses = fit_bridge(history, consequence, target, train, device, args.epochs, args.batch_size, args.seed)
        result, prediction = evaluate(model, history, consequence, target, test, keys, test_groups, device, ym, ys, args.batch_size)
        results[name] = {**result, 'training_loss': losses}
        predictions[name] = prediction
        checkpoints[name] = {'model': {k: v.cpu() for k, v in model.state_dict().items()}, 'target_mean': float(ym), 'target_scale': float(ys)}
        if name == 'HE_GT':
            result, prediction = evaluate(model, history, epred, target, test, keys, test_groups, device, ym, ys, args.batch_size)
            results['HE_CM_swap'] = result
            predictions['HE_CM_swap'] = prediction
        print(json.dumps({'arm': name, **result}), flush=True)
    metrics = {}
    for name, value in [('CM', pred), ('train_mean', mean.expand_as(gt)), ('zero', torch.zeros_like(gt))]:
        error = value[test] - gt[test]
        metrics[name] = {'translation_rmse_m': float(error[..., :3].square().mean().sqrt()),
                         'axis_angle_rmse_rad': float(error[..., 3:].square().mean().sqrt()),
                         'train_standardized_rmse': float((error / scale).square().mean().sqrt())}
    comparisons = {name: comparison(results['H'], results[name], args.seed)
                   for name in results if name != 'H'}
    comparisons['HE_CM_fit_vs_HA'] = comparison(results['HA'], results['HE_CM_fit'], args.seed)
    gt_gain = comparisons['HE_GT']['relative_mae_reduction']
    cm_gain = comparisons['HE_CM_fit']['relative_mae_reduction']
    retained = comparisons['HE_CM_fit']['absolute_mae_reduction'] / max(comparisons['HE_GT']['absolute_mae_reduction'], 1e-8)
    status = ('PROMISING' if gt_gain >= .05 and cm_gain >= .03 and retained >= .25
              and comparisons['HE_CM_fit_vs_HA']['relative_mae_reduction'] > 0
              else 'UNCLEAR' if gt_gain < .05 else 'UNPROMISING')
    report = {'schema': 'ref2dex.pointflow_g_probe.v1', 'status': status,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'run_id': args.output_dir.name, 'task': 'cm-interaction-oracle',
              'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'input_sha256': sha256(args.input), 'checkpoint_sha256': sha256(args.checkpoint),
              'script_sha256': sha256(__file__), 'geometry': geometry, 'train_groups': train_groups,
              'test_groups': test_groups, 'train_rows': int(train.sum()), 'test_rows': int(test.sum()),
              'effect_metrics': metrics, 'arms': results, 'comparisons': comparisons,
              'gt_gain_retained': retained if gt_gain > 0 else None,
              'elapsed_seconds': time.monotonic() - start,
              'decision_rule': 'GT>=5%, predicted-fit>=3%, retains>=25% GT gain and beats HA; otherwise do not enter policy training',
              'limitations': ['single model seed; descriptive episode bootstrap, four source runs',
                              'GT E is a future oracle; offline G is not policy utility',
                              'checkpoint chosen on a separate native trace; no new predictor tuning',
                              'K1 pose-only E, corrected history and subsample; not comparable to old K8 raw G MAEs',
                              'H uses same HE architecture with constant E; HA has a larger action encoder']}
    (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    torch.save({'indices': idx, 'predicted_effect': pred, 'gt_effect': gt, 'test_predictions': predictions,
                'target': target[test], 'test_keys': keys[test], 'checkpoints': checkpoints,
                'history_mean': history_raw[train].mean((0, 1)),
                'history_scale': history_raw[train].std((0, 1), unbiased=False).clamp_min(1e-6),
                'effect_mean': mean, 'effect_scale': scale}, args.output_dir / 'bridge.pt')
    print(json.dumps({'status': status, 'comparisons': comparisons, 'effect_metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
