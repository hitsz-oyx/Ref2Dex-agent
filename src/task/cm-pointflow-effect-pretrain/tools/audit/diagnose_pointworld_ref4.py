#!/usr/bin/env python3
"""Read-only diagnosis of ref4 using the unchanged model's actual input/loss seams."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.pointworld import PointWorldWM, capped_collate
from oakink_wm.data import Windows, balanced_indices
from oakink_wm.model import rigid_points


class CaptureBackbone(nn.Module):
    """Capture the actual adapter input without spending GPU time on PTv3."""
    def __init__(self, **kwargs):
        super().__init__()
        self.inputs = []

    def forward(self, point):
        self.inputs.append({k: v.detach().clone() if torch.is_tensor(v) else v
                            for k, v in point.items()})
        return SimpleNamespace(feat=point['feat'])


def capture_model(stats):
    with patch('oakink_wm.pointworld.PointTransformerV3', CaptureBackbone):
        model = PointWorldWM(stats)
    return model.eval()


def synthetic_stats():
    return dict(flow_mean=np.zeros((24, 3)), flow_std=np.ones((24, 3)) * .02,
                translation_mean=np.zeros((24, 3)), translation_std=np.ones((24, 3)) * .02,
                rotation_scale=np.ones(24), scene_mean=np.zeros(18), scene_std=np.ones(18),
                action_mean=np.zeros(9), action_std=np.ones(9))


def minimal_repro(timestamps=2):
    torch.manual_seed(216)
    model = capture_model(synthetic_stats())
    action = torch.zeros(1, 24, 22, 9)
    action[0, :timestamps, 0, 0] = torch.linspace(.002, .008, timestamps)
    action[0, :timestamps, 0, 3] = action[0, :timestamps, 0, 0]
    action[0, :timestamps, 0, 6] = .00025
    valid = torch.zeros(1, 24, 22, dtype=torch.bool)
    valid[:, :timestamps, 0] = True
    batch = dict(xyz=torch.tensor([[[-.02, 0., 0.]]]), point_valid=torch.ones(1, 1, dtype=torch.bool),
                 features=torch.zeros(1, 1, 18), scene_object=torch.zeros(1, 1, dtype=torch.long),
                 object_valid=torch.ones(1, 1, dtype=torch.bool), object_features=torch.zeros(1, 1, 15),
                 action=action, action_valid=valid)
    with torch.no_grad():
        model(batch)
        permuted_action = action.clone()
        permuted_action[:, :timestamps] = action[:, :timestamps].flip(1)
        model(dict(batch, action=permuted_action))
    first, permuted = model.backbone.inputs
    # Permute full 9D point records while reassigning time embeddings. This
    # isolates position/time association, not physically reversed velocities.
    result = dict(valid_action_points=timestamps, scene_points=1,
                  backbone_rows=len(first['feat']),
                  temporal_records_merged=len(first['feat']) < timestamps + 1,
                  record_permutation_feature_max_difference=float((first['feat'] - permuted['feat']).abs().max()))
    assert result['temporal_records_merged'], 'Expected the current adapter to reproduce temporal merging'
    assert result['record_permutation_feature_max_difference'] < 2e-6

    weights = []
    sigmoid = torch.sigmoid
    def capture_weight(value):
        out = sigmoid(value)
        weights.append(out.detach().clone())
        return out
    slow = torch.eye(4).repeat(1, 1, 24, 1, 1)
    slow[0, 0, :, 0, 3] = torch.arange(1, 25) * (.04 / 24)
    truth = dict(effect=slow, points=torch.zeros(1, 1, 512, 3),
                 object_valid=torch.ones(1, 1, dtype=torch.bool))
    pred = dict(translation=torch.zeros(1, 1, 24, 3), rotation=torch.eye(3).repeat(1, 1, 24, 1, 1))
    with torch.no_grad(), patch('oakink_wm.pointworld.torch.sigmoid', capture_weight):
        loss, terms = model.loss(pred, truth)
    weight = weights[0]
    result.update(slow_window_final_displacement_mm=40., slow_per_frame_displacement_mm=40 / 24,
                  actual_loss_selector_mean=float(weight.mean()),
                  normalized_selector_uniform_share=float(weight[0, 0, 0, 0] / weight.sum()),
                  actual_normalized_point_loss=float(terms['point']))
    target = slow[..., :3, 3] / .02
    unweighted = torch.nn.functional.huber_loss(torch.zeros_like(target), target)
    result['slow_only_weighted_to_unweighted_point_loss_ratio'] = float(terms['point'] / unweighted)
    assert abs(result['slow_only_weighted_to_unweighted_point_loss_ratio'] - 1) < 2e-6
    assert result['actual_loss_selector_mean'] < .04
    return result


def describe(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(a):
        return dict(count=0, mean=None, p10=None, p50=None, p90=None)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite diagnostic values')
    q = np.quantile(a, [.1, .5, .9])
    return dict(count=len(a), mean=float(a.mean()), p10=float(q[0]), p50=float(q[1]), p90=float(q[2]))


def audit(args):
    if not 2 <= args.samples <= 1024 or args.samples % 2:
        raise ValueError('Use an even sample count in [2,1024], matching microbatch=2')
    if args.output.exists():
        raise FileExistsError('Preserve prior diagnostic outputs')
    args.output.mkdir(parents=True)
    runtime_commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    torch.set_num_threads(2)
    torch.manual_seed(216)
    started = time.monotonic()
    dataset = Windows(args.data, 'train')
    # Exactly reuse a prefix of the frozen train-only normalization draw.
    all_indices = balanced_indices(dataset, 4096, 216)
    indices = all_indices[:args.samples]
    stats = json.loads(args.stats.read_text())
    assert hashlib.sha256(all_indices.tobytes()).hexdigest() == stats['index_sha256']
    model = capture_model(stats)
    records, anchor_weights, anchor_steps = [], [], []
    keypoint_totals = np.zeros((2, 11, 3), np.int64)
    timestep_totals = np.zeros((24, 3), np.int64)
    source_files = [Path(__file__).resolve(), TASK/'src/oakink_wm/pointworld.py',
                    TASK/'src/oakink_wm/data.py', TASK/'src/oakink_wm/model.py', args.stats.resolve(),
                    args.data.resolve()/'processed/manifest.json', args.data.resolve()/'processed/index_train.npy']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    completed = True
    for begin in range(0, len(indices), 2):
        if time.monotonic() - started > args.budget_seconds:
            completed = False
            break
        samples = [dataset[int(i)] for i in indices[begin:begin+2]]
        batch = capped_collate(samples)
        captured, captured_weights = [], []
        unique, sigmoid = torch.unique, torch.sigmoid
        def capture_unique(keys, **kwargs):
            value = unique(keys, **kwargs)
            captured.append((keys.detach().clone(), value[0], value[1]))
            return value
        def capture_weight(value):
            out = sigmoid(value)
            captured_weights.append(out.detach().clone())
            return out
        model.backbone.inputs.clear()
        with torch.no_grad(), patch('oakink_wm.pointworld.torch.unique', capture_unique):
            pred = model(batch)
        with torch.no_grad(), patch('oakink_wm.pointworld.torch.sigmoid', capture_weight):
            model.loss(pred, batch)
        assert len(captured) == len(captured_weights) == 1
        keys, unique_keys, inverse = [x.numpy() for x in captured[0]]
        assert np.array_equal(unique_keys[:, 1:], model.backbone.inputs[0]['grid_coord'].numpy())
        weight = captured_weights[0].numpy()
        N = batch['point_valid'].shape[1]
        av = batch['action_valid'].numpy().reshape(2, 24, 2, 11)
        exists = np.concatenate((batch['point_valid'].numpy(), av.reshape(2, 528)), 1)
        raw_action = np.broadcast_to(np.r_[np.full(N, -1), np.arange(528)], exists.shape)[exists]
        raw_batch = np.broadcast_to(np.arange(2)[:, None], exists.shape)[exists]
        batch_weight = float(weight.sum())
        batch_labels = int(batch['object_valid'].sum()) * 24 * 512
        for i, sample in enumerate(samples):
            mask = raw_batch == i
            aid, inv = raw_action[mask], inverse[mask]
            action_inv = inv[aid >= 0]
            action_ids = aid[aid >= 0]
            scene_inv = inv[aid < 0]
            times = action_ids // 22
            uq, counts = np.unique(action_inv, return_counts=True)
            time_pairs = np.unique(np.stack((action_inv, times), 1), axis=0)
            temporal_count = np.bincount(time_pairs[:, 0], minlength=len(unique_keys))
            cross_time = temporal_count[action_inv] > 1
            same_kp = np.zeros((24, 2, 11), bool)
            trajectory_reduction = []
            for hand in range(2):
                for kp in range(11):
                    sel = (action_ids % 22 == hand * 11 + kp)
                    point_inv = action_inv[sel]
                    if not len(point_inv):
                        continue
                    _, kp_inverse, kp_counts = np.unique(point_inv, return_inverse=True, return_counts=True)
                    collided = kp_counts[kp_inverse] > 1
                    same_kp[times[sel], hand, kp] = collided
                    keypoint_totals[hand, kp] += [len(point_inv), int(collided.sum()), len(kp_counts)]
                    trajectory_reduction.append(1-len(kp_counts)/len(point_inv))
            cross_full = np.zeros((24, 2, 11), bool)
            cross_full.reshape(-1)[action_ids] = cross_time
            timestep_totals[:, 0] += av[i].sum((1, 2))
            timestep_totals[:, 1] += same_kp.sum((1, 2))
            timestep_totals[:, 2] += cross_full.sum((1, 2))
            anchor = int(np.flatnonzero(sample['object_features'][:, 13] > .5)[0])
            w = weight[i, anchor].copy()
            effect = batch['effect'][i:i+1, anchor:anchor+1]
            points = batch['points'][i:i+1, anchor:anchor+1]
            actual = rigid_points(effect[..., :3, :3], effect[..., :3, 3], points)
            step = torch.diff(torch.cat((points[:, :, None], actual), 2), dim=2).norm(dim=-1).numpy()[0, 0]
            centroid = points.mean(-2)
            centered = points - centroid[..., None, :]
            rotation_flow = torch.einsum('bmtij,bmpj->bmtpi', effect[..., :3, :3], centered) - centered[:, :, None]
            rotational_rms = float(rotation_flow.square().sum(-1).mean().sqrt())
            center_future = torch.einsum('bmtij,bmj->bmti', effect[..., :3, :3], centroid) + effect[..., :3, 3]
            translational_rms = float((center_future-centroid[:, :, None]).square().sum(-1).mean().sqrt())
            rotation = effect[0, 0, :, :3, :3].numpy()
            angle = np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2)-1)/2, -1, 1))
            sid, object_id, tick = map(int, sample['sample_id'])
            data = dataset.sequence(dataset.sequences[sid])
            objects = len(sample['points'])
            window_weight = float(weight[i, :objects].sum())
            denominator = batch_weight / batch_labels
            record = dict(index=int(indices[begin+i]), sample_id=sample['sample_id'].tolist(),
                          category=int(sample['category']), moving=int(sample['category']) == 0,
                          program=bool(data['program'][tick, object_id]),
                          hand_presence=sample['hand_presence'].tolist(), objects=objects,
                          valid_action_points=len(action_inv), unique_action_voxels=len(uq),
                          unique_action_ratio=len(uq)/len(action_inv),
                          scene_action_shared_voxels=len(np.intersect1d(uq, scene_inv)),
                          unique_time_action_voxels=len(time_pairs),
                          cross_time_voxel_point_fraction=float(cross_time.mean()),
                          same_keypoint_collision_fraction=float(same_kp[av[i]].mean()),
                          mean_same_keypoint_token_reduction=float(np.mean(trajectory_reduction)),
                          palm_collision_fraction=float(same_kp[:, :, :6][av[i, :, :, :6]].mean()),
                          fingertip_collision_fraction=float(same_kp[:, :, 6:][av[i, :, :, 6:]].mean()),
                          anchor_mean_raw_weight=float(w.mean()),
                          anchor_relative_uniform_weight=float(w.mean()/denominator),
                          window_relative_uniform_weight=window_weight/(objects*24*512)/denominator,
                          anchor_normalized_microbatch_share=float(w.sum()/batch_weight),
                          anchor_unweighted_microbatch_share=24*512/batch_labels,
                          anchor_frame_fraction_over_5mm=float((step >= .005).mean()),
                          anchor_radius_mm=float(sample['radius'][anchor]*1000),
                          max_rotation_rad=float(angle.max()),
                          max_origin_translation_mm=float(effect[..., :3, 3].norm(dim=-1).max()*1000),
                          rotation_dominant=bool(angle.max() > .02 and rotational_rms >= 2*translational_rms),
                          rotation_origin_static=bool(angle.max() > .02 and effect[..., :3, 3].norm(dim=-1).max() <= .002))
            records.append(record)
            anchor_weights.append(w.reshape(-1))
            anchor_steps.append(step.reshape(-1))
        if begin % 64 == 0:
            print(json.dumps(dict(windows=len(records), seconds=time.monotonic()-started)), flush=True)
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise RuntimeError('Diagnostic inputs changed')
    group_masks = dict(all=lambda x: True, moving=lambda x: x['moving'], static=lambda x: not x['moving'],
                       program=lambda x: x['program'], moving_program=lambda x: x['moving'] and x['program'],
                       single_hand=lambda x: sum(x['hand_presence']) == 1,
                       bimanual=lambda x: sum(x['hand_presence']) == 2,
                       rotation_dominant=lambda x: x['rotation_dominant'],
                       rotation_origin_static=lambda x: x['rotation_origin_static'])
    fields = ['valid_action_points','unique_action_voxels','unique_action_ratio','unique_time_action_voxels',
              'cross_time_voxel_point_fraction','same_keypoint_collision_fraction','mean_same_keypoint_token_reduction',
              'palm_collision_fraction','fingertip_collision_fraction','scene_action_shared_voxels',
              'anchor_mean_raw_weight','anchor_relative_uniform_weight','window_relative_uniform_weight',
              'anchor_frame_fraction_over_5mm','anchor_radius_mm']
    groups = {}
    for name, predicate in group_masks.items():
        chosen = [i for i,r in enumerate(records) if predicate(r)]
        groups[name] = dict(windows=len(chosen), statistics={f:describe([records[i][f] for i in chosen]) for f in fields})
        groups[name]['anchor_point_weights'] = describe(np.concatenate([anchor_weights[i] for i in chosen]) if chosen else [])
        groups[name]['anchor_adjacent_point_displacement_mm'] = describe(np.concatenate([anchor_steps[i]*1000 for i in chosen]) if chosen else [])
    result = dict(schema='pointworld-ref4-input-loss-audit.v1', status='COMPLETED' if completed else 'BUDGET_STOP',
                  git_commit=runtime_commit,
                  input_hashes=hashes, split='train', seed=216, draw='prefix of frozen balanced 4096-window normalization draw',
                  sample_index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(), requested_samples=args.samples,
                  completed_samples=len(records), unique_windows=len({tuple(r['sample_id']) for r in records}),
                  sequences=len({r['sample_id'][0] for r in records}), seconds=time.monotonic()-started,
                  device='cpu', cpu_reason='Pure voxel/label statistics; PTv3 replaced by a capture seam, no learned predictor inference or fitting',
                  microbatch=2, groups=groups, minimal_repro=minimal_repro(),
                  per_keypoint=[], per_timestep=[])
    for hand in range(2):
        for kp in range(11):
            n, collision, unique_n = keypoint_totals[hand,kp]
            result['per_keypoint'].append(dict(hand=['right','left'][hand],keypoint=kp,
                         kind='wrist_mcp' if kp<6 else 'fingertip',valid_points=int(n),
                         same_keypoint_collision_fraction=float(collision/n) if n else None,
                         retained_trajectory_fraction=float(unique_n/n) if n else None))
    for t,(n,same,cross) in enumerate(timestep_totals):
        result['per_timestep'].append(dict(timestep=t+1,valid_points=int(n),
                          same_keypoint_collision_fraction=float(same/n) if n else None,
                          cross_time_voxel_point_fraction=float(cross/n) if n else None))
    (args.output/'windows.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],samples=len(records),seconds=result['seconds'],output=str(args.output))),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repro', action='store_true')
    parser.add_argument('--assert-no-time-merging', action='store_true')
    parser.add_argument('--repro-timesteps', type=int, choices=(2,24), default=2)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--stats', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--budget-seconds', type=int, default=300)
    args = parser.parse_args()
    if not args.repro:
        if not all((args.data,args.stats,args.output)):
            parser.error('Choose --repro or supply --data --stats --output')
        return audit(args)
    result = minimal_repro(args.repro_timesteps)
    print(json.dumps(result, indent=2))
    if args.assert_no_time_merging:
        assert not result['temporal_records_merged'], 'Different action timesteps were merged before PTv3'


if __name__ == '__main__':
    main()
