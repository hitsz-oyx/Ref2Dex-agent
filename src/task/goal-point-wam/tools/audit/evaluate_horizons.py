#!/usr/bin/env python3
"""Per-horizon hand EPE on the unchanged frozen object-goal validation panel."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import GoalPointWAM, GoalWindows, atomic_json, digest, loader, move_batch, gpu_snapshot
from oakink_wm.multisource import MixedWindows


@torch.no_grad()
def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    started = time.time()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('artificial_labels') or state.get('checkpoint_kind') != 'goal-point-wam.single.v1':
        raise ValueError('real WAM checkpoint required')
    config, identity = state['config'], state['identity']
    for path, key in ((args.stats, 'stats_sha256'),
                      (args.data / 'processed/manifest.json', 'data_manifest_sha256'),
                      (TASK / 'src/goal_point_wam/model.py', 'model_sha256')):
        if digest(path) != identity[key]:
            raise ValueError('frozen input/model drift: ' + key)
    dataset = GoalWindows(MixedWindows(args.data, 'val'), goal_type='object_points', goal_points=config['goal_points'])
    indices_path = args.checkpoint.parent / 'validation_balanced.npy'
    indices = np.load(indices_path)
    if len(indices) != config['validation_samples']:
        raise ValueError('validation panel size drift')
    model = GoalPointWAM(json.loads(args.stats.read_text()), config).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    names = dataset.source_names
    sums = {n: {k: np.zeros(24, np.float64) for k in ('real', 'static_goal', 'zero', 'velocity')} for n in names}
    counts = {n: np.zeros(24, np.int64) for n in names}
    window_errors = {n: [] for n in names}
    for cpu in loader(dataset, indices, config['microbatch'], 0):
        batch = move_batch(cpu, torch.device('cuda'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            scene = model.scene(batch)
            observed = model.observed_hand_condition(batch)
            def predict(goal):
                tokens, valid = model.goal(batch['start_object_points'], goal, batch['object_goal_valid'])
                return model.trajectory(scene['scene_tokens'], scene['scene_mask'], tokens, valid, observed)
            real = predict(batch['goal_object_points'])
            static = predict(batch['start_object_points'])
        seconds = torch.arange(1, 25, device='cuda', dtype=torch.float32) / 30
        velocity = seconds[None, :, None, None, None] * batch['start_hand_history'][:, None, ..., :3]
        valid = batch['trajectory_valid'].bool()
        errors = {key: torch.linalg.vector_norm(pred.float() - batch['trajectory_target'], dim=-1)
                  for key, pred in (('real', real), ('static_goal', static), ('zero', torch.zeros_like(real)), ('velocity', velocity))}
        for row, source in enumerate(batch['source_id'].tolist()):
            name = names[source]
            mask = valid[row]
            counts[name] += mask.sum((1, 2)).cpu().numpy()
            for key, error in errors.items():
                sums[name][key] += torch.where(mask, error[row], 0).sum((1, 2)).double().cpu().numpy()
            c = int(mask[:8].sum())
            if c:
                window_errors[name].append(float(errors['real'][row, :8][mask[:8]].mean()) * 1000)
    result = {}
    for name in names:
        if not np.all(counts[name] > 0):
            raise ValueError('empty horizon mask')
        groups = {}
        for label, start, end in (('nearest_1_8', 0, 8), ('middle_9_16', 8, 16), ('far_17_24', 16, 24), ('all_1_24', 0, 24)):
            groups[label] = {key + '_epe_mm': float(value[start:end].sum() / counts[name][start:end].sum() * 1000)
                             for key, value in sums[name].items()}
        result[name] = dict(valid_points_per_step=counts[name].tolist(),
                            per_step_mm={k: (v / counts[name] * 1000).tolist() for k, v in sums[name].items()},
                            groups=groups, nearest_window_percentiles_mm=dict(zip(('p25', 'p50', 'p75', 'p90'),
                                np.percentile(window_errors[name], [25, 50, 75, 90]).tolist())))
    macro = {label: {key: float(np.mean([r['groups'][label][key] for r in result.values()]))
                    for key in result[names[0]]['groups'][label]} for label in result[names[0]]['groups']}
    atomic_json(args.output, dict(status='COMPLETED', checkpoint_step=state['step'],
        checkpoint_sha256=digest(args.checkpoint), validation_indices_sha256=digest(indices_path),
        input_identity=identity, panel_windows=len(indices), fps=30,
        future_seconds=(np.arange(1, 25)/30).tolist(), sources=result, macro=macro,
        macro_per_step_real_mm=np.mean([r['per_step_mm']['real'] for r in result.values()], axis=0).tolist(),
        elapsed_seconds=time.time()-started, gpu=gpu_snapshot(torch.device('cuda')),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        audit_script_sha256=digest(Path(__file__)),
        limitations='Frozen single-checkpoint descriptive audit; no uncertainty lower bound or new training.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'stats', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    evaluate(parser.parse_args())
