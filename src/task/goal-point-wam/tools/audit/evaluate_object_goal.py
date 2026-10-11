#!/usr/bin/env python3
"""Frozen object-goal validation ablation with identical cached scene tokens."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import GoalPointWAM, GoalWindows, atomic_json, digest, loader, move_batch, gpu_snapshot
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES


def geometric_baselines(batch):
    """Goal-only geometric references, not physically validated hand actions."""
    valid = batch['object_goal_valid']
    count = valid.sum(1)
    if (count < 3).any(): raise ValueError('rigid goal reference needs at least three paired points')
    start = torch.where(valid[..., None], batch['start_object_points'], 0.)
    goal = torch.where(valid[..., None], batch['goal_object_points'], 0.)
    start_center, goal_center = start.sum(1)/count[:,None], goal.sum(1)/count[:,None]
    a = torch.where(valid[..., None], start-start_center[:,None], 0.)
    b = torch.where(valid[..., None], goal-goal_center[:,None], 0.)
    u, _, vh = torch.linalg.svd(a.transpose(1,2) @ b)
    correction = torch.ones_like(start_center)
    correction[:,-1] = torch.linalg.det(vh.transpose(1,2) @ u.transpose(1,2))
    rotation = vh.transpose(1,2) @ torch.diag_embed(correction) @ u.transpose(1,2)
    translation = goal_center - (rotation @ start_center[...,None]).squeeze(-1)
    hand = batch['start_hand']
    endpoint_delta = (rotation[:,None,None] @ hand[...,None]).squeeze(-1) + translation[:,None,None] - hand
    phase = torch.arange(1,25,device=hand.device,dtype=hand.dtype)[None,:,None,None,None]/24
    return dict(object_translation=phase*(goal_center-start_center)[:,None,None,None],
                object_rigid_follow=phase*endpoint_delta[:,None])


@torch.no_grad()
def evaluate(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state.get('artificial_labels') or state.get('checkpoint_kind') != 'goal-point-wam.single.v1':
        raise ValueError('real-data checkpoint required; artificial diagnostic weights are prohibited')
    config, identity = state['config'], state['identity']
    if config.get('goal_type') != 'object_points': raise ValueError('object-goal model required')
    for actual, expected in ((digest(args.stats), identity['stats_sha256']),
                            (digest(args.data / 'processed/manifest.json'), identity['data_manifest_sha256']),
                            (digest(TASK / 'src/goal_point_wam/model.py'), identity['model_sha256'])):
        if actual != expected: raise ValueError('frozen model/data/stats identity drift')
    base = MixedWindows(args.data, 'val')
    dataset = GoalWindows(base, goal_type='object_points', goal_points=config['goal_points'])
    indices_path = args.checkpoint.parent / 'validation_balanced.npy'
    indices = np.load(indices_path)
    if len(indices) != config['validation_samples']: raise ValueError('validation panel size drift')
    model = GoalPointWAM(json.loads(args.stats.read_text()), config).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    totals = {name: dict(count=0, real=0., static_goal=0., zero=0., object_translation=0., object_rigid_follow=0.,
                         endpoint_count=0, endpoint=0., response=0.)
              for name in MAIN_SOURCE_NAMES}
    windows = []
    for cpu in loader(dataset, indices, config['microbatch'], 0):
        batch = move_batch(cpu, torch.device('cuda'))
        if 'goal_hand' in batch: raise ValueError('hand endpoint goal leaked into object validation')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            scene = model.scene(batch)
        def predict(goal):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                tokens, mask = model.goal(batch['start_object_points'], goal, batch['object_goal_valid'])
                return model.trajectory(scene['scene_tokens'], scene['scene_mask'], tokens, mask)
        pred = predict(batch['goal_object_points'])
        static = predict(batch['start_object_points'])
        errors = {key: torch.linalg.vector_norm(value - batch['trajectory_target'], dim=-1)
                  for key, value in (('real', pred), ('static_goal', static), ('zero', torch.zeros_like(pred)))}
        for key, value in geometric_baselines(batch).items():
            errors[key] = torch.linalg.vector_norm(value-batch['trajectory_target'], dim=-1)
        response = torch.linalg.vector_norm(pred - static, dim=-1)
        valid = batch['trajectory_valid']
        for row, sid in enumerate(batch['source_id'].tolist()):
            name = MAIN_SOURCE_NAMES[sid]; target = totals[name]; mask = valid[row]; end = mask[-1]
            target['count'] += int(mask.sum()); target['endpoint_count'] += int(end.sum())
            for key, error in errors.items(): target[key] += float(error[row][mask].sum())
            target['endpoint'] += float(errors['real'][row,-1][end].sum())
            target['response'] += float(response[row][mask].sum())
            windows.append(dict(source=name, sample_id=batch['sample_id'][row].tolist(), valid_points=int(mask.sum())))
    result = {}
    for name, x in totals.items():
        if not x['count'] or not x['endpoint_count']: raise ValueError('empty source mask')
        result[name] = dict(trajectory_epe_m=x['real']/x['count'], static_object_goal_epe_m=x['static_goal']/x['count'],
                            zero_hand_epe_m=x['zero']/x['count'], endpoint_epe_m=x['endpoint']/x['endpoint_count'],
                            real_vs_static_goal_prediction_difference_m=x['response']/x['count'],
                            object_translation_epe_m=x['object_translation']/x['count'],
                            object_rigid_follow_epe_m=x['object_rigid_follow']/x['count'],
                            improvement_vs_static_goal=1-x['real']/x['static_goal'],
                            improvement_vs_zero_hand=1-x['real']/x['zero'])
    value = dict(status='COMPLETED', split='val', scope='96 balanced validation windows; no robot/RL claim',
                 checkpoint_sha256=digest(args.checkpoint), checkpoint_step=state['step'], config=config,
                 input_identity=identity, validation_indices_sha256=digest(indices_path), windows=windows,
                 sources=result, macro_trajectory_epe_m=float(np.mean([x['trajectory_epe_m'] for x in result.values()])),
                 gpu=gpu_snapshot(torch.device('cuda')), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 audit_script_sha256=digest(Path(__file__)))
    atomic_json(args.output, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'stats', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    evaluate(parser.parse_args())
