#!/usr/bin/env python3
"""Cached-observation candidate ranking/action sensitivity; no physical counterfactual labels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import GoalPointWAM, GoalWindows, loader, move_batch, digest, atomic_json, gpu_snapshot
from ref12_methods import Ref12Method, action_from_delta, goal_points_from_world
from oakink_wm.multisource import MixedWindows, MAIN_SOURCE_NAMES
from oakink_wm.model import metrics, object_motion_masks


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['identity']['stats_sha256'] != digest(args.stats) or state['identity']['data_manifest_sha256'] != digest(args.data / 'processed/manifest.json'):
        raise ValueError('parent data/stats drift')
    if state['identity']['model_sha256'] != digest(TASK / 'src/goal_point_wam/model.py'):
        raise ValueError('parent model drift')
    model_class = Ref12Method if 'method' in state['config'] else GoalPointWAM
    model = model_class(json.loads(args.stats.read_text()), state['config']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False)
    dataset = GoalWindows(MixedWindows(args.data, 'val'), goal_type='object_points', goal_points=32)
    totals = {name: dict(windows=0, teacher_better_than_static=0, recorded_candidate_best=0,
                         action_response=0., repeated_world_noise=0., actual_teacher_target=0.,
                         static_target=0., predicted_actor_goal=0., world_gradient_norm_sum=0.,
                         moving_objects=0, moving_epe=0., moving_identity_epe=0.) for name in MAIN_SOURCE_NAMES}
    windows = []
    for cpu in loader(dataset, np.load(args.indices), args.batch, 2):
        batch = move_batch(cpu, torch.device('cuda'))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            scene = model.scene(batch)
            if isinstance(model, Ref12Method):
                actor = model.actor_from_scene(scene, batch)['trajectory']
            else:
                goal, goal_mask = model.goal(batch['start_object_points'], batch['goal_object_points'], batch['object_goal_valid'])
                actor = model.trajectory(scene['scene_tokens'], scene['scene_mask'], goal, goal_mask, model.observed_hand_condition(batch))
        def world(delta):
            action, valid = action_from_delta(delta, batch['start_hand'], batch['trajectory_valid'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens, mask = model.action(action, valid)
                return model.world(scene['scene_tokens'], scene['scene_mask'], scene['object_tokens'], scene['object_mask'], tokens, mask)
        truth = batch['trajectory_target']
        with torch.no_grad():
            predictions = {name: world(delta) for name, delta in
                           [('recorded', truth), ('static', torch.zeros_like(truth)), ('half', .5 * truth), ('one_half', 1.5 * truth), ('actor', actor)]}
            scores = {name: torch.linalg.vector_norm(goal_points_from_world(value, batch) - batch['goal_object_points'], dim=-1).mean(-1)
                      for name, value in predictions.items()}
            repeat = world(truth)
            teacher = predictions['recorded']
            rows = torch.arange(len(truth), device=truth.device)
            slots = batch['goal_object_slot']
            response = torch.linalg.vector_norm(goal_points_from_world(teacher, batch) - goal_points_from_world(predictions['static'], batch), dim=-1).mean(-1)
            noise = torch.linalg.vector_norm(goal_points_from_world(teacher, batch) - goal_points_from_world(repeat, batch), dim=-1).mean(-1)
            identity_goal = torch.linalg.vector_norm(batch['start_object_points'] - batch['goal_object_points'], dim=-1).mean(-1)
            moving, _ = object_motion_masks(batch)
            measured = metrics(teacher, batch)['point_epe'][:, :, -1]
            identity_world = dict(translation=torch.zeros_like(teacher['translation']), rotation=torch.eye(3, device=truth.device).expand_as(teacher['rotation']))
            identity_error = metrics(identity_world, batch)['point_epe'][:, :, -1]
            # This candidate set contains a recorded-label oracle; no counterfactual effects are known.
            order = ['recorded', 'static', 'half', 'one_half', 'actor']
            best = torch.stack([scores[name] for name in order]).argmin(0)
        delta = actor.detach().requires_grad_()
        induced = world(delta)
        distance = (goal_points_from_world(induced, batch) - batch['goal_object_points']).square().mean((1, 2))
        gradient, = torch.autograd.grad(distance.sum(), delta)
        gradnorm = gradient.flatten(1).norm(dim=1)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('nonfinite world-to-action gradient')
        for row, sid in enumerate(batch['source_id'].tolist()):
            name = MAIN_SOURCE_NAMES[sid]
            t = totals[name]; t['windows'] += 1
            t['teacher_better_than_static'] += int(scores['recorded'][row] < scores['static'][row])
            t['recorded_candidate_best'] += int(best[row] == 0)
            for key, value in [('action_response', response), ('repeated_world_noise', noise), ('actual_teacher_target', scores['recorded']),
                               ('static_target', identity_goal), ('predicted_actor_goal', scores['actor']), ('world_gradient_norm_sum', gradnorm)]:
                t[key] += float(value[row])
            mask = moving[row]
            t['moving_objects'] += int(mask.sum())
            t['moving_epe'] += float(measured[row][mask].sum())
            t['moving_identity_epe'] += float(identity_error[row][mask].sum())
            windows.append(dict(source=name, sample_id=batch['sample_id'][row].tolist(), candidate_goal_errors_m={key: float(value[row]) for key, value in scores.items()},
                                world_to_action_gradient_norm=float(gradnorm[row])))
    sources = {}
    for name, t in totals.items():
        n = t['windows']; m = t['moving_objects']
        sources[name] = {key: value / n for key, value in t.items() if key not in ('windows', 'moving_objects', 'moving_epe', 'moving_identity_epe')}
        sources[name].update(windows=n, moving_objects=m, moving_world_epe_m=t['moving_epe']/m if m else None,
                             moving_identity_epe_m=t['moving_identity_epe']/m if m else None)
    atomic_json(args.output, dict(status='COMPLETED', checkpoint_sha256=digest(args.checkpoint), checkpoint_step=state['step'],
                                 indices_sha256=digest(args.indices), parent_identity=state['identity'], sources=sources, windows=windows,
                                 gpu=gpu_snapshot(torch.device('cuda')), audit_sha256=digest(Path(__file__)),
                                 scope='Cached H, same action masks, real labels only for recorded action; candidate ranking and gradients do not establish counterfactual physics.'))
    print(json.dumps(sources, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'stats', 'indices', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch', type=int, default=64)
    run(p.parse_args())
