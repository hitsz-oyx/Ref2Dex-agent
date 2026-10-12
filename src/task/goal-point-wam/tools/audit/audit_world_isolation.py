#!/usr/bin/env python3
"""Diagnose full-forward noise versus cached-world goal independence; no training."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    task = args.code_root / 'src/task/goal-point-wam'
    sys.path.insert(0, str(task / 'tools/run'))
    from train_three_domain import GoalPointWAM, GoalWindows, atomic_json, digest, loader, move_batch
    from oakink_wm.multisource import MixedWindows
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config, identity = state['config'], state['identity']
    if state['checkpoint_kind'] != 'goal-point-wam.single.v1': raise ValueError('real checkpoint required')
    if digest(task / 'src/goal_point_wam/model.py') != identity['model_sha256']:
        raise ValueError('model source identity drift')
    if digest(args.stats) != identity['stats_sha256']: raise ValueError('stats drift')
    if digest(args.data/'processed/manifest.json') != identity['data_manifest_sha256']:
        raise ValueError('data drift')
    path = args.checkpoint.parent/'train_indices.npy'
    if digest(path) != identity['train_indices_sha256']: raise ValueError('train stream drift')
    indices = np.load(path)[-config['microbatch']:]
    dataset = GoalWindows(MixedWindows(args.data, 'train'), goal_type='object_points', goal_points=config['goal_points'])
    batch = move_batch(next(iter(loader(dataset, indices, config['microbatch'], 0))), torch.device('cuda'))
    model = GoalPointWAM(json.loads(args.stats.read_text()), config).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    changed = dict(batch, goal_object_points=batch['goal_object_points'].flip(1)+.07)
    with torch.no_grad():
        original, repeat, altered = model(batch), model(batch), model(changed)
    valid = batch['object_valid']
    def difference(a, b):
        return {key: dict(all_max=float((a[key]-b[key]).abs().max()),
                          valid_max=float((a[key]-b[key])[valid].abs().max()))
                for key in ('translation', 'rotation')}
    with torch.no_grad():
        scene = model.scene(batch)
        actions, mask = model.action(batch['action'].reshape(-1,24,2,11,9),
                                     batch['action_valid'].reshape(-1,24,2,11))
        def cached_world():
            return model.world(scene['scene_tokens'],scene['scene_mask'],scene['object_tokens'],
                               scene['object_mask'],actions,mask)
        first = cached_world()
        if model.goal_type != 'object_points': raise ValueError('object contract required')
        tokens, goal_mask = model.goal(changed['start_object_points'],changed['goal_object_points'],changed['object_goal_valid'])
        hand = model.observed_hand_condition(changed) if hasattr(model,'observed_hand_condition') else None
        if hand is None:
            model.trajectory(scene['scene_tokens'],scene['scene_mask'],tokens,goal_mask)
        else:
            model.trajectory(scene['scene_tokens'],scene['scene_mask'],tokens,goal_mask,hand)
        second = cached_world()
    gradient_batch = dict(batch, goal_object_points=batch['goal_object_points'].detach().requires_grad_())
    predicted = model(gradient_batch)
    value = predicted['translation'][valid].sum() + predicted['rotation'][valid].sum()
    gradient, = torch.autograd.grad(value, gradient_batch['goal_object_points'], allow_unused=True)
    cached = difference(first, second)
    passed = gradient is None and all(m['all_max'] <= 1e-5 for m in cached.values())
    atomic_json(args.output, dict(engineering_only=True, checkpoint_sha256=digest(args.checkpoint),
                                  checkpoint_step=state['step'],input_identity=identity,
                                  actual_model_sha256=digest(task/'src/goal_point_wam/model.py'),
                                  audit_script_sha256=digest(Path(__file__)), valid_objects=int(valid.sum()),
                                  padded_objects=int((~valid).sum()), same_goal_full_forward=difference(original,repeat),
                                  changed_goal_full_forward=difference(original,altered), cached_world=cached,
                                  world_to_goal_gradient_is_none=gradient is None, cached_isolation_pass=passed,
                                  note='Original full-forward raw threshold remains recorded; this isolates fixed W inputs.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root','checkpoint','data','stats','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(parser.parse_args())
