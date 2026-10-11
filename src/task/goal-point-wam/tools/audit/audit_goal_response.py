#!/usr/bin/env python3
"""Frozen, train-only endpoint goal response; no physical counterfactual claim."""
import argparse
import json
import sys
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import GoalPointWAM, atomic_json, digest, move_batch
from oakink_wm.multisource import MAIN_SOURCE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    torch.set_num_threads(2)
    batch = move_batch(torch.load(args.batch, weights_only=False, map_location='cpu'), torch.device('cuda'))
    changed = dict(batch)
    changed['goal_hand'] = batch['goal_hand'].clone()
    valid = batch['goal_hand_valid'] & batch['start_hand_valid']
    changed['goal_hand'][..., 0] += valid.to(torch.float32) * .1
    result = dict(schema='goal-point-wam.frozen-goal-response.v1', split='train', shift_m=.1,
                  batch_sha256=digest(args.batch), evaluator_sha256=digest(Path(__file__)),
                  linear_endpoint_gain=1., diagnosis_only=True, checkpoints=[])
    for path in args.checkpoint:
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = GoalPointWAM(json.loads(args.stats.read_text()), state['config']).cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        with torch.no_grad():
            base = model(batch)
            repeated = model(batch)
            shifted = model(changed)
        world_delta = max(float((base[k] - shifted[k]).abs().max()) for k in ('translation','rotation'))
        repeat_delta = max(float((base[k] - repeated[k]).abs().max()) for k in ('translation','rotation'))
        # Sparse scene recomputation can vary even with unchanged inputs. Keep
        # that raw failed gate visible, then isolate the actual goal intervention
        # at identical scene memory, without changing production model code.
        original_scene_forward = model.scene.forward
        with torch.no_grad():
            scene = original_scene_forward(batch)
            model.scene.forward = lambda _: scene
            try:
                cached_base = model(batch)
                cached_repeat = model(batch)
                cached_shifted = model(changed)
            finally:
                model.scene.forward = original_scene_forward
        cached_delta = max(float((cached_base[k] - cached_shifted[k]).abs().max())
                           for k in ('translation','rotation'))
        cached_repeat_delta = max(float((cached_base[k] - cached_repeat[k]).abs().max())
                                  for k in ('translation','rotation'))
        base, shifted = cached_base, cached_shifted
        delta = shifted['trajectory'][:, -1, ..., 0] - base['trajectory'][:, -1, ..., 0]
        target = batch['goal_hand'] - batch['start_hand']
        endpoint_error = torch.linalg.vector_norm(base['trajectory'][:, -1] - target, dim=-1)
        measured = dict(checkpoint=str(path), checkpoint_sha256=digest(path), step=state['step'],
                        endpoint_x_gain=float(delta[valid].mean()) / .1,
                        endpoint_response_rms_m=float(delta[valid].square().mean().sqrt()),
                        endpoint_epe_m=float(endpoint_error[valid].mean()), world_isolation_max=world_delta,
                        unchanged_world_repeat_max=repeat_delta, world_isolation_pass=world_delta <= 1e-5,
                        cached_scene_world_isolation_max=cached_delta,
                        cached_scene_world_repeat_max=cached_repeat_delta,
                        cached_scene_isolation_pass=cached_delta <= 1e-5,
                        response_scope='identical cached scene, changed endpoint goal',
                        sources={})
        for source_id, name in enumerate(MAIN_SOURCE_NAMES):
            rows = batch['source_id'] == source_id
            mask = valid[rows]
            measured['sources'][name] = dict(endpoint_x_gain=float(delta[rows][mask].mean()) / .1,
                                            endpoint_epe_m=float(endpoint_error[rows][mask].mean()))
        result['checkpoints'].append(measured)
        del model, state, base, repeated, shifted, cached_base, cached_repeat, cached_shifted, scene
        torch.cuda.empty_cache()
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)
    if not all(x['cached_scene_isolation_pass'] for x in result['checkpoints']):
        raise ValueError('cached-scene goal intervention exceeds1e-5; raw diagnostics saved')


if __name__ == '__main__': main()
