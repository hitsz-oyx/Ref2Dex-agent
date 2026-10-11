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
            shifted = model(changed)
        world_delta = max(float((base[k] - shifted[k]).abs().max()) for k in ('translation','rotation'))
        if world_delta > 1e-5: raise ValueError('goal leaked into world prediction')
        delta = shifted['trajectory'][:, -1, ..., 0] - base['trajectory'][:, -1, ..., 0]
        target = batch['goal_hand'] - batch['start_hand']
        endpoint_error = torch.linalg.vector_norm(base['trajectory'][:, -1] - target, dim=-1)
        measured = dict(checkpoint=str(path), checkpoint_sha256=digest(path), step=state['step'],
                        endpoint_x_gain=float(delta[valid].mean()) / .1,
                        endpoint_response_rms_m=float(delta[valid].square().mean().sqrt()),
                        endpoint_epe_m=float(endpoint_error[valid].mean()), world_isolation_max=world_delta,
                        sources={})
        for source_id, name in enumerate(MAIN_SOURCE_NAMES):
            rows = batch['source_id'] == source_id
            mask = valid[rows]
            measured['sources'][name] = dict(endpoint_x_gain=float(delta[rows][mask].mean()) / .1,
                                            endpoint_epe_m=float(endpoint_error[rows][mask].mean()))
        result['checkpoints'].append(measured)
        del model, state, base, shifted
        torch.cuda.empty_cache()
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
