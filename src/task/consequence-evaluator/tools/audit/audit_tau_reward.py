"""Rescore recorded physical behavior through the actual tau reward (no model)."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/task/consequence-evaluator/src'))
from consequence_evaluator.reference_tracking import (
    ACTIVE, RESIDUAL_LIMITS, apply_coupling, native_action, reference_velocity, wrist_feedforward)
from consequence_evaluator.tau_tracking import tau_reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT / 'outputs/consequence-evaluator')
    if output.exists():
        raise ValueError('fresh task-owned audit directory required')
    manifest = json.loads((args.evaluation / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED' or not manifest['tau_only']:
        raise ValueError('completed tau-only evaluation required')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(args.reference) != manifest['input_sha256'][str(args.reference.resolve())]:
        raise ValueError('reference provenance mismatch')
    with args.reference.open('rb') as stream:
        ref = pickle.load(stream)
    with np.load(args.evaluation / 'trajectory.npz', allow_pickle=False) as stream:
        a = {k: stream[k] for k in ('hand_keypoints', 'object_pose', 'pair', 'latent',
                                  'surface_gap', 'support_gap', 'table_footprint', 'dof_position')}
    geometry_path = Path(manifest['geometry_reference']) / 'geometry.npz'
    if digest(geometry_path) != manifest['input_sha256'][str(geometry_path.resolve())]:
        raise ValueError('geometry provenance mismatch')
    if not np.array_equal(a['object_pose'][0], np.broadcast_to(ref['object_pose'][0, 0], a['object_pose'][0].shape)):
        raise ValueError('initial object state mismatch')
    with np.load(geometry_path) as stream:
        q = torch.from_numpy(stream['q'].copy())
    controller = manifest['native_controller']; ff = manifest['wrist_feedforward_contract']
    base = wrist_feedforward(q[1:], reference_velocity(q, ff['control_dt'])[1:],
                             torch.tensor(ff['damping']) / torch.tensor(ff['stiffness']))
    limits = torch.tensor(RESIDUAL_LIMITS)
    scale = torch.tensor(controller['scale']); offset = torch.tensor(controller['offset'])
    initial_height = float(ref['object_pose'][0, 0, 2, 3])
    roles = np.asarray(manifest['roles']); summary = {}
    with torch.no_grad():
        for role in ('tau_nominal', 'tracker'):
            ids = np.flatnonzero(roles == role)
            latent = torch.from_numpy(a['latent'][:, ids])
            hand = torch.from_numpy(a['hand_keypoints'][1:, ids])
            obj = torch.from_numpy(a['object_pose'][1:, ids])
            pair = torch.from_numpy(a['pair'][1:, ids])
            future = torch.from_numpy(ref['hand_keypoints'][1:, 0])[:, None].expand_as(hand)
            reward = tau_reward(hand.flatten(0, 1), obj.flatten(0, 1), future.flatten(0, 1),
                                pair.flatten(), initial_height,
                                latent.flatten(0, 1)).reshape(542, len(ids))
            target = base[:, None].expand(542, len(ids), 18).clone()
            target[..., list(ACTIVE)] += torch.tanh(latent) * limits
            intended = native_action(apply_coupling(target),
                                     torch.from_numpy(a['dof_position'][:-1, ids]), offset, scale)
            excess = (intended - intended.clamp(-1, 1)).abs().amax(-1)
            reward -= .20 * excess
            lift = a['object_pose'][1:, ids, 2, 3] - initial_height
            held = ((a['surface_gap'][1:, ids] <= .01) & (lift >= .03)
                    & ~(a['table_footprint'][1:, ids] & (abs(a['support_gap'][1:, ids]) <= .02)))
            forceheld = a['pair'][1:, ids] & (lift > .03)
            summary[role] = dict(mean_reward=float(reward.mean()),
                                discounted_return=float((reward * torch.pow(.99, torch.arange(542))[:, None]).sum(0).mean()),
                                held_fraction=float(held.mean()), forceheld_fraction=float(forceheld.mean()),
                                forceheld_recall=float((held & forceheld).sum() / max(held.sum(), 1)),
                                forceheld_false_fraction=float((~held & forceheld).sum() / max(forceheld.sum(), 1)))
    passed = (summary['tracker']['held_fraction'] > .5
              and summary['tracker']['mean_reward'] > summary['tau_nominal']['mean_reward']
              and summary['tracker']['discounted_return'] > summary['tau_nominal']['discounted_return'])
    result = dict(status='PASS' if passed else 'FAIL', roles=summary,
                  note='CPU statistical replay only, no neural inference; check reward ordering, not policy success',
                  reward_sha256=digest(ROOT / 'src/task/consequence-evaluator/src/consequence_evaluator/tau_tracking.py'),
                  trajectory_sha256=digest(args.evaluation / 'trajectory.npz'),
                  evaluation=str(args.evaluation.resolve()))
    output.mkdir(parents=True)
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
