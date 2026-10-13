#!/usr/bin/env python3
"""Export planned human wrist goals; never read measured future trajectories."""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.reference.parent / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETED' or not manifest['planned_reference_only']:
        raise ValueError('planned-reference-only source required')
    with np.load(args.reference) as data:
        if set(data.files) != {'hand_pos', 'hand_rot', 'obj'}:
            raise ValueError('unexpected input fields')
        hand, rotation, obj = data['hand_pos'], data['hand_rot'], data['obj']
    raw = torch.load(manifest['motion'], map_location='cpu', weights_only=False).numpy()
    checks = {}
    def check(name, actual, expected, tolerance=2e-5):
        checks[name] = float(np.max(np.abs(actual - expected)))
        if checks[name] > tolerance: raise ValueError('reference contract: ' + name)
    check('right_hand_positions', hand, raw[:, 150:198].reshape(-1, 16, 3))
    check('right_wrist_is_point0', hand[:, 0], raw[:, 51:54])
    check('right_hand_rotations', rotation, Rotation.from_quat(raw[:, 309:373].reshape(-1, 4)).as_matrix().reshape(-1, 16, 3, 3))
    check('object_translation', obj[:, :3, 3], raw[:, 198:201])
    check('object_rotation', obj[:, :3, :3], Rotation.from_quat(raw[:, 201:205]).as_matrix())
    if not all(np.isfinite(x).all() for x in (hand, rotation, obj)):
        raise ValueError('nonfinite reference')
    offsets = np.array([8, 16, 24], np.int64)
    frames = np.minimum(np.arange(len(hand))[:, None] + offsets, len(hand) - 1)
    positions, orientations, goal_obj = hand[frames, 0], rotation[frames, 0], obj[frames]
    local = np.einsum('nhi,nhij->nhj', positions - goal_obj[..., :3, 3], goal_obj[..., :3, :3])
    local_rotation = goal_obj[..., :3, :3].swapaxes(-1, -2) @ orientations
    reconstructed = np.einsum('nhi,nhji->nhj', local, goal_obj[..., :3, :3]) + goal_obj[..., :3, 3]
    check('relative_pose_reconstruction', reconstructed, positions)
    # Runtime uses measured current object as the anchor, NOT planned current object.
    # Verify that this transform convention is equivariant to a shared world rigid transform.
    q = Rotation.from_rotvec([.2, -.3, .1]).as_matrix(); shift = np.array([.4, .1, -.2])
    own = obj.copy(); own[:, :3, 3] += [.013, -.007, .019]
    base = np.einsum('nhi,nij->nhj', positions - own[:, None, :3, 3], own[:, :3, :3])
    moved_points = positions @ q.T + shift
    moved_center = own[:, :3, 3] @ q.T + shift
    moved_rotation = q @ own[:, :3, :3]
    changed = np.einsum('nhi,nij->nhj', moved_points - moved_center[:, None], moved_rotation)
    check('live_anchor_rigid_equivariance', changed, base)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'sparse_reference_goals.npz',
        offsets=offsets, reference_frame=frames, planned_object_pose=goal_obj,
        planned_wrist_position=positions, planned_wrist_rotation=orientations,
        wrist_in_desired_object=local, wrist_rotation_in_desired_object=local_rotation,
        wrist_goal_valid=np.ones(frames.shape, bool), active_hand=np.zeros(len(hand), np.int64))
    summary = dict(schema='ref2dex.planned-sparse-wrist-goal.v1', status='COMPLETED',
        engineering_only=True, training_allowed=False, planned_reference_only=True,
        source=str(args.reference), raw_source=manifest['motion'], frames=len(hand), fps=30,
        offsets=offsets.tolist(), active_hand='right', point_semantics='human right wrist; not Inspire hand_base_link',
        coordinate_contract='world planned poses plus desired-object-relative geometry; runtime must transform using actual current object frame',
        clocks='fixed reference time, terminal frame clamped; event-based phase alignment not implemented',
        excluded_fields='no measured future hand/object, future robot q/action, contact force or teacher commands',
        cpu_reason='small file/pose conversion and independent geometry checks; no neural computation',
        checks=checks, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        limitations='Human wrist reference is not a robot command or guaranteed feasible palm goal; embodiment offset and timing need admission before training')
    (args.output / 'result.json').write_text(json.dumps(summary, indent=2))
    t = np.arange(len(hand)) / 30
    distance = np.linalg.norm(hand[:, 0] - obj[:, :3, 3], axis=-1) * 1000
    displacement = np.linalg.norm(obj[frames[:, -1], :3, 3] - obj[:, :3, 3], axis=-1) * 1000
    wrist_motion = np.linalg.norm(hand[frames[:, -1], 0] - hand[:, 0], axis=-1) * 1000
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, end, title in [(axes[0], len(hand), 'Full planned reference'), (axes[1], 100, 'Approach / early interaction')]:
        ax.plot(t[:end], distance[:end], label='Wrist to object origin (not surface gap)')
        ax.plot(t[:end], displacement[:end], label='Object translation over next 24 frames')
        ax.plot(t[:end], wrist_motion[:end], label='Wrist displacement over next 24 frames')
        ax.set(xlabel='Reference time (s)', ylabel='mm', title=title); ax.legend(fontsize=7)
    fig.suptitle('Decision-known reference goals: object motion and hand intention are distinct')
    fig.savefig(args.output / 'reference_goal_timing.png', dpi=150); plt.close(fig)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
