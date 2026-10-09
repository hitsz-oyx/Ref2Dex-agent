"""Historical contact-window audit; no training or simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.contracts import is_within
from consequence_evaluator.retargeter import commanded_targets


def object_local(hand, pose):
    return np.einsum('tji,tpj->tpi', pose[:, :3, :3], hand-pose[:, None, :3, 3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--asset-identity', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not is_within(output, ROOT / 'outputs/consequence-evaluator'):
        parser.error('fresh task-owned output required')
    output.mkdir(parents=True)
    with args.packet.open('rb') as stream:
        p = pickle.load(stream)
    source_path = Path(p['gt_source'])
    with source_path.open('rb') as stream:
        s = pickle.load(stream)
    names = json.loads(args.asset_identity.read_text())['body_names']
    assert p['native_contact_forces'].shape[2] == len(names)
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    ids = {finger: [i for i, name in enumerate(names) if name.startswith(finger+'_')] for finger in fingers}
    source_hand, source_pose = s['hand_keypoints'][:, 0], s['object_pose'][:, 0]
    source_local = object_local(source_hand, source_pose)
    role_data = [('GT source', {key: value[:, 0] for key, value in s.items()
                             if key in ('hand_keypoints', 'object_pose', 'object_velocity', 'surface_gap',
                                        'dof_position', 'native_contact_forces')})]
    role_data += [(name, {key: p[key][:, i] for key in role_data[0][1]})
                  for i, name in enumerate(p['role_names'][:2])]
    arrays, summaries = {}, {}
    for name, role in role_data:
        h, pose = role['hand_keypoints'], role['object_pose']
        local = object_local(h, pose)
        world_error = np.sqrt(np.mean((h-source_hand)**2, axis=(1, 2)))*1000
        relative_error = np.sqrt(np.mean((local-source_local)**2, axis=(1, 2)))*1000
        object_error = np.linalg.norm(pose[:, :3, 3]-source_pose[:, :3, 3], axis=-1)*1000
        rot = Rotation.from_matrix(np.swapaxes(source_pose[:, :3, :3], 1, 2) @ pose[:, :3, :3]).magnitude()
        forces = np.stack([np.linalg.norm(role['native_contact_forces'][:, ids[finger]], axis=-1).sum(-1)
                           for finger in fingers], -1)
        data = dict(world_error_mm=world_error, object_local_error_mm=relative_error,
            object_translation_error_mm=object_error, object_rotation_error_rad=rot,
            fingertip_local_error_mm=np.linalg.norm(local[:, [2, 4, 6, 8, 10]]-source_local[:, [2, 4, 6, 8, 10]], axis=-1)*1000,
            finger_net_force_norm_sum_N=forces, surface_gap_mm=role['surface_gap']*1000,
            object_linear_velocity=role['object_velocity'][:, :3],
            object_angular_velocity=role['object_velocity'][:, 3:], q=role['dof_position'])
        arrays[name] = data
        summaries[name] = {f'{left}:{right}': dict(
            world_coordinate_rmse_mm=float(np.sqrt(np.mean(world_error[left:right]**2))),
            object_local_coordinate_rmse_mm=float(np.sqrt(np.mean(relative_error[left:right]**2))),
            object_translation_mean_mm=float(object_error[left:right].mean()),
            object_rotation_mean_rad=float(rot[left:right].mean()),
            surface_gap_mean_mm=float(data['surface_gap_mm'][left:right].mean()),
            finger_net_force_mean_N=forces[left:right].mean(0).tolist())
            for left, right in ((1, 60), (60, 120), (120, 140), (140, 160), (160, 177), (177, 201))}
    source_pd = commanded_targets(s['dof_position'][:-1, 0], s['actions'][:, 0])
    live_pd = commanded_targets(p['dof_position'][:-1], p['actions'])
    arrays['GT source']['pd_targets'] = source_pd
    for index, name in enumerate(p['role_names'][:2]):
        arrays[name]['pd_targets'] = live_pd[:, index]
    np.savez_compressed(output / 'traces.npz', **{name+'__'+key: value for name, data in arrays.items() for key, value in data.items()})
    servo = arrays['direct_gt_target_servo']
    zero_force = {finger: np.flatnonzero((servo['finger_net_force_norm_sum_N'][:, i] < .1)
                  & (arrays['GT source']['finger_net_force_norm_sum_N'][:, i] > 1.)
                  & (np.arange(543) >= 140) & (np.arange(543) <= 200)).tolist()
                  for i, finger in enumerate(fingers)}
    report = dict(status='COMPLETED', engineering_only=True, summaries=summaries,
        fingers=fingers, native_body_names=names, source=str(source_path),
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        packet_sha256=hashlib.sha256(args.packet.read_bytes()).hexdigest(),
        source_force_gt1N_live_lt_point1N_ticks=zero_force,
        snapshot_ticks={str(t): {name: {key: np.asarray(value[t]).tolist() for key, value in data.items()}
                                 for name, data in arrays.items()} for t in (140, 150, 160, 170, 176, 177, 180, 190, 200)},
        limitations='native net forces are not hand-object pair labels; force norms are not grasp quality')
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ticks = np.arange(140, 201)
    for name, data in arrays.items():
        for ax, key in zip(axes[:3], ('world_error_mm', 'object_local_error_mm', 'object_translation_error_mm')):
            ax.plot(ticks, data[key][ticks], label=name)
        axes[3].plot(ticks, data['finger_net_force_norm_sum_N'][ticks].sum(-1), label=name)
    for ax, label in zip(axes, ('World hand error (mm)', 'Object-local hand error (mm)',
                               'Object position error (mm)', 'Finger net force sum (N)')):
        ax.set_ylabel(label); ax.axvline(176, color='gray', linestyle=':'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Native control tick'); fig.tight_layout()
    fig.savefig(output / 'world_vs_relative_contact.png', dpi=150)
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    for i, finger in enumerate(fingers):
        for name, data in arrays.items():
            axes[i].plot(ticks, data['finger_net_force_norm_sum_N'][ticks, i], label=name)
        axes[i].set_ylabel(finger+' (N)'); axes[i].grid(alpha=.2); axes[i].axvline(176, color='gray', linestyle=':')
    axes[0].legend(fontsize=8); axes[-1].set_xlabel('Native control tick'); fig.tight_layout()
    fig.savefig(output / 'finger_net_forces.png', dpi=150)
    print(json.dumps(dict(status='COMPLETED', summaries=summaries,
                         force_loss_candidates=zero_force)), flush=True)


if __name__ == '__main__':
    main()
