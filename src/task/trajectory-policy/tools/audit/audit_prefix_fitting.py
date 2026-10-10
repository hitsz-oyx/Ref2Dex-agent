"""Oracle-label fitting bounds plus actual fixed-D decoding; not policy success."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/consequence-evaluator/src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry', 'reference', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned output required')
    state = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    utilization, memory = map(int, state.split(','))
    if utilization > 10 or memory > 512:
        raise RuntimeError('GPU not idle: '+state)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from trajectory_policy.decoder import TrajectoryDecoder
    from trajectory_policy.prefix_fitting import fit_prefix, future_xyz_velocity
    from trajectory_policy.inputs import load_geometry
    from consequence_evaluator.data import sha
    torch.set_num_threads(2)
    started = time.monotonic()
    urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    geometry, initial, hand, hashes = load_geometry(args.geometry, args.reference, urdf)
    sources = list((TASK/'src/trajectory_policy').glob('*.py'))+[Path(__file__)]
    sources += [ROOT/'src/task/consequence-evaluator/src/consequence_evaluator'/(name+'.py')
        for name in ('reset_kinematics', 'retargeter', 'reference_tracking', 'tau_tracking', 'contracts', 'reference_motion')]
    for path in sources:
        hashes[str(path.resolve())] = sha(path)
    args.output.mkdir(parents=True)
    manifest = dict(status='RUNNING', schema='ref2dex.decoder-prefix-fitting.v1', physical_gpu=args.gpu,
        gpu_before=state, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_sha256=hashes, claim='Linear XYZ reproduction bounds only, not control performance bound')
    def write(name, data):
        (args.output/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    write('manifest.json', manifest)
    try:
        ticks = np.arange(0, 542, 8)
        indices = np.minimum(ticks[:, None]+np.arange(1, 25), 542)
        labels = geometry['q'][indices]
        fit = fit_prefix(labels[:, :, :3])
        def metric(error):
            length = np.linalg.norm(error, axis=-1)
            rms = np.sqrt((length**2).mean(-1))
            return dict(per_window_3d_rms_mm=(rms*1000).tolist(), maximum_window_3d_rms_mm=float(rms.max()*1000),
                median_window_3d_rms_mm=float(np.median(rms)*1000), maximum_point_mm=float(length.max()*1000))
        pos_bound, ff_bound = metric(fit['position_bound_error']), metric(fit['ff_bound_error'])
        joint_pos = metric(fit['fitted_xyz'][:, :8]-labels[:, :8, :3])
        joint_ff = metric(fit['fitted_ff'][:, :8]-fit['target_ff'][:, :8])
        baseline_nodes = labels[:, [0, 7, 15, 23], :3]
        baseline_pos = np.einsum('tk,nkd->ntd', fit['position'], baseline_nodes)
        baseline_ff = np.einsum('tk,nkd->ntd', fit['feedforward'], baseline_nodes)
        bounded_labels = labels.copy()
        bounded_labels[:, :, :3] = fit['fitted_xyz']
        current = geometry['q'][ticks]
        obj = np.repeat(initial['object_pose'][None], len(ticks), axis=0)
        decoder = TrajectoryDecoder(urdf, 'cuda:0')
        c = decoder.encode(bounded_labels, current, obj)
        with torch.no_grad():
            decoded = decoder.decode(c, current, obj, 1/30)
        q, points = decoded['q'].cpu().numpy(), decoded['hand'].cpu().numpy()
        limits = np.asarray([1.6, 1.6, 1.6, 1.6, 1.15, .55])
        fingers = q[:, 1:, [6, 8, 10, 12, 14, 15]]
        if not np.isfinite(q).all() or not np.isfinite(points).all() or (fingers < -1e-6).any() or (fingers > limits+1e-6).any():
            raise ValueError('decoded geometry violates finite/bounds contract')
        for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05), (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
            if abs(q[:, 1:, distal]-ratio*q[:, 1:, parent]).max() > 1e-6:
                raise ValueError('decoded geometry violates native coupling')
        xyz_error = float(abs(q[:, 1:, :3]-fit['fitted_xyz']).max())
        v_error = float(abs(decoded['velocity'].cpu().numpy()[:, 1:, :3]-future_xyz_velocity(q[:, 1:, :3], 1/30)).max())
        if xyz_error > 2e-6 or v_error > 2e-6 or max(fit['normal_errors'].values()) > 1e-10:
            raise ValueError('projection/decoder operator mismatch')
        bounds_pass = pos_bound['maximum_window_3d_rms_mm'] <= 5 and ff_bound['maximum_window_3d_rms_mm'] <= 10
        joint_pass = joint_pos['maximum_window_3d_rms_mm'] <= 5 and joint_ff['maximum_window_3d_rms_mm'] <= 10
        status = 'UNPROMISING' if not bounds_pass else ('PROMISING' if joint_pass else 'UNCLEAR')
        result = dict(status=status, windows=len(ticks), bounds_pass=bounds_pass, joint_pass=joint_pass,
            position_lower_bound=pos_bound, feedforward_lower_bound=ff_bound,
            joint_position=joint_pos, joint_feedforward=joint_ff,
            copied_position=metric(baseline_pos[:, :8]-labels[:, :8, :3]),
            copied_feedforward=metric(baseline_ff[:, :8]-fit['target_ff'][:, :8]),
            joint_prefix8_hand_point_3d_rms_mm=float(np.sqrt(((points[:, :8]-geometry['fitted_points'][indices][:, :8])**2).sum(-1).mean())*1000),
            normal_errors=fit['normal_errors'], maximum_decoded_xyz_error_m=xyz_error, maximum_velocity_operator_error=v_error,
            claim=manifest['claim'], native_simulation=False)
        np.savez_compressed(args.output/'fitting.npz', ticks=ticks, c=c, decoded_q=q, decoded_hand=points,
            target_q=labels, target_hand=hand[indices],
            **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
        write('result.json', result)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for col, name in enumerate(('X', 'Y', 'Z')):
            axes[0].plot(np.arange(1, 9), labels[0, :8, col], label=name+' dense')
            axes[0].plot(np.arange(1, 9), fit['fitted_xyz'][0, :8, col], '--', label=name+' joint fit')
            axes[1].plot(np.arange(1, 9), fit['target_ff'][0, :8, col], label=name+' dense')
            axes[1].plot(np.arange(1, 9), fit['fitted_ff'][0, :8, col], '--', label=name+' joint fit')
        axes[0].set_ylabel('Wrist position (m)')
        axes[1].set_ylabel('Wrist FF base target (m)')
        axes[1].set_xlabel('Future frame of first window')
        axes[0].legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output/'first-window.png', dpi=160)
        plt.close(fig)
        if any(sha(path) != digest for path, digest in hashes.items()) or time.monotonic()-started > 120:
            raise ValueError('source drift or time bound exceeded')
        manifest.update(status='COMPLETED', elapsed_s=time.monotonic()-started,
            torch_peak_bytes=torch.cuda.max_memory_allocated())
        print(json.dumps({key: value for key, value in result.items() if key not in (
            'position_lower_bound', 'feedforward_lower_bound', 'joint_position', 'joint_feedforward', 'copied_position', 'copied_feedforward')}, indent=2), flush=True)
        for key in ('position_lower_bound', 'feedforward_lower_bound', 'joint_position', 'joint_feedforward', 'copied_position', 'copied_feedforward'):
            print(key, json.dumps({k:v for k,v in result[key].items() if k!='per_window_3d_rms_mm'}), flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic()-started)
        raise
    finally:
        write('manifest.json', manifest)


if __name__ == '__main__':
    main()
