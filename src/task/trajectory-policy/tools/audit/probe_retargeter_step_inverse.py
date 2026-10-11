"""Train-only local wrist inverse calibration; not a manipulation controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from audit_learned_retargeter import ROOT, TASK, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned output required')
    before = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    utilization, memory = map(int, before.split(','))
    if utilization > 10 or memory > 512:
        raise ValueError('calibration GPU not idle: '+before)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='2')
    import torch
    sys.path.insert(0, str(TASK.parents[0]/'consequence-evaluator/src'))
    from consequence_evaluator.object_relative_servo import recover_wrist
    from consequence_evaluator.retargeter import wrist_rotation
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest = dict(status='RUNNING', git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        physical_gpu=args.gpu, gpu_before=before,
        claim='Local six-wrist-DOF supervised inverse diagnostic only; actual future q is a privileged comparator, never policy input')
    try:
        fit = json.loads((args.fit/'manifest.json').read_text())
        if fit['status'] != 'COMPLETED' or len(fit['metadata']) != 2:
            raise ValueError('fixed completed two-source fit required')
        hashes = dict(fit['input_sha256'])
        for path in (args.fit/'manifest.json', Path(__file__), TASK.parents[0]/'consequence-evaluator/src/consequence_evaluator/object_relative_servo.py', TASK.parents[0]/'consequence-evaluator/src/consequence_evaluator/retargeter.py'):
            hashes[str(path.resolve())] = sha(path)
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('fixed source drift')
        pieces = {split: [] for split in ('train', 'validation')}
        geometry_errors = []
        for metadata in fit['metadata']:
            with np.load(Path(metadata['source'])/'low.npz', allow_pickle=False) as f:
                data = {key: f[key] for key in ('q', 'dq', 'hand', 'applied', 'episode_start', 'episode_tick')}
            template = (data['hand'][0, 0]-data['q'][0, 0, :3]) @ wrist_rotation(data['q'][0, 0])
            for split, starts in (('train', [0, 542, 1084, 1626]), ('validation', [2168])):
                indices = np.concatenate([np.arange(start, start+518) for start in starts])
                if not np.all(data['episode_start'][indices+24] == data['episode_start'][indices]):
                    raise ValueError('whole reset-wave boundary differs')
                current = data['q'][indices].reshape(-1, 18)
                following = data['q'][indices+1].reshape(-1, 18)
                velocity = data['dq'][indices].reshape(-1, 18)
                hand_next = data['hand'][indices+1].reshape(-1, 11, 3)
                recovered = recover_wrist(hand_next, template, current)
                joint_delta = following[:, :6]-current[:, :6]
                raw_jumps = int((np.abs(joint_delta[:, 3:6]) > np.pi).sum())
                joint_delta[:, 3:6] = (joint_delta[:, 3:6]+np.pi) % (2*np.pi)-np.pi
                angular_difference = wrist_rotation(recovered) @ np.swapaxes(wrist_rotation(following), -1, -2)
                so3_error = Rotation.from_matrix(angular_difference).magnitude()
                geometry_errors.append(dict(source=metadata['source'], split=split,
                    xyz_max_m=float(np.abs(recovered[:, :3]-following[:, :3]).max()),
                    xyz_p95_m=float(np.percentile(np.abs(recovered[:, :3]-following[:, :3]).max(-1), 95)),
                    so3_max_rad=float(so3_error.max()), so3_p95_rad=float(np.percentile(so3_error, 95)),
                    raw_joint_angle_boundary_jumps=raw_jumps))
                scale = np.asarray(fit['native_controller']['scale'][:6], np.float64)
                target = data['applied'][indices].reshape(-1, 18)[:, :6]*scale
                ticks = np.repeat(data['episode_tick'][indices], 16)
                # Columns per DOF: motion, current velocity, intercept. No future action input.
                measured_features = np.stack((joint_delta, velocity[:, :6], np.ones_like(target)), -1)
                hand_features = np.stack((recovered[:, :6]-current[:, :6], velocity[:, :6], np.ones_like(target)), -1)
                pieces[split].append((measured_features, hand_features, target, ticks))
        arrays = {split: [np.concatenate([part[k] for part in pieces[split]]) for k in range(4)] for split in pieces}
        if len(arrays['train'][0]) != 66304 or len(arrays['validation'][0]) != 16576:
            raise ValueError('fixed sample counts differ')
        reports = {}
        for branch, column in (('privileged_next_q', 0), ('next_hand_wrist', 1)):
            x, y = arrays['train'][column], arrays['train'][2]
            coefficients = []
            for dof in range(6):
                tx = torch.as_tensor(x[:, dof], dtype=torch.float64, device='cuda')
                ty = torch.as_tensor(y[:, dof], dtype=torch.float64, device='cuda')
                coefficients.append(torch.linalg.lstsq(tx, ty).solution.cpu().numpy())
            coefficients = np.stack(coefficients)
            branch_reports = {}
            for split in arrays:
                features, target, ticks = arrays[split][column], arrays[split][2], arrays[split][3]
                predicted = np.sum(features*coefficients[None], axis=-1)
                # Native wrist offset scale is the physical PD displacement/rotation unit.
                scale = np.asarray(fit['native_controller']['scale'][:6], np.float64)
                predicted = (predicted/scale).clip(-1, 1)*scale
                error = np.abs(predicted-target)
                values = {}
                for label, mask in (('all', np.ones(len(error), bool)), ('startup', ticks < 64)):
                    values[label] = dict(xyz_pd_p95_m=float(np.percentile(error[mask, :3].max(-1), 95)),
                        rotation_pd_p95_rad=float(np.percentile(error[mask, 3:6].max(-1), 95)),
                        pd_mae_by_dof=error[mask].mean(0).tolist())
                branch_reports[split] = values
            val = branch_reports['validation']
            passed = all(val[key]['xyz_pd_p95_m'] <= .005 and val[key]['rotation_pd_p95_rad'] <= .05 for key in ('all', 'startup'))
            reports[branch] = dict(coefficients=coefficients.tolist(), metrics=branch_reports, wrist_screen_passed=passed)
        processes = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
        for process in processes.splitlines():
            pid, used = map(int, process.split(','))
            if pid != os.getpid() and used > 512:
                raise RuntimeError('foreign GPU compute detected')
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('source drift during calibration')
        if not all(np.isfinite(arr).all() for arrs in arrays.values() for arr in arrs):
            raise ValueError('nonfinite calibration data')
        result = dict(status='PROMISING' if reports['next_hand_wrist']['wrist_screen_passed'] else 'UNCLEAR',
            branches=reports, geometry_errors=geometry_errors, claim=manifest['claim'])
        write(args.output/'result.json', result)
        manifest.update(status='COMPLETED', input_sha256=hashes, elapsed_s=time.monotonic()-started,
            torch_peak_bytes=torch.cuda.max_memory_allocated(), train_samples=66304, validation_samples=16576)
        write(args.output/'manifest.json', manifest)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status='FAILED', elapsed_s=time.monotonic()-started, error=repr(error))
        write(args.output/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    def deadline(signum, frame):
        raise TimeoutError('local inverse calibration exceeded120s')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    main()
