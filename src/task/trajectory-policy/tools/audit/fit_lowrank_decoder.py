"""Fit fixed PCA trajectory D on hand-derived geometry; in-sample coverage only."""
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
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry', 'reference', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--metric', action='store_true', help='physical pose/FF metric instead of bounded-coordinate PCA')
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task output required')
    state = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    util, used = map(int, state.split(','))
    if util > 10 or used > 512:
        raise ValueError('GPU not idle: '+state)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    if args.metric:
        from trajectory_policy.metric_decoder import coordinates, MetricDecoder as Decoder, SCHEMA, fit_basis
    else:
        from trajectory_policy.lowrank_decoder import coordinates, LowrankDecoder as Decoder, SCHEMA
    from trajectory_policy.inputs import load_geometry
    from consequence_evaluator.data import sha
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    g, initial, hand, hashes = load_geometry(args.geometry, args.reference, urdf)
    sources = list((TASK/'src/trajectory_policy').glob('*.py'))+[Path(__file__)]
    sources += [ROOT/'src/task/consequence-evaluator/src/consequence_evaluator'/(name+'.py') for name in
        ('reference_tracking', 'tau_tracking', 'reset_kinematics', 'retargeter', 'contracts', 'reference_motion')]
    for path in sources:
        hashes[str(path.resolve())] = sha(path)
    args.output.mkdir(parents=True)
    m = dict(status='RUNNING', schema=SCHEMA, git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), input_sha256=hashes, physical_gpu=args.gpu, gpu_before=state)
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write('manifest.json', m)
    try:
        ticks = np.arange(543)
        indices = np.minimum(ticks[:, None]+np.arange(1, 25), 542)
        obj = np.repeat(initial['object_pose'][None], 543, axis=0)
        values = coordinates(g['q'][indices], g['q'], obj)
        if args.metric:
            np.savez(args.output/'basis.npz', **fit_basis(values, 'cuda:0'))
        else:
            tensor = torch.as_tensor(values, device='cuda:0', dtype=torch.float64)
            mean = tensor.mean(0)
            _, singular, vh = torch.linalg.svd(tensor-mean, full_matrices=False)
            components = vh[:48].cpu().numpy().astype(np.float32)
            sign = np.sign(components[np.arange(48), np.abs(components).argmax(1)])
            components *= sign[:, None]
            scale = np.maximum(singular[:48].cpu().numpy()/np.sqrt(542), 1e-4).astype(np.float32)
            np.savez(args.output/'basis.npz', schema=np.asarray(SCHEMA), mean=mean.cpu().numpy().astype(np.float32), components=components, latent_scale=scale)
        decoder = Decoder(urdf, 'cuda:0', args.output/'basis.npz')
        query = np.arange(0, 542, 8)
        c = decoder.encode(g['q'][indices[query]], g['q'][query], obj[query])
        with torch.no_grad():
            decoded = decoder.decode(c, g['q'][query], obj[query], 1/30)
        q, point, vel = (decoded[key].cpu().numpy() for key in ('q', 'hand', 'velocity'))
        target = g['q'][indices[query]]
        delta = np.diff(target[:, :, :3], axis=1)
        target_v = np.concatenate((delta[:, :1], (delta[:, :-1]+delta[:, 1:])/2, delta[:, -1:]), 1)*30
        ff_error = q[:, 1:9, :3]+.1*vel[:, 1:9, :3]-(target[:, :8, :3]+.1*target_v[:, :8])
        error = np.linalg.norm(point-g['fitted_points'][indices[query]], axis=-1)
        prefix_rms = float(np.sqrt((error[:, :8]**2).mean())*1000)
        palm_max = float(error[:, :8, 0].max()*1000)
        ff_rms = float(np.sqrt((ff_error**2).sum(-1).mean(-1)).max()*1000)
        finite = all(np.isfinite(value).all() for value in (c, q, point, vel))
        fingers = q[:, 1:, [6, 8, 10, 12, 14, 15]]
        bounds = bool((fingers >= -1e-6).all() and (fingers <= np.asarray([1.6]*4+[1.15, .55])+1e-6).all())
        for distal, parent, ratio in ((7, 6, 1.05), (9, 8, 1.05), (11, 10, 1.05), (13, 12, 1.05), (16, 15, .6), (17, 15, .8)):
            bounds = bounds and bool(abs(q[:, 1:, distal]-ratio*q[:, 1:, parent]).max() <= 1e-6)
        if not finite or not bounds:
            raise ValueError('finite/native finger bounds or coupling violation')
        passed = finite and prefix_rms <= 5 and palm_max <= 10 and ff_rms <= 10
        result = dict(status='PROMISING' if passed else 'UNPROMISING', offline_screen_pass=passed,
            windows=68, train_windows=543, evaluation='In-sample geometric representation coverage; not held-out policy',
            prefix8_hand_3d_rms_mm=prefix_rms, prefix8_palm_max_mm=palm_max, prefix8_wrist_ff_worst_window_3d_rms_mm=ff_rms,
            full24_hand_3d_rms_mm=float(np.sqrt((error**2).mean())*1000), native_evaluation_required=passed)
        result.update(representation='metric48' if args.metric else 'lowrank48', bounds_and_coupling_pass=bounds)
        np.savez_compressed(args.output/'coverage.npz', ticks=query, c=c, q=q, hand=point, velocity=vel, target_q=target, target_hand=hand[indices[query]])
        if any(sha(path) != digest for path, digest in hashes.items()) or time.monotonic()-started > 120:
            raise ValueError('identity drift or120s budget exceeded')
        m.update(status='COMPLETED', basis_sha256=sha(args.output/'basis.npz'), elapsed_s=time.monotonic()-started,
            torch_peak_bytes=torch.cuda.max_memory_allocated())
        write('result.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        m.update(status='FAILED', error=repr(error), elapsed_s=time.monotonic()-started)
        raise
    finally:
        write('manifest.json', m)


if __name__ == '__main__':
    main()
