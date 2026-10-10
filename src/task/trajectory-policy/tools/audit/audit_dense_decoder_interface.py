"""Replay D288 against recorded dense plans; engineering identity, no rollout."""
import argparse
import hashlib
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or ROOT/'outputs/trajectory-policy' not in args.output.parents:
        raise ValueError('fresh task-owned output required')
    manifest_path = args.evaluation/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'COMPLETED':
        raise ValueError('completed native wave required')
    for path, digest in manifest['input_sha256'].items():
        if sha(path) != digest:
            raise ValueError('recorded source/input drift: '+path)
    roles = np.asarray(manifest['roles'])
    dense = roles == 'dense_fk'
    if dense.sum() != 4:
        raise ValueError('four dense calibration rows required')
    with np.load(args.evaluation/'plans.npz', allow_pickle=False) as stream:
        plans = {key: stream[key].copy() for key in stream.files}
    if plans['tick'].tolist() != list(range(0, 542, 8)):
        raise ValueError('all 68 query windows required')
    state = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
        '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], text=True).strip()
    util, used = map(int, state.split(','))
    if util > 10 or used > 512:
        raise ValueError('GPU not idle: '+state)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from trajectory_policy.dense_decoder import DenseDecoder, SCHEMA
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    urdf = ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    source = list((TASK/'src/trajectory_policy').glob('*.py'))+[Path(__file__), urdf,
        manifest_path, args.evaluation/'plans.npz', args.evaluation/'result.json']
    source += [ROOT/'src/task/consequence-evaluator/src/consequence_evaluator'/(name+'.py') for name in
        ('reference_tracking', 'tau_tracking', 'reset_kinematics', 'retargeter', 'contracts', 'reference_motion')]
    hashes = {str(path.resolve()): sha(path) for path in source}
    record = dict(status='RUNNING', schema=SCHEMA, kind='engineering-interface-replay',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        input_sha256=hashes, physical_gpu=args.gpu, gpu_before=state,
        no_simulation=True, no_policy_training=True)
    args.output.mkdir(parents=True)
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    write('manifest.json', record)
    started = time.monotonic()
    try:
        q = plans['q'][:, dense].reshape(-1, 25, 18)
        obj = plans['query_obj'][:, dense].reshape(-1, 4, 4)
        hand = plans['hand'][:, dense].reshape(-1, 24, 11, 3)
        velocity = plans['velocity'][:, dense].reshape(-1, 25, 18)
        decoder = DenseDecoder(urdf, 'cuda:0')
        c = decoder.encode(q[:, 1:], q[:, 0], obj)
        with torch.no_grad():
            decoded = decoder.decode(c, q[:, 0], obj, manifest['dt'])
        actual_q, actual_hand, actual_v = (decoded[key].cpu().numpy() for key in ('q', 'hand', 'velocity'))
        errors = dict(native_q=float(abs(actual_q-q).max()),
            hand=float(abs(actual_hand-hand).max()), velocity=float(abs(actual_v-velocity).max()),
            wrist_feedforward=float(abs(actual_q[:, 1:, :6]+.1*actual_v[:, 1:, :6]-(q[:, 1:, :6]+.1*velocity[:, 1:, :6])).max()))
        tolerances = dict(native_q=2e-6, hand=2e-6, velocity=1e-4, wrist_feedforward=1e-5)
        passed = all(errors[key] <= tolerance for key, tolerance in tolerances.items())
        if any(sha(path) != digest for path, digest in hashes.items()):
            raise ValueError('source/input changed during replay')
        result = dict(status='COMPLETED', engineering_pass=passed, windows=68, rows=4,
            error_max=errors, tolerance=tolerances, elapsed_seconds=time.monotonic()-started,
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            limitation='Geometry labels from an existing oracle dense trace; no H->c learning or new physical outcome.')
        np.savez_compressed(args.output/'replay.npz', c=c, q=actual_q, hand=actual_hand, velocity=actual_v)
        write('result.json', result)
        record['status'] = 'COMPLETED'
        record['replay_sha256'] = sha(args.output/'replay.npz')
        write('manifest.json', record)
        print(json.dumps(result, indent=2), flush=True)
        if not passed:
            raise RuntimeError('dense native interface identity failed; preserve evidence')
    except BaseException as error:
        if record['status'] != 'COMPLETED':
            record.update(status='FAILED', error=repr(error))
            write('manifest.json', record)
        raise


if __name__ == '__main__':
    main()
