"""Bounded recovery of ref13 whole-world cold replay, before changing Y."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
TASK = Path(__file__).resolve().parents[2]
PYTHON = Path('/home2/wyy/miniconda3/envs/graspenv/bin/python')
ACTOR = ROOT/'outputs/consequence-evaluator/baseline-transfer-s3-20261007-r2/airplane_base'
EXPECTED_SHA = '8882fabd2d83c56312ca90e3b27ea145f628dc1ddcedafbcc751303a27d52871'
MOTION_NAMES = ('s3_airplane_lift', 's7_airplane_lift_Retake', 's9_airplane_lift')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def prepare(root):
    sys.path.insert(0, str(ROOT/'src/task/consequence-evaluator/src'))
    from consequence_evaluator.provenance import self_trained_ancestry
    self_trained_ancestry(ACTOR, ROOT/'outputs/consequence-evaluator')
    actor = json.loads((ACTOR/'run_manifest.json').read_text())
    checkpoint = Path(actor['checkpoint'])
    if sha(checkpoint) != EXPECTED_SHA:
        raise ValueError('frozen reconstructed actor changed')
    motions = root/'motions'
    motions.mkdir(exist_ok=True)
    sources = [ACTOR/'motions'/MOTION_NAMES[0]]
    mixed = ROOT/'outputs/consequence-evaluator/baseline-transfer-mixed12-20261008-r3/mixed12/motions'
    sources += [mixed/name for name in MOTION_NAMES[1:]]
    for name, source in zip(MOTION_NAMES, sources):
        if not (source/'interaction_hand_inspire.pt').is_file():
            raise FileNotFoundError(source)
        target = motions/name
        if not target.exists():
            target.symlink_to(source.resolve(), target_is_directory=True)
    config = ROOT/'src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7'
    paths = [checkpoint, ACTOR/'run_manifest.json', ACTOR/'input_manifest.json',
             Path(__file__), Path(__file__).with_name('ref13_progress_worker.py'),
             Path(__file__).with_name('collect_oracle_y_candidates.py'),
             config/'environment.yaml', config/'training.yaml']
    paths += [motions/name/'interaction_hand_inspire.pt' for name in MOTION_NAMES]
    value = dict(checkpoint=str(checkpoint), checkpoint_sha256=EXPECTED_SHA,
                 original_checkpoint_available=False, original_panel_available=False,
                 runtime=str(PYTHON), torch='2.4.1+cu121', num_envs=96,
                 hybrid_init_prob=.5, backend='gpu_physx_cpu_pipeline',
                 physics_num_threads=1, action_semantics='per_env_reactive_native_residual8',
                 input_sha256={str(path.resolve()): sha(path) for path in paths},
                 motion_root=str(motions), config_root=str(config),
                 scope='reconstructed frozen actor; old experimental method, not original weight/motion replication')
    previous = root/'inputs.json'
    if previous.exists() and json.loads(previous.read_text()) != value:
        raise ValueError('recovery input identity drift')
    write(previous, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=263)
    parser.add_argument('--stage', choices=('baseline', 'repeat', 'candidate'), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--schedule', type=Path)
    parser.add_argument('--candidate', type=int, default=0, choices=range(7))
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--name')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    try:
        root.relative_to(ROOT/'outputs/cm-interaction-oracle')
    except ValueError:
        raise ValueError('task-owned output required')
    if not 100 <= args.seed <= 299 or args.offset not in range(0,89,8):
        raise ValueError('task-owned output and frozen Probe seed/offset required')
    root.mkdir(parents=True, exist_ok=True)
    inputs = prepare(root)
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied before worker launch: '+occupied)
    folder = root/(args.name or args.stage)
    if folder.exists():
        raise FileExistsError(folder)
    config = Path(inputs['config_root'])
    cmd = [str(PYTHON), str(Path(__file__).with_name('ref13_progress_worker.py')),
           '--run-dir', str(folder), '--candidate', str(args.candidate),
           '--capture-progress-geometry', '--expected-checkpoint-sha256', EXPECTED_SHA,
           '--wall-seconds', '180', '--num_threads', '1', '--task', 'Dexplore_Inspire',
           '--cfg_env', str(config/'environment.yaml'), '--cfg_train', str(config/'training.yaml'),
           '--checkpoint', inputs['checkpoint'], '--motion_file', inputs['motion_root'],
           '--headless', '--num_envs', '96', '--seed', str(args.seed),
           '--sim_device', 'cuda:0', '--rl_device', 'cuda:0', '--graphics_device_id', '0',
           '--pipeline', 'cpu', '--output', str(folder/'native_eval.json'),
           '--output_path', str(folder/'native')]
    if args.stage == 'repeat':
        cmd += ['--reference', str(root/'baseline')]
    if args.stage == 'candidate':
        if args.reference is None or args.schedule is None:
            raise ValueError('candidate requires frozen reference/synchronous schedule')
        cmd += ['--reference', str(args.reference.resolve()), '--anchor-schedule', str(args.schedule.resolve()),
                '--group-id', '0', '--rolling-offset', str(args.offset), '--post-window', '32',
                '--record-rolling-trace']
        if args.plan is not None:
            cmd += ['--rolling-plan', str(args.plan.resolve())]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(ROOT/'tmp'),
               TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/torch_extensions'), MAX_JOBS='2',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               LD_LIBRARY_PATH=str(PYTHON.parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH',''))
    manifest = dict(status='RUNNING', stage=args.stage, command=cmd,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    physical_gpu=args.gpu, input_sha256=inputs['input_sha256'], seed=args.seed,
                    original_checkpoint_match=False, wall_seconds_cap=240, no_training=True)
    started = time.monotonic()
    status_file = root/(folder.name+'-status.json')
    with (root/(folder.name+'.log')).open('x') as log:
        process = subprocess.Popen(cmd, cwd=ROOT/'third_party/DExplore', env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        manifest['pid'] = process.pid
        write(status_file,manifest)
        print(json.dumps(dict(stage=args.stage,pid=process.pid,log=str(log.name))),flush=True)
        try:
            code = process.wait(timeout=240)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            manifest.update(status='FAILED', error='owned worker interrupted or timed out')
            raise
        finally:
            manifest['elapsed_s'] = time.monotonic()-started
            write(status_file,manifest)
    manifest.update(status='COMPLETED' if code == 0 else 'FAILED', returncode=code)
    if any(sha(path) != digest for path,digest in inputs['input_sha256'].items()):
        manifest.update(status='FAILED', error='input drift')
    write(status_file,manifest)
    print(json.dumps(manifest),flush=True)
    if manifest['status'] != 'COMPLETED':
        raise RuntimeError('native worker failed; see '+str(root/(folder.name+'.log')))


if __name__ == '__main__':
    main()
