"""One bounded holding-adaptation experiment, using existing oracle labels."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tarfile
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
ARTIFACTS = ROOT / 'outputs/wam-rl-policy'


def archive_rows(directory, ticks):
    """Read only requested student states and their replay-verified labels."""
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as data:
        trajectory = {k: data[k].copy() for k in ('obs', 'action', 'action_prefix', 'success', 'object_pose')}
    if not np.array_equal(trajectory['action'], trajectory['action_prefix']):
        raise ValueError('noncanonical archived prefix')
    rows = dict(p2=[], tau=[], action=[], student_action=[], tick=[], success=[], object_pose=[], replay_error=[])
    for tick in ticks:
        query_dir = directory / f'oracle/tick-{tick:03d}'
        result = json.loads((query_dir / 'result.json').read_text())
        if result['status'] != 'COMPLETE' or not result['fork_valid'] or result['replay_error'] > 1e-5:
            raise ValueError('unverified adaptation query')
        np.testing.assert_array_equal(np.load(query_dir / 'actions.npy'), trajectory['action'][:tick])
        with np.load(query_dir / 'tau.npz', allow_pickle=False) as query:
            np.testing.assert_array_equal(query['replay_obs'], trajectory['obs'][tick])
            rows['tau'].append(query['tau'])
            rows['action'].append(query['teacher_actions'][0])
        obs = trajectory['obs'][tick]
        rows['p2'].append(np.concatenate((obs[:44], obs[66:131], obs[161:191])))
        rows['student_action'].append(trajectory['action'][tick])
        rows['tick'].append(tick)
        rows['success'].append(trajectory['success'][tick])
        rows['object_pose'].append(trajectory['object_pose'][tick])
        rows['replay_error'].append(result['replay_error'])
    return {key: np.asarray(value) for key, value in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=2700)
    args = parser.parse_args()
    output = args.output.resolve()
    if ARTIFACTS.resolve() not in output.parents or not 1 <= args.seconds <= 2700:
        raise ValueError('bounded task output and at most2700s required')
    if shutil.disk_usage(ROOT).free < 20 * 2**30:
        raise RuntimeError('less than20GiB free space')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    python = '/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    code = output / 'code'
    code.mkdir()
    blob = subprocess.check_output(['git', 'archive', commit, 'src/task/wam-rl-policy',
                                   'src/task/video-data-preparation/tools/run/replay_unidexgrasp.py'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        archive.extractall(code)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='2', MAX_JOBS='2', PYTHONDONTWRITEBYTECODE='1',
               TORCH_EXTENSIONS_DIR=str(output / 'extensions'), TMPDIR=str(output / 'tmp'),
               PYTHONPATH=str(ARTIFACTS / 'P-20261018-unidex-live-oracle-tau-deps-r5'),
               LD_LIBRARY_PATH=str(Path(python).parents[1] / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    (output / 'tmp').mkdir()
    metadata = dict(status='PREPARING', experiment_id='P-20261018-unidex-live-oracle-tau',
                    run_id=output.name, work_version='oracle-tau-holding-adaptation.v1',
                    branch='main', git_commit=commit, python=python, physical_gpu=0,
                    supervisor_pid=os.getpid(), max_seconds=args.seconds, max_bytes=2 * 2**30,
                    adaptation_seeds=[421, 453], evaluation_seeds=[471, 482],
                    physical_evaluation_seed=482, stages=[])

    def save():
        metadata['elapsed_s'] = time.monotonic() - started
        partial = output / 'process.json.part'
        partial.write_text(json.dumps(metadata, indent=2) + '\n')
        partial.replace(output / 'process.json')

    def run(name, command):
        metadata['status'] = 'RUNNING'
        with (output / (name + '.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            stage = dict(name=name, command=list(map(str, command)), pid=process.pid)
            metadata['stages'].append(stage)
            last_progress, last_signature = time.monotonic(), None
            try:
                while True:
                    stage['exit_code'] = process.poll()
                    save()
                    resource = subprocess.check_output(['nvidia-smi', '-i', '0',
                                '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'], text=True)
                    with (output / 'gpu.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(elapsed_s=metadata['elapsed_s'], stage=name, sample=resource.strip())) + '\n')
                    if process.poll() is not None:
                        stage['exit_code'] = process.returncode
                        if process.returncode:
                            raise RuntimeError('failed stage: ' + name)
                        break
                    packets = list((output / 'run-tau/oracle').glob('tick-*/result.json'))
                    signature = (log.tell(), len(packets), (output / (name + '.log')).stat().st_size)
                    if signature != last_signature:
                        last_progress, last_signature = time.monotonic(), signature
                    if time.monotonic() - last_progress > 180:
                        raise TimeoutError('no stage progress for180s')
                    if time.monotonic() - started > args.seconds or (output / 'STOP').exists():
                        raise TimeoutError('job deadline or STOP')
                    if shutil.disk_usage(ROOT).free < 20 * 2**30:
                        raise RuntimeError('less than20GiB free space')
                    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > metadata['max_bytes']:
                        raise RuntimeError('artifact budget exceeded')
                    time.sleep(10)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

    try:
        save()
        old = ARTIFACTS / 'P-20261018-unidex-closed-loop-r1'
        with np.load(old / 'dagger-p2_tau/dagger_data.npz', allow_pickle=False) as source:
            original = {k: source[k].copy() for k in source.files}
        extra = archive_rows(ARTIFACTS / 'P-20261018-unidex-heldout-eval-r1/seed-453-tau', range(90, 150))
        combined = {key: np.concatenate((original[key], extra[key]), axis=0) for key in extra}
        combined['origin_seed'] = np.array([421] * 150 + [453] * 60, dtype=np.int32)
        combined['origin_tick'] = combined['tick'].copy()
        if len(combined['p2']) != 210 or not all(np.isfinite(v).all() for v in combined.values()):
            raise ValueError('finite210-row packet required')
        packet_dir = output / 'dagger-421-453-late'
        packet_dir.mkdir()
        np.savez_compressed(packet_dir / 'dagger_data.npz', **combined)
        metadata['packet'] = dict(path=str(packet_dir), rows=210,
                                  sources=[dict(seed=421, ticks='0:150', rows=150),
                                           dict(seed=453, ticks='90:150', rows=60)])
        run('environment', [python, '-c', 'import sys,torch,numpy;print(sys.executable,sys.version,torch.__version__,numpy.__version__)'])
        trainer = code / 'src/task/wam-rl-policy/tools/run/train_closed_loop_student.py'
        data = ARTIFACTS / 'P-20261018-unidex-partial-obs-200tick-32seed-r1'
        for variant, initial, padding in [
                ('p2_tau', old / 'student-p2_tau-dagger1/model.pt', 0),
                ('p2', ARTIFACTS / 'oracle-tau-capacity-20261018-r1/student-dagger/model.pt', 403)]:
            command = [python, '-u', str(trainer), '--data', str(data), '--variant', variant,
                       '--success-only', '--pad-to', str(padding), '--gpu', '0', '--seed', '0',
                       '--steps', '2000', '--lr', '.0003', '--dagger-data', str(packet_dir),
                       '--dagger-weight', '10', '--init', str(initial), '--output', str(output / ('student-' + variant))]
            run('fit-' + variant, command)
        runner = code / 'src/task/wam-rl-policy/tools/run/closed_loop_unidex_oracle_tau.py'
        common = [python, '-u', str(runner), '--subset', str(ROOT / 'outputs/video-data-preparation/unidexgrasp-local-subset-20261014-r1'),
                  '--vendor', str(ROOT / 'third_party/UniDexGrasp2'),
                  '--assets', str(ROOT / 'outputs/video-data-preparation/unidexgrasp-policy-assets-20261014-r1/assets'),
                  '--seed', '482', '--ticks', '200', '--gpu', '0', '--gpu-seconds', '2500',
                  '--episode-length', '220',
                  '--max-replay-error', '.00001']
        previous = ARTIFACTS / 'oracle-tau-capacity-20261018-r1/student-dagger/model.pt'
        for deadline in (200, 220):
            smoke = list(common)
            smoke[smoke.index('--ticks') + 1] = '2'
            smoke[smoke.index('--episode-length') + 1] = str(deadline)
            run('deadline-' + str(deadline), smoke + ['--variant', 'p2', '--policy', str(previous),
                                                     '--output', str(output / ('deadline-' + str(deadline)))])
        with np.load(output / 'deadline-200/trajectory.npz') as left, np.load(
                output / 'deadline-220/trajectory.npz') as right, np.load(
                ARTIFACTS / 'oracle-tau-capacity-20261018-r2/seed-482/trajectory.npz') as reference:
            for key in ('obs', 'action', 'hand', 'object_pose'):
                np.testing.assert_array_equal(left[key], right[key])
                np.testing.assert_array_equal(left[key], reference[key][:2])
        metadata['deadline_smoke'] = 'bitwise identical200/220 and archived initial two states/actions'
        run('state-control', common + ['--variant', 'p2', '--policy', str(output / 'student-p2/model.pt'),
                                     '--output', str(output / 'run-state')])
        run('tau', common + ['--variant', 'p2_tau', '--policy', str(output / 'student-p2_tau/model.pt'),
                           '--output', str(output / 'run-tau')])
        metadata['status'] = 'COMPLETED'
    except Exception as error:
        metadata.update(status='FAILED', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
