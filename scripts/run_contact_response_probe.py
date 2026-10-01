#!/usr/bin/env python3
"""Sequential, budgeted fixed matrix; shared inputs are never written."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT.parent / 'Ref2Dex-agent'
REFERENCE = SOURCE_ROOT / 'src/task/CmResidual/research/physical_value/output/P-20261001-paired-evaluator-resolution-r3'
PYTHON = '/home2/wyy/miniconda3/envs/graspenv/bin/python'
PANELS = ((286, 288), (286, 289), (287, 288), (287, 289))
ARMS = (('zero_a', 0.), ('zero_b', 0.), ('plus', .01), ('minus', -.01))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def admission(index):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    row = next(line.split(',') for line in rows.splitlines() if int(line.split(',')[0]) == index)
    uuid = row[1].strip()
    if int(row[2]) > 100 or uuid in apps:
        raise RuntimeError('GPU occupied; no interference')
    return dict(index=index, uuid=uuid, time=time.time(), memory_used_mib=int(row[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError('unique output inside isolated worktree required')
    output.mkdir(parents=True)
    ref = json.loads((REFERENCE / 'run_manifest.json').read_text())
    if ref['run_status'] != 'COMPLETED':
        raise ValueError('reference still running')
    phases = {(p['training_seed'], p['evaluation_seed']): p for p in ref['phases'] if not p['repeat']}
    paths = [ROOT / 'scripts/run_contact_response_environment.py', Path(__file__),
             ROOT / 'src/task/CmResidual/contact_response.py',
             ROOT / 'src/task/CmResidual/paired_evaluation.py',
             ROOT / 'src/task/CmResidual/physical_value_live.py',
             ROOT / 'src/task/CmResidual/physical_value_contract.py',
             ROOT / 'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',
             ROOT / 'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',
             ROOT / 'third_party/DExplore/dexplore/evaluate.py',
             ROOT / 'docs/experiments/probes/P-20261001-contact-response-resolution.md']
    hashes = {str(p.resolve()): sha(p) for p in paths}
    manifest = dict(experiment_id='P-20261001-contact-response-resolution', run_id=output.name,
                    run_status='RUNNING', pid=os.getpid(), isolated_worktree=str(ROOT),
                    input_worktree_read_only=str(SOURCE_ROOT), phases=[], source_sha256=hashes,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    wall_limit_seconds=1800, storage_limit_bytes=1 << 30, no_training=True)
    begin = time.monotonic()
    def save():
        manifest['wall_seconds'] = time.monotonic() - begin
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    try:
        for key in PANELS:
            reference = phases[key]
            directory = Path(reference['directory'])
            if reference['gpu_uuid'] != admission(args.gpu)['uuid']:
                raise ValueError('must use the frozen reference GPU')
            for arm, amplitude in ARMS:
                if time.monotonic() - begin > 1620:
                    raise TimeoutError('whole probe wall budget')
                if any(sha(Path(p)) != h for p, h in hashes.items()):
                    raise ValueError('source drift')
                if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > (1 << 30):
                    raise RuntimeError('storage budget')
                gpu = admission(args.gpu)
                name = f't{key[0]}_s{key[1]}_{arm}'
                run_dir = output / name
                command = list(reference['command'])
                command[2] = str(ROOT / 'scripts/run_contact_response_environment.py')
                for flag in ('--arm', '--training-seed', '--eval-seed'):
                    i = command.index(flag)
                    del command[i:i + 2]
                for flag, value in (('--run-dir', run_dir), ('--output', run_dir / 'unused.json'),
                                    ('--output_path', run_dir / 'player'), ('--wall-seconds', 150)):
                    command[command.index(flag) + 1] = str(value)
                command += ['--initial', str(directory / 'initial_state.pt'), '--trace', str(directory / 'trace.pt'),
                            '--amplitude', str(amplitude)]
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], LOCAL_RANK='0', RANK='0', WORLD_SIZE='1',
                                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                                   PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                                   PYTHONHASHSEED=str(key[1]), TORCH_EXTENSIONS_DIR=str(output / 'cache/torch_extensions'),
                                   XDG_CACHE_HOME=str(output / 'cache'))
                environment['LD_LIBRARY_PATH'] = '/home2/wyy/miniconda3/envs/graspenv/lib:' + environment.get('LD_LIBRARY_PATH', '')
                phase = dict(name=name, panel=list(key), arm=arm, amplitude=amplitude,
                             command=command, admission=gpu, run_status='RUNNING')
                manifest['phases'].append(phase)
                save()
                start = time.monotonic()
                process = None
                print(json.dumps(dict(name=name, run_status='STARTED')), flush=True)
                try:
                    with (output / (name + '.log')).open('x') as log:
                        process = subprocess.Popen(command, cwd=ROOT / 'third_party/DExplore', env=environment,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        phase.update(pid=process.pid, pgid=process.pid)
                        save()
                        code = process.wait(timeout=180)
                    if code:
                        raise RuntimeError(f'native exit {code}; inspect {name}.log')
                    result = json.loads((run_dir / 'results.json').read_text())
                    if result['run_status'] != 'COMPLETED' or result['complete_response_windows'] != 96:
                        raise ValueError('incomplete response windows')
                    phase.update(run_status='COMPLETED', results_sha256=sha(run_dir / 'results.json'))
                except BaseException as error:
                    if process is not None and process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait(timeout=5)
                    phase.update(run_status='FAILED', error=repr(error))
                    raise
                finally:
                    phase['wall_seconds'] = time.monotonic() - start
                    save()
                print(json.dumps(dict(name=name, run_status='COMPLETED', wall_seconds=phase['wall_seconds'])), flush=True)
        manifest.update(run_status='COMPLETED', sources_unchanged=all(sha(Path(p)) == h for p, h in hashes.items()))
        if not manifest['sources_unchanged']:
            raise ValueError('source drift after probe')
    except BaseException as error:
        manifest.update(run_status='FAILED', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
