#!/usr/bin/env python3
"""Run the fixed HF19 source panel; never train or choose a policy."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha, gpu_admission, PYTHON


def run(args):
    begin = time.monotonic()
    base = (ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output = args.output.resolve()
    if output.parent != base or args.output.exists() or args.output.is_symlink():
        raise ValueError('unique owned scientific output required')
    engineering = ROOT/'docs/experiments/probes/P-20261002-contact-geometry-native-engineering-completion-r2.json'
    e = json.loads(engineering.read_text())
    if e['run_status'] != 'COMPLETED' or not e['engineering_passed']:
        raise ValueError('native engineering required')
    if any(sha(Path(k)) != v for k,v in e['artifact_sha256'].items()):
        raise ValueError('engineering artifact drift')
    old_manifest = Path(next(p for p in e['artifact_sha256'] if 'engineering-r2/run_manifest' in p))
    old = json.loads(old_manifest.read_text())
    if any(sha(Path(k)) != v for k,v in old['input_sha256'].items()):
        raise ValueError('native source input drift')
    hashes = dict(old['input_sha256'])
    for p in [Path(__file__),engineering,ROOT/'docs/experiments/probes/P-20261002-contact-geometry-action-information.md']:
        hashes[str(p.resolve())] = sha(p)
    admission = gpu_admission(args.gpu)
    output.mkdir(exist_ok=False)
    prior_seconds = e['total_seconds_conservative']
    manifest = dict(experiment_id='P-20261002-contact-geometry-action-information',family='HF19',
                    probe_index_in_family=1,run_status='STARTED',smoke_only=False,pid=os.getpid(),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    input_sha256=hashes,gpu=admission,phases=[],
                    wall_limit_seconds=3600,source_limit_seconds=2500,output_limit_bytes=8<<30,
                    prior_engineering_seconds=prior_seconds,fit_started=False)
    def save():
        manifest['elapsed_seconds'] = time.monotonic()-begin
        manifest['cumulative_seconds'] = manifest['elapsed_seconds']+prior_seconds
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin > 2500 or time.monotonic()-begin+prior_seconds > 3600:
            raise TimeoutError('fixed source/slot budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+e['total_output_bytes'] > 8<<30:
            raise ValueError('fixed output budget')
        if any(sha(Path(k)) != v for k,v in hashes.items()):
            raise ValueError('input drift')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=admission['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',PYTHONDONTWRITEBYTECODE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),
               XDG_CACHE_HOME=str(output/'cache'))
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    process = None
    save()
    try:
        for seed in range(571,583):
            check()
            directory = output/f'seed{seed}'
            command = list(old['command'])
            changes = {'--output-dir':str(directory),'--panel-seed':str(seed),
                       '--assignment-seed':str(seed+15000),'--windows-per-stratum':'4',
                       '--seed':str(seed),'--output':str(directory/'unused.json'),
                       '--output_path':str(directory/'player')}
            for key,value in changes.items():
                command[command.index(key)+1] = value
            env['PYTHONHASHSEED'] = str(seed)
            phase = dict(seed=seed,directory=str(directory),command=command,run_status='STARTED')
            manifest['phases'].append(phase)
            with (output/f'seed{seed}.log').open('x') as log:
                process = subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,
                                           stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase['pid'] = process.pid
                save()
                child_begin = time.monotonic()
                while process.poll() is None:
                    remaining = 290-(time.monotonic()-child_begin)
                    if remaining <= 0:
                        raise TimeoutError('native phase parent timeout')
                    try:
                        process.wait(timeout=min(30,remaining))
                    except subprocess.TimeoutExpired:
                        check()
            phase['exit_code'] = process.returncode
            if process.returncode:
                raise RuntimeError(f'native seed{seed} exit{process.returncode}')
            result = json.loads((directory/'results.json').read_text())
            if result['run_status'] != 'COMPLETED' or sha(directory/'records.pt') != result['record_sha256']:
                raise ValueError('terminal record contract')
            phase.update(run_status='COMPLETED',result=result)
            check()
            save()
            print(json.dumps(dict(seed=seed,rows=result['rows'],complete_phases=len(manifest['phases']),
                                  cumulative_seconds=manifest['cumulative_seconds'])),flush=True)
        manifest.update(run_status='COMPLETED',child_exit_code=0,input_hashes_unchanged=True)
    except BaseException as exc:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=5)
        if manifest['phases'] and manifest['phases'][-1]['run_status'] != 'COMPLETED':
            manifest['phases'][-1].update(run_status='FAILED',exit_code=process.returncode if process else None)
        manifest.update(run_status='FAILED',error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps(dict(run_status=manifest['run_status'],phases=len(manifest['phases']),
                          cumulative_seconds=manifest['cumulative_seconds'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    run(p.parse_args())
