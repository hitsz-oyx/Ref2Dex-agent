#!/usr/bin/env python3
"""One bounded native generated-action engineering phase; not a science Probe."""
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
from run_paired_evaluator_resolution import sha, gpu_admission


def run(args):
    begin = time.monotonic()
    base = (ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output = args.output.resolve()
    if output.parent != base or output.exists() or args.output.is_symlink():
        raise ValueError('unique owned engineering output required')
    gradient_path = ROOT/'docs/experiments/probes/P-20261002-optimized-contact-actions-engineering-r2.json'
    gradient = json.loads(gradient_path.read_text())
    if gradient['run_status'] != 'COMPLETED' or not gradient['engineering_passed']:
        raise ValueError('gradient engineering required')
    if any(sha(Path(k)) != v for k,v in gradient['input_sha256'].items()):
        raise ValueError('gradient input drift')
    native_path = ROOT/'docs/experiments/probes/P-20261002-contact-geometry-native-engineering-completion-r2.json'
    native = json.loads(native_path.read_text())
    if not native['engineering_passed'] or native['run_status'] != 'COMPLETED':
        raise ValueError('native source engineering required')
    if any(sha(Path(k)) != v for k,v in native['artifact_sha256'].items()):
        raise ValueError('native engineering drift')
    old_path = Path(next(k for k in native['artifact_sha256'] if 'engineering-r2/run_manifest' in k))
    old = json.loads(old_path.read_text())
    if any(sha(Path(k)) != v for k,v in old['input_sha256'].items()):
        raise ValueError('native source input drift')
    hashes = dict(old['input_sha256'], **gradient['input_sha256'])
    for path in [Path(__file__), ROOT/'scripts/collect_optimized_contact_source.py',
                 ROOT/'scripts/audit_optimized_contact_source.py', gradient_path, native_path,
                 ROOT/'docs/experiments/probes/P-20261002-optimized-contact-native-engineering.md']:
        hashes[str(path.resolve())] = sha(path)
    checkpoint = Path(next(k for k in gradient['input_sha256'] if k.endswith('contact_geometry_consequence.pt')))
    admission = gpu_admission(args.gpu)
    directory = output/'seed590'
    command = list(old['command'])
    command[2] = str(ROOT/'scripts/collect_optimized_contact_source.py')
    changes = {'--output-dir':str(directory), '--panel-seed':'590', '--assignment-seed':'15590',
               '--seed':'590', '--output':str(directory/'unused.json'),
               '--output_path':str(directory/'player')}
    for key,value in changes.items():
        command[command.index(key)+1] = value
    command.extend(['--generator-checkpoint',str(checkpoint)])
    output.mkdir(exist_ok=False)
    phase = dict(seed=590, directory=str(directory), command=command, run_status='STARTED')
    manifest = dict(run_status='STARTED', smoke_only=True, scientific_probe_started=False,
        pid=os.getpid(), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashes, gpu=admission, phases=[phase], wall_limit_seconds=300,
        output_limit_bytes=256<<20, model_training=False, expert_training=False,
        scope='native generated-action execution engineering; no utility claim')
    def save():
        manifest['elapsed_seconds'] = time.monotonic()-begin
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin > 290:
            raise TimeoutError('engineering parent timeout')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 256<<20:
            raise ValueError('engineering output budget')
        if any(sha(Path(k)) != v for k,v in hashes.items()):
            raise ValueError('input drift')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=admission['uuid'], OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', LOCAL_RANK='0', RANK='0', WORLD_SIZE='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='590', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'), XDG_CACHE_HOME=str(output/'cache'))
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    process = None
    save()
    try:
        with (output/'native.log').open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            phase['pid'] = process.pid
            save()
            while process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    check()
        phase['exit_code'] = process.returncode
        if process.returncode:
            raise RuntimeError('native child exit '+str(process.returncode))
        result = json.loads((directory/'results.json').read_text())
        if result['run_status'] != 'COMPLETED' or sha(directory/'records.pt') != result['record_sha256']:
            raise ValueError('terminal native record contract')
        if sha(directory/'planning.pt') != result['planning_sha256']:
            raise ValueError('planning hash')
        phase.update(run_status='COMPLETED',result=result)
        check()
        manifest.update(run_status='COMPLETED',child_exit_code=0,input_hashes_unchanged=True)
    except BaseException as exc:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=5)
        phase.update(run_status='FAILED',exit_code=process.returncode if process else None)
        manifest.update(run_status='FAILED',error=repr(exc))
        raise
    finally:
        save()
    print(json.dumps(dict(run_status=manifest['run_status'],elapsed_seconds=manifest['elapsed_seconds'],
                         result=result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=1)
    run(p.parse_args())
