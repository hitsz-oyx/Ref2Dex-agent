#!/usr/bin/env python3
"""Budgeted isolated native actor-only feasibility screen on held references."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import admission, sha, PYTHON, REFERENCE

PANELS = ((286,498),(286,499),(287,498),(287,499))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=5)
    args = parser.parse_args()
    output, references = args.output.resolve(), args.references.resolve()
    if output.exists() or ROOT not in output.parents:
        raise ValueError('unique independent output required')
    generated = json.loads((references / 'run_manifest.json').read_text())
    if generated['run_status'] != 'COMPLETED' or not generated['synthetic'] or len(generated['references']) != 3:
        raise ValueError('frozen synthetic generation incomplete')
    old = ROOT / 'src/task/CmResidual/research/contact_response/output/P-20261001-randomized-task-selection-r1/run_manifest.json'
    prior = json.loads(old.read_text())
    if prior['run_status'] != 'COMPLETED':
        raise ValueError('prior task not closed')
    hashes = {**prior['input_sha256'], **generated['input_sha256']}
    hashes.update({r['generated']: r['generated_sha256'] for r in generated['references']})
    sources = [Path(__file__), ROOT / 'scripts/run_hold_plateau_environment.py',
        ROOT / 'scripts/analyze_hold_plateau_substrate.py', old, references / 'run_manifest.json']
    hashes.update({str(p.resolve()): sha(p) for p in sources})
    if any(sha(Path(p)) != h for p,h in hashes.items()):
        raise ValueError('frozen input drift')
    original = json.loads((REFERENCE / 'run_manifest.json').read_text())
    templates = {p['training_seed']: p for p in original['phases'] if not p['repeat'] and p['evaluation_seed'] == 288}
    output.mkdir(parents=True)
    begin = time.monotonic()
    manifest = dict(experiment_id='P-20261001-hold-plateau-substrate', run_id=output.name,
        run_status='RUNNING', pid=os.getpid(), isolated_worktree=str(ROOT), synthetic_task=True,
        no_training=True, gpu_index=args.gpu, references=str(references), panels=PANELS, phases=[],
        input_sha256=hashes, wall_limit_seconds=1800, storage_limit_bytes=256*(1<<20),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        manifest['wall_seconds'] = time.monotonic() - begin
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def check():
        if time.monotonic() - begin > 1800:
            raise TimeoutError('whole feasibility probe budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 256*(1<<20):
            raise RuntimeError('storage budget')
        if any(sha(Path(p)) != h for p,h in hashes.items()):
            raise ValueError('protected input drift')
    def execute(name, command, timeout, native=True):
        check()
        gpu = admission(args.gpu) if native else None
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'] if native else '', LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'),XDG_CACHE_HOME=str(output/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase = dict(name=name, command=command, run_status='RUNNING', admission=gpu,
            device_reason='native GPU simulation/inference' if native else 'CPU trajectory labels/statistics, no model compute')
        manifest['phases'].append(phase)
        save()
        process = None
        start = time.monotonic()
        print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        try:
            with (output / (name + '.log')).open('x') as log:
                process = subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase.update(pid=process.pid,pgid=process.pid)
                save()
                code = process.wait(timeout=min(timeout,max(1,1800-(time.monotonic()-begin))))
            if code:
                raise RuntimeError(f'{name} exit {code}; retained log')
            phase['run_status']='COMPLETED'
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=5)
            phase.update(run_status='FAILED',error=repr(error))
            raise
        finally:
            phase['wall_seconds']=time.monotonic()-start
            save()
        print(json.dumps(dict(name=name,status='COMPLETED',wall_seconds=phase['wall_seconds'])),flush=True)
    save()
    try:
        for actor,evaluation in PANELS:
            name=f't{actor}_s{evaluation}'
            directory=output/name
            command=list(templates[actor]['command'])
            command[2]=str(ROOT/'scripts/run_hold_plateau_environment.py')
            position=command.index('--arm')
            del command[position:position+2]
            for flag,value in (('--run-dir',directory),('--eval-seed',evaluation),('--seed',evaluation),('--num_envs',192),
                ('--wall-seconds',300),('--motion_file',references/'references'),
                ('--output',directory/'unused.json'),('--output_path',directory/'player')):
                command[command.index(flag)+1]=str(value)
            command.extend(['--references-manifest',str(references/'run_manifest.json')])
            execute(name,command,400)
            result=json.loads((directory/'results.json').read_text())
            if result['complete_episodes']!=192 or not result['all_episodes_complete']:
                raise ValueError('incomplete panel')
        manifest['run_status']='COLLECTION_COMPLETED'
        save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_hold_plateau_substrate.py'),'--directory',str(output)],120,native=False)
        result=json.loads((output/'results.json').read_text())
        check()
        manifest.update(run_status='COMPLETED',label=result['label'],inputs_unchanged=True)
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error))
        raise
    finally:
        save()


if __name__=='__main__':
    main()
