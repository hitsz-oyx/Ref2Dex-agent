#!/usr/bin/env python3
"""Plan or run one bounded native engineering phase; never starts scientific fits."""
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
from run_paired_evaluator_resolution import sha, gpu_admission, PYTHON, R7, MOTIONS


def run(args):
    begin = time.monotonic()
    base = (ROOT/'src/task/CmResidual/research/contact_consequence/output').resolve()
    output = args.run_output.resolve()
    if output.parent != base or args.run_output.exists() or args.run_output.is_symlink():
        raise ValueError('unique owned native output required')
    if args.plan_output.exists():
        raise ValueError('unique plan artifact required')
    offline_path = ROOT/'docs/experiments/probes/P-20261002-contact-geometry-action-engineering-r1.json'
    offline = json.loads(offline_path.read_text())
    if offline['run_status'] != 'COMPLETED' or not offline['engineering_passed']:
        raise ValueError('offline command contract required')
    if any(sha(Path(k)) != v for k, v in offline['input_sha256'].items()):
        raise ValueError('offline engineering source drift')
    record_audit_path = ROOT/'docs/experiments/probes/P-20261002-contact-geometry-record-audit-engineering-r1.json'
    record_audit = json.loads(record_audit_path.read_text())
    if record_audit['run_status'] != 'COMPLETED' or not record_audit['engineering_passed']:
        raise ValueError('independent record audit engineering required')
    if any(sha(Path(k)) != v for k,v in record_audit['input_sha256'].items()):
        raise ValueError('record audit source drift')
    prior_seconds = offline['elapsed_seconds'] + record_audit['elapsed_seconds'] + 3
    route_path = ROOT/'src/task/CmResidual/configs/hf02_temporal_canonical_route.json'
    route = json.loads(route_path.read_text())
    inputs = [offline_path, record_audit_path, route_path, R7/'environment.yaml', R7/'training.yaml', Path(__file__),
              ROOT/'scripts/collect_contact_geometry_source.py', ROOT/'scripts/collect_contact_consequences.py',
              ROOT/'scripts/audit_contact_geometry_source.py', ROOT/'scripts/smoke_contact_geometry_record_audit.py',
              ROOT/'docs/experiments/probes/P-20261002-contact-geometry-source.md']
    inputs += [ROOT/'src/task/CmResidual'/name for name in ['contact_geometry_actions.py','executable_contact_options.py',
                'orientation_anchored_options.py','weight_normalized_contact.py','paired_evaluation.py','physical_value_live.py','contact_consequence.py']]
    inputs += [ROOT/'third_party/DExplore/dexplore'/name for name in ['evaluate.py','env/tasks/dexplore_inspire.py',
                'env/tasks/base_dexplore_task.py','data/assets/mjcf/airplane.urdf','data/assets/mjcf/table.urdf',
                'data/assets/mjcf/objects/airplane/airplane.obj','data/assets/mjcf/objects/table/table.obj']]
    for spec in route['experts'].values():
        path = (ROOT/spec['checkpoint']).resolve()
        if sha(path) != spec['sha256']:
            raise ValueError('expert drift')
        inputs.append(path)
    for spec in route['motions']:
        path = MOTIONS/spec['name']/'interaction_hand_inspire.pt'
        if sha(path) != spec['interaction_hand_sha256']:
            raise ValueError('motion drift')
        inputs.append(path)
    hashes = {str(path.resolve()): sha(path) for path in inputs}
    directory = output/'seed570'
    command = [PYTHON, '-u', str(ROOT/'scripts/collect_contact_geometry_source.py'), '--output-dir', str(directory),
               '--panel-seed','570','--assignment-seed','15570','--windows-per-stratum','1','--max-steps','650','--wall-seconds','240',
               '--task','Dexplore_Inspire','--cfg_env',str(R7/'environment.yaml'),'--cfg_train',str(R7/'training.yaml'),
               '--checkpoint',str((ROOT/route['experts']['source_e260']['checkpoint']).resolve()),'--motion_file',str(MOTIONS),
               '--headless','--num_envs','96','--seed','570','--sim_device','cuda:0','--rl_device','cuda:0',
               '--graphics_device_id','0','--disable-early-termination','--output',str(directory/'unused.json'),
               '--output_path',str(directory/'player')]
    plan = dict(experiment_id='P-20261002-contact-geometry-source', run_status='PLANNED', smoke_only=True,
                scientific_probe_started=False, command=command, input_sha256=hashes,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                run_output=str(output), wall_limit_seconds=600, native_limit_seconds=240, output_limit_bytes=256 << 20,
                prior_engineering_seconds=prior_seconds, gpu=args.gpu,
                scope='one native engineering phase; science/fits/utility/RL require separately fixed cards')
    args.plan_output.write_text(json.dumps(plan,indent=2)+'\n')
    if not args.native:
        print(json.dumps(dict(run_status='PLANNED', native_started=False, input_files=len(hashes), plan=str(args.plan_output),
                             elapsed_seconds=time.monotonic()-begin)))
        return
    admission = gpu_admission(args.gpu)  # Fails before creating a native run if CUDA admission is unavailable.
    output.mkdir(exist_ok=False)
    manifest = dict(plan, run_status='STARTED', pid=os.getpid(), gpu=admission, command=command, phases=[])
    def save():
        (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def check():
        if time.monotonic()-begin+prior_seconds > 600:
            raise TimeoutError('engineering wall limit')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 256 << 20:
            raise ValueError('engineering output limit')
        if any(sha(Path(k)) != v for k,v in hashes.items()):
            raise ValueError('input drift')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=admission['uuid'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               LOCAL_RANK='0', RANK='0', WORLD_SIZE='1', PYTHONDONTWRITEBYTECODE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='570',
               TORCH_EXTENSIONS_DIR=str(output/'cache/torch_extensions'), XDG_CACHE_HOME=str(output/'cache'))
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    process = None
    save()
    try:
        check()
        with (output/'seed570.log').open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest['child_pid'] = process.pid
            save()
            child_begin = time.monotonic()
            while process.poll() is None:
                remaining = 290-(time.monotonic()-child_begin)
                if remaining <= 0:
                    raise TimeoutError('native parent timeout')
                try:
                    process.wait(timeout=min(30,remaining))
                except subprocess.TimeoutExpired:
                    check()
        if process.returncode:
            raise RuntimeError('native exit '+str(process.returncode))
        result = json.loads((directory/'results.json').read_text())
        if result['run_status'] != 'COMPLETED' or result['record_sha256'] != sha(directory/'records.pt'):
            raise ValueError('terminal record contract')
        check()
        manifest.update(run_status='COMPLETED', child_exit_code=process.returncode,
                        phases=[dict(seed=570,run_status='COMPLETED',directory=str(directory),result=result)],
                        cumulative_seconds=time.monotonic()-begin+prior_seconds, input_hashes_unchanged=True)
    except BaseException as exc:
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=5)
        manifest.update(run_status='FAILED',error=repr(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic()-begin
        save()
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['input_sha256','command','phases']},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-output',type=Path,required=True)
    p.add_argument('--plan-output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--native',action='store_true')
    run(p.parse_args())
