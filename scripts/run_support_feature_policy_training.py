#!/usr/bin/env python3
"""Sequential fresh off-policy collection, matched learning, and native evaluation."""
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

PANELS = tuple((286,s) for s in range(529,543))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--physical-source',type=Path,required=True)
    args = parser.parse_args()
    physical_source=args.physical_source.resolve()
    output, references = args.output.resolve(), args.references.resolve()
    if output.exists() or ROOT not in output.parents:
        raise ValueError('unique independent output required')
    generated = json.loads((references / 'run_manifest.json').read_text())
    if generated['run_status'] != 'COMPLETED' or not generated['synthetic'] or len(generated['references']) != 3:
        raise ValueError('frozen synthetic generation incomplete')
    old = ROOT / 'src/task/CmResidual/research/contact_response/output/P-20261002-reference-target-policy-r1/run_manifest.json'
    prior = json.loads(old.read_text())
    if prior['run_status'] != 'COMPLETED' or prior['label'] != 'PROMISING':
        raise ValueError('prior task not closed')
    hashes = {**prior['input_sha256'], **generated['input_sha256']}
    hashes.update({r['generated']: r['generated_sha256'] for r in generated['references']})
    sources = [Path(__file__), ROOT / 'scripts/run_support_response_environment_v2.py',
        ROOT / 'src/task/CmResidual/research/contact_response/output/P-20261002-support-removal-witness-r1/results.json',
        ROOT / 'docs/decisions/D-20261002-support-response-information.md',
        ROOT / 'scripts/audit_support_response_collection_v2.py', old, references / 'run_manifest.json', ROOT / 'src/task/CmResidual/static_hold_feasibility.py', ROOT / 'src/task/CmResidual/reference_target_policy.py', ROOT / 'src/task/CmResidual/support_response.py',
        ROOT / 'docs/experiments/probes/P-20261002-support-response-information.md',
        ROOT / 'src/task/CmResidual/finger_preload.py', ROOT / 'src/task/CmResidual/frame0_tracking.py', ROOT / 'src/task/CmResidual/observation_hold_policy.py', ROOT / 'src/task/CmResidual/dexplore_bc_policy.py', ROOT / 'scripts/fit_support_response_information.py', ROOT / 'src/task/CmResidual/tabletop_clearance.py',
        ROOT / 'scripts/analyze_static_hold_feasibility.py',
        ROOT / 'third_party/DExplore/dexplore/data/assets/mjcf/table.urdf',
        ROOT / 'third_party/DExplore/dexplore/data/assets/mjcf/objects/table/table.obj']
    sources.extend([ROOT/'src/task/CmResidual/support_feature_bank.py',ROOT/'src/task/CmResidual/support_feature_policy.py',
        ROOT/'scripts/update_support_feature_policy.py',ROOT/'scripts/audit_support_feature_panel.py',
        ROOT/'scripts/audit_support_feature_update.py',ROOT/'scripts/run_support_feature_evaluation.py',ROOT/'scripts/analyze_support_feature_evaluation.py',
        ROOT/'docs/decisions/D-20261002-support-feature-policy-training.md',ROOT/'docs/experiments/probes/P-20261002-support-feature-policy-training.md'])
    sources.extend(physical_source/f for f in ('results.json','dataset.pt','fit/cm.pt','fit/state_only.pt','fit/predictions.pt'))
    policy=old.parent/'fit/policy.pt'
    sources.extend([policy,ROOT/'src/task/CmResidual/native_reset_transaction.py',ROOT/'docs/decisions/D-20261002-support-response-initialization-fix.md',Path('/home2/wyy/isaac-gym/isaacgym/docs/programming/tensors.html')])
    hashes.update({str(p.resolve()): sha(p) for p in sources})
    if any(sha(Path(p)) != h for p,h in hashes.items()):
        raise ValueError('frozen input drift')
    original = json.loads((REFERENCE / 'run_manifest.json').read_text())
    templates = {p['training_seed']: p for p in original['phases'] if not p['repeat'] and p['evaluation_seed'] == 288}
    output.mkdir(parents=True)
    begin = time.monotonic()
    manifest = dict(experiment_id='P-20261002-support-feature-policy-training', run_id=output.name,
        run_status='RUNNING', pid=os.getpid(), isolated_worktree=str(ROOT), synthetic_task=True,
        no_training=False, fixed_forecast_training=False, no_ppo=True, no_cm_policy_training=False, gpu_index=args.gpu, references=str(references), panels=PANELS, training_seeds=list(range(529,541)),evaluation_seeds=[541,542],training_seed=752,updates=12,physical_source=str(physical_source),policy_sha256=sha(policy),initial_xy_uniform_mm=10,primitive_count=8,source_actor_weights_used=False, phases=[],
        input_sha256=hashes, wall_limit_seconds=1800, storage_limit_bytes=3<<30,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        manifest['wall_seconds'] = time.monotonic() - begin
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def check():
        if time.monotonic() - begin > 1800:
            raise TimeoutError('whole feasibility probe budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 3<<30:
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
        def native_command(evaluation,directory,mode):
            command=list(templates[286]['command']);command[2]=str(ROOT/'scripts/run_support_response_environment_v2.py')
            position=command.index('--arm');del command[position:position+2]
            for flag,value in (('--run-dir',directory),('--eval-seed',evaluation),('--seed',evaluation),('--num_envs',768),
                ('--wall-seconds',240),('--motion_file',references/'references'),('--output',directory/'unused.json'),('--output_path',directory/'player')):
                command[command.index(flag)+1]=str(value)
            command.extend(['--references-manifest',str(references/'run_manifest.json'),'--mode',mode])
            return command
        initialize=output/'u00'
        execute('initialize',[PYTHON,'-u',str(ROOT/'scripts/update_support_feature_policy.py'),'--source',str(physical_source),'--output',str(initialize)],60,native=False)
        previous=initialize/'policy_heads.pt';hashes[str(previous)]=sha(previous)
        for update,evaluation in enumerate(range(529,541),1):
            name=f's{evaluation}';directory=output/name;command=native_command(evaluation,directory,'policy')
            command.extend(['--policy-checkpoint',str(policy),'--policy-sha256',sha(policy)])
            execute(name,command,300)
            for filename in ('initial.pt','trace.pt','physical_metadata.json','results.json'):
                path=directory/filename;hashes[str(path)]=sha(path)
            execute('audit-'+name,[PYTHON,'-u',str(ROOT/'scripts/audit_support_feature_panel.py'),'--directory',str(directory),'--seed',str(evaluation)],180,native=False)
            for filename in ('audited.pt','panel_audit.json'):
                path=directory/filename;hashes[str(path)]=sha(path)
            update_dir=output/f'u{update:02d}'
            execute(f'update-{update:02d}',[PYTHON,'-u',str(ROOT/'scripts/update_support_feature_policy.py'),'--source',str(physical_source),'--previous',str(previous),'--panel',str(directory),'--output',str(update_dir)],120)
            previous=update_dir/'policy_heads.pt';hashes[str(previous)]=sha(previous);hashes[str(update_dir/'update_packet.pt')]=sha(update_dir/'update_packet.pt')
            execute(f'gradient-{update:02d}',[PYTHON,'-u',str(ROOT/'scripts/audit_support_feature_update.py'),'--directory',str(update_dir)],120,native=False)
            hashes[str(update_dir/'gradient_audit.json')]=sha(update_dir/'gradient_audit.json')
            manifest['last_update']=update;save()
        for evaluation in (541,542):
            name=f's{evaluation}';directory=output/name;command=native_command(evaluation,directory,'policy')
            command[2]=str(ROOT/'scripts/run_support_feature_evaluation.py')
            command.extend(['--policy-checkpoint',str(policy),'--policy-sha256',sha(policy),'--macro-checkpoint',str(previous),'--macro-sha256',sha(previous),'--physical-source',str(physical_source)])
            execute(name,command,300)
            for filename in ('initial.pt','trace.pt','physical_metadata.json','results.json','decisions.pt'):
                path=directory/filename;hashes[str(path)]=sha(path)
            execute('audit-'+name,[PYTHON,'-u',str(ROOT/'scripts/audit_support_feature_panel.py'),'--directory',str(directory),'--seed',str(evaluation),'--evaluation'],180,native=False)
            for filename in ('audited.pt','panel_audit.json'):
                path=directory/filename;hashes[str(path)]=sha(path)
        execute('evaluation',[PYTHON,'-u',str(ROOT/'scripts/analyze_support_feature_evaluation.py'),'--directory',str(output)],180,native=False)
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
