#!/usr/bin/env python3
"""Budgeted matched continuous on-policy auxiliary-critic experiment."""
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

PANELS = tuple((286,s) for s in range(547,567)) + ((286,568),(286,569))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
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
    sources = [Path(__file__), ROOT / 'scripts/run_continuous_critic_environment_v2.py',
        ROOT / 'src/task/CmResidual/research/contact_response/output/P-20261002-support-removal-witness-r1/results.json',
        ROOT / 'docs/archive/2026-10-04-research-governance/decisions/../experiments/probes/P-20261002-continuous-critic-policy.md',
        ROOT / 'scripts/audit_continuous_critic_panel.py', old, references / 'run_manifest.json', ROOT / 'src/task/CmResidual/static_hold_feasibility.py', ROOT / 'src/task/CmResidual/reference_target_policy.py', ROOT / 'src/task/CmResidual/support_response.py',
        ROOT / 'docs/archive/2026-10-04-research-governance/decisions/D-20261002-continuous-critic-cm.md',
        ROOT / 'src/task/CmResidual/finger_preload.py', ROOT / 'src/task/CmResidual/frame0_tracking.py', ROOT / 'src/task/CmResidual/observation_hold_policy.py', ROOT / 'src/task/CmResidual/dexplore_bc_policy.py', ROOT / 'scripts/fit_support_response_information.py', ROOT / 'src/task/CmResidual/tabletop_clearance.py',
        ROOT / 'scripts/analyze_static_hold_feasibility.py',
        ROOT / 'third_party/DExplore/dexplore/data/assets/mjcf/table.urdf',
        ROOT / 'third_party/DExplore/dexplore/data/assets/mjcf/objects/table/table.obj']
    policy=old.parent/'fit/policy.pt'
    sources.extend([policy,ROOT/'src/task/CmResidual/native_reset_transaction.py',ROOT/'src/task/CmResidual/continuous_critic_cm.py',ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261002-support-response-initialization-fix.md',Path('/home2/wyy/isaac-gym/isaacgym/docs/programming/tensors.html')])
    sources += [ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261002-natural-retention-headroom.md',ROOT/'docs/archive/2026-10-04-root-research/research/20261002-natural-retention-headroom-results.md',ROOT/'src/task/CmResidual/research/contact_response/output/P-20261002-natural-retention-headroom-r1/results.json']
    sources += [ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf',ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261002-contact-action-representation-review.md']
    sources += [ROOT/'src/task/CmResidual/research/contact_response/output/P-20261002-natural-retention-feedback-r1/results.json',ROOT/'docs/archive/2026-10-04-root-research/research/20261002-natural-retention-feedback-results.md']
    sources += [ROOT/'src/task/CmResidual/selective_finger_response.py',ROOT/'scripts/initialize_continuous_critic_cm.py',ROOT/'scripts/update_continuous_critic_policy.py',ROOT/'scripts/audit_continuous_critic_update.py',ROOT/'scripts/analyze_continuous_critic_policy.py',ROOT/'src/task/CmResidual/research/contact_response/output/P-20261002-continuous-critic-smoke-r1/run_manifest.json',ROOT/'src/task/CmResidual/research/contact_response/output/P-20261002-continuous-critic-smoke-r1/collection_audit.json']
    hashes.update({str(p.resolve()): sha(p) for p in sources})
    if any(sha(Path(p)) != h for p,h in hashes.items()):
        raise ValueError('frozen input drift')
    original = json.loads((REFERENCE / 'run_manifest.json').read_text())
    templates = {p['training_seed']: p for p in original['phases'] if not p['repeat'] and p['evaluation_seed'] == 288}
    output.mkdir(parents=True)
    begin = time.monotonic()
    manifest = dict(experiment_id='P-20261002-continuous-critic-policy', run_id=output.name,
        run_status='RUNNING', pid=os.getpid(), isolated_worktree=str(ROOT), synthetic_task=True,
        no_training=False, fixed_forecast_training=False, no_ppo=False, no_cm_policy_training=False, gpu_index=args.gpu, references=str(references), panels=PANELS, training_seeds=list(range(547,567)),evaluation_seeds=[568,569],panel_checkpoints={},policy_sha256=sha(policy),initial_xy_uniform_mm=10,independent_action_dimensions=12,engineering_only=False,no_external_force=True,base_checkpoint=str(policy),source_actor_weights_used=False, phases=[],
        input_sha256=hashes, wall_limit_seconds=3600, storage_limit_bytes=6<<30,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        manifest['wall_seconds'] = time.monotonic() - begin
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def check():
        if time.monotonic() - begin > 3600:
            raise TimeoutError('whole feasibility probe budget')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 6<<30:
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
            device_reason='native GPU simulation/inference' if native else ('tiny deterministic CPU checkpoint initialization; GPU startup exceeds initialization cost' if name=='initialize' else 'independent CPU NumPy replay of GPU forward, geometry and probability; no fitting'))
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
                code = process.wait(timeout=min(timeout,max(1,3600-(time.monotonic()-begin))))
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
            command=list(templates[286]['command']);command[2]=str(ROOT/'scripts/run_continuous_critic_environment_v2.py')
            position=command.index('--arm');del command[position:position+2]
            for flag,value in (('--run-dir',directory),('--eval-seed',evaluation),('--seed',evaluation),('--num_envs',768),
                ('--wall-seconds',240),('--motion_file',references/'references'),('--output',directory/'unused.json'),('--output_path',directory/'player')):
                command[command.index(flag)+1]=str(value)
            command.extend(['--references-manifest',str(references/'run_manifest.json'),'--mode',mode])
            return command
        execute('initialize',[PYTHON,'-u',str(ROOT/'scripts/initialize_continuous_critic_cm.py'),'--output',str(output/'u00')],60,native=False)
        heads=output/'u00/policy_heads.pt';hashes[str(heads)]=sha(heads);manifest['continuous_checkpoint_sha256']=sha(heads);save()
        for actor,evaluation in PANELS:
            update=evaluation-547 if evaluation<567 else 20
            manifest['panel_checkpoints'][str(evaluation)]=dict(path=str(heads),sha256=sha(heads),update=update)
            save()
            name=f's{evaluation}';directory=output/name;command=native_command(evaluation,directory,'policy')
            command.extend(['--policy-checkpoint',str(policy),'--policy-sha256',sha(policy),'--continuous-checkpoint',str(heads),'--continuous-sha256',sha(heads),'--expected-update',str(update)])
            if evaluation in (568,569):command.append('--deterministic')
            execute(name,command,300)
            result=json.loads((directory/'results.json').read_text())
            if result['mechanical_trajectories']!=768 or result['phase_steps_each']!=90 or result['mode']!='policy':raise ValueError('policy coverage')
            for filename in ('initial.pt','trace.pt','physical_metadata.json','results.json'):
                path=directory/filename;hashes[str(path)]=sha(path)
            save()
            execute(name+'_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_continuous_critic_panel.py'),'--directory',str(output),'--panel',str(evaluation)],300,native=False)
            for filename in ('rows.json','panel_audit.json'):
                path=directory/filename;hashes[str(path)]=sha(path)
            save()
            if evaluation<567:
                ud=output/f'u{update+1:02d}'
                execute(ud.name,[PYTHON,'-u',str(ROOT/'scripts/update_continuous_critic_policy.py'),'--output',str(ud),'--previous',str(heads),'--panel',str(directory)],180)
                for filename in ('policy_heads.pt','update_packet.pt','results.json'):
                    path=ud/filename;hashes[str(path)]=sha(path)
                save()
                execute(ud.name+'_gradient_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_continuous_critic_update.py'),'--directory',str(ud)],180,native=False)
                path=ud/'gradient_audit.json';hashes[str(path)]=sha(path);heads=ud/'policy_heads.pt';save()
        manifest['run_status']='COLLECTION_COMPLETED';manifest['final_checkpoint_sha256']=sha(heads);save()
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_continuous_critic_policy.py'),'--directory',str(output)],300,native=False)
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
