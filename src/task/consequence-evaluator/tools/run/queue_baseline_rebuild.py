"""Bounded native smoke/parent fit; optionally wait for an owned launcher."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.contracts import is_within
from consequence_evaluator.provenance import endpoint_epoch, self_trained_ancestry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait-status', type=Path)
    p.add_argument('--env-config', type=Path)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--source-run', type=Path,
                   help='completed owned self-trained run for a bounded transfer')
    p.add_argument('--expert', choices=['parent_s1','airplane_base','mixed12','train5','balanced5','duck','cup'], default='parent_s1')
    p.add_argument('--target-epoch', type=int, default=200)
    p.add_argument('--continue-unqualified', action='store_true',
                   help='explicit same-input s3 continuation after a failed readiness gate; <=40epochs')
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    if a.gpu < 0 or a.seed < 0 or not math.isfinite(a.learning_rate) or a.learning_rate <= 0:
        p.error('nonnegative GPU/seed and finite positive learning rate required')
    if a.continue_unqualified and a.expert!='airplane_base':
        p.error('unqualified continuation remains limited to the explicit same-input s3 protocol')
    inputs, output = a.inputs.resolve(), a.output.resolve()
    if not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists():
        p.error('fresh task-owned output required')
    if a.wait_status and not is_within(a.wait_status.resolve(), ROOT/'outputs/cm-pointflow-effect-pretrain'):
        raise ValueError('wait status must belong to the owned PointWorld task')
    parent = json.loads(a.wait_status.read_text()) if a.wait_status else dict(pid=None,status='COMPLETED',exit_code=0,progress=dict(status='COMPLETED'))
    parent_pid = parent['pid']
    cmdline = Path('/proc')/str(parent_pid)/'cmdline'
    live_parent = (a.wait_status is not None and cmdline.is_file() and
        str(ROOT/'src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py')
        in cmdline.read_bytes().decode().replace('\0',' '))
    completed_parent = (parent.get('status') == 'COMPLETED' and parent.get('exit_code') == 0
                        and parent.get('progress', {}).get('status') == 'COMPLETED')
    if not live_parent and not completed_parent:
        raise ValueError('wait target must be a live owned launcher or its successful terminal record')
    stage = json.loads((inputs/'manifest.json').read_text())
    env_config = a.env_config.resolve() if a.env_config else inputs/'cfg/inspire_object_balanced.yaml'
    if stage['status'] != 'STAGED_CPU_CONTRACT_PASS':
        raise ValueError('staged input validation is incomplete')
    ancestry, source, source_epoch = {}, None, 0
    if a.source_run:
        source_run = a.source_run.resolve()
        ancestry = self_trained_ancestry(source_run, ROOT/'outputs/consequence-evaluator')
        source = json.loads((source_run/'run_manifest.json').read_text())
        source_epoch = endpoint_epoch(source)
        extra = 40 if a.continue_unqualified else 20
        if a.expert == 'parent_s1' or not source_epoch < a.target_epoch <= min(500,source_epoch + extra):
            p.error('qualified-source expert transfer is bounded to20epochs; explicit same-input s3 continuation to40')
        qualification_dir = source_run.parent/'qualification-s290'
        qualification_manifest = json.loads((qualification_dir/'run_manifest.json').read_text())
        qualification = json.loads((qualification_dir/'qualification.json').read_text())
        if (qualification_manifest.get('status') != 'COMPLETED' or
                qualification_manifest.get('checkpoint_sha256') != source['checkpoint_sha256'] or
                qualification_manifest.get('reference_action_lead') is not None or
                qualification.get('episodes') != 64 or
                len(qualification.get('per_episode',[])) != 64 or
                qualification.get('qualified_episodes') != sum(e['qualified'] for e in qualification['per_episode'])):
            raise ValueError('source needs its own completed learned-policy qualification')
        if a.continue_unqualified:
            if qualification.get('data_readiness_pass') or source.get('run_id') != a.expert:
                raise ValueError('unqualified continuation is only for the evaluated same s3 expert')
            if a.learning_rate != source.get('learning_rate'):
                raise ValueError('unqualified continuation preserves the evaluated learning rate')
            spec=json.loads((inputs/('specs/'+a.expert+'.json')).read_text())
            previous_inputs=json.loads(Path(source['input_manifest']).read_text())
            current={str((ROOT/Path(item)).resolve()):sha(ROOT/Path(item)/'interaction_hand_inspire.pt')
                     for item in spec['motions']}
            previous={str(Path(item['path']).resolve()):item['tensor_sha256']
                      for item in previous_inputs['motions']}
            if current != previous or sha(env_config) != source['env_config_sha256']:
                raise ValueError('unqualified continuation cannot change motion inputs/environment')
            ancestry[str(Path(source['input_manifest']).resolve())]=sha(source['input_manifest'])
        elif not qualification.get('data_readiness_pass') or qualification.get('qualified_episodes',0) < 8:
            raise ValueError('initial transfer source must pass the operational readiness gate')
        ancestry.update({str(qualification_dir/name):sha(qualification_dir/name)
                         for name in ['run_manifest.json','qualification.json']})
    elif a.expert != 'parent_s1' or a.target_epoch != 200 or a.continue_unqualified:
        p.error('scratch queue is fixed to the200epoch parent')
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('disk below20GiB reserve')
    output.mkdir(parents=True)
    source_paths = [Path(__file__), env_config,inputs/('specs/'+a.expert+'.json'),
                    *[TASK/'src/consequence_evaluator'/name for name in (
                        'contracts.py','provenance.py','native_reset.py','reset_kinematics.py','native_approach.py')],
                    TASK/'tools/run/rebuild_train.py', TASK/'tools/run/rebuild_rank_bootstrap.py',
                    ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
                    ROOT/'src/task/CmResidual/tools/run_multitrajectory_baseline_probe.py',
                    *sorted((ROOT/'src/task/CmResidual/tools').glob('dexplore_ddp*.py')),
                    ROOT/'src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py',
                    *[ROOT/'src/task/CmResidual'/name for name in (
                        'dexplore_approach.py','dexplore_approach_agent.py','dexplore_cm_geometry.py',
                        'dexplore_contact_curriculum.py','dexplore_grasp_reward.py','v118_planner.py')],
                    ROOT/'third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py',
                    *sorted((ROOT/'third_party/DExplore/dexplore').rglob('*.py'))]
    source_paths += [ROOT/'third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml',
                    *sorted(p for p in (ROOT/'third_party/DExplore/dexplore/data/assets').rglob('*') if p.is_file())]
    frozen = {str(path):sha(path) for path in source_paths}
    frozen.update({record['path']:record['sha256'] for record in stage['motion_inputs']})
    frozen.update({str(inputs/path):value for key in ('assets','configs') for path,value in stage[key].items()})
    frozen[str(inputs/'manifest.json')] = sha(inputs/'manifest.json')
    frozen.update(stage.get('recovery_dependencies',{}))
    frozen.update(ancestry)
    record = dict(status='WAITING', task='consequence-evaluator', run_id=output.name, pid=os.getpid(),
                  parent_pid=parent_pid, physical_gpu=a.gpu, sources=frozen, seed=a.seed,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                  scope='bounded expert transfer from qualified self-trained source' if source else
                        'new parent_s1 scratch PPO; native smoke first; further six-expert training pending parent evaluation',
                  trained_run_dir=str(output/a.expert), expert=a.expert,
                  source_run=str(a.source_run.resolve()) if a.source_run else None,
                  continue_unqualified=a.continue_unqualified,
                  source_epoch=source_epoch, target_epoch=a.target_epoch,
                  learning_rate=a.learning_rate, env_config=str(env_config),
                  wait_budget_s=3600, smoke_budget_s=300,
                  fit_budget_s=900 if source else 3600, output_budget_bytes=5*2**30)
    path = output/'run_manifest.json'
    write(path,record)
    started = time.monotonic()
    process = None
    scratch = ROOT/'tmp/consequence-baseline-rebuild'
    scratch.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch), TORCH_EXTENSIONS_DIR=str(scratch/'torch-extensions'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    def stopped(signum, frame):
        raise KeyboardInterrupt('owned rebuild queue stopped')
    signal.signal(signal.SIGTERM, stopped)
    try:
        while live_parent and cmdline.is_file():
            if time.monotonic()-started > 3600:
                raise TimeoutError('fixed wait deadline')
            time.sleep(20)
        terminal = json.loads(a.wait_status.read_text()) if a.wait_status else parent
        if terminal['status'] != 'COMPLETED':
            raise RuntimeError('PointWorld did not complete; inspect its terminal state before reuse')
        occupied = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
                                           '--format=csv,noheader'],text=True).strip()
        if occupied:
            raise RuntimeError('GPU occupied after parent completion: '+occupied)
        for name, digest in frozen.items():
            if sha(name) != digest:
                raise RuntimeError('queued input/source drift: '+name)
        phases = [('smoke',source_epoch+2,17,8,
                   ['--anneal-start','40','--anneal-end','80'] if source else []),
                  (a.expert,a.target_epoch,a.seed,64,['--anneal-start','40','--anneal-end','80'])]
        for phase, epochs, seed, count, anneal in phases:
            occupied = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
                                               '--format=csv,noheader'],text=True).strip()
            if occupied:
                raise RuntimeError('GPU occupied before '+phase+': '+occupied)
            command = [sys.executable,str(TASK/'tools/run/rebuild_train.py'),
                '--output',str(output/phase),'--gpu',str(a.gpu),'--source-epoch',str(source_epoch),
                '--target-epoch',str(epochs),'--spec',str(inputs/('specs/'+a.expert+'.json')),
                '--cfg-env',str(env_config),'--learning-rate',str(a.learning_rate),
                '--num-envs',str(count),'--minibatch-size','256','--seed',str(seed),
                '--save-frequency','2' if phase=='smoke' else '20',*anneal]
            command += (['--source-checkpoint',source['checkpoint'],
                         '--source-sha256',source['checkpoint_sha256']] if source else ['--from-scratch'])
            record.update(status=phase.upper(), command=command)
            write(path,record)
            with (output/(phase+'.log')).open('w',buffering=1) as log:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record['child_pid']=process.pid
                write(path,record)
                phase_started=time.monotonic()
                while process.poll() is None:
                    limit=300 if phase=='smoke' else 900 if source else 3600
                    if time.monotonic()-phase_started > limit:
                        raise TimeoutError('fixed '+phase+' deadline')
                    if any(sha(name)!=digest for name,digest in frozen.items()):
                        raise RuntimeError('implementation/input drift during native fit')
                    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 5*2**30:
                        raise RuntimeError('rebuild output budget exceeded5GiB')
                    time.sleep(20)
                if process.returncode:
                    raise RuntimeError(phase+' failed: exit '+str(process.returncode))
        record['status']='PARENT_TRAINED_EVALUATION_PENDING'
    except BaseException as error:
        record.update(status='TIMED_OUT' if isinstance(error,TimeoutError) else 'FAILED',error=repr(error))
        raise
    finally:
        if process is not None:
            # The process group was created here and contains only this run's children.
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
        record['elapsed_s']=time.monotonic()-started
        write(path,record)


if __name__ == '__main__':
    main()
