"""Wait for the owned PointWorld launcher, then run native smoke and parent fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.contracts import is_within


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
    p.add_argument('--wait-status', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    inputs, output = a.inputs.resolve(), a.output.resolve()
    if not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists():
        p.error('fresh task-owned output required')
    parent = json.loads(a.wait_status.read_text())
    parent_pid = parent['pid']
    cmdline = Path('/proc')/str(parent_pid)/'cmdline'
    if (not cmdline.is_file() or str(ROOT/'src/task/cm-pointflow-effect-pretrain/tools/run/launch_pointworld_action_ddp.py')
            not in cmdline.read_bytes().decode().replace('\0',' ')):
        raise ValueError('wait target is not a confirmed live owned PointWorld launcher')
    stage = json.loads((inputs/'manifest.json').read_text())
    if stage['status'] != 'STAGED_CPU_CONTRACT_PASS':
        raise ValueError('staged input validation is incomplete')
    output.mkdir(parents=True)
    source_paths = [Path(__file__), TASK/'src/consequence_evaluator/contracts.py',
                    TASK/'tools/run/rebuild_train.py', TASK/'tools/run/rebuild_rank_bootstrap.py',
                    ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
                    ROOT/'src/task/CmResidual/tools/run_multitrajectory_baseline_probe.py',
                    *sorted((ROOT/'src/task/CmResidual/tools').glob('dexplore_ddp*.py')),
                    ROOT/'src/task/CmResidual/tools/dexplore_cm_off_rank_bootstrap.py',
                    *[ROOT/'src/task/CmResidual'/name for name in (
                        'dexplore_approach.py','dexplore_approach_agent.py','dexplore_cm_geometry.py',
                        'dexplore_contact_curriculum.py','dexplore_grasp_reward.py')],
                    *sorted((ROOT/'third_party/DExplore/dexplore').rglob('*.py'))]
    frozen = {str(path):sha(path) for path in source_paths}
    frozen.update({record['path']:record['sha256'] for record in stage['motion_inputs']})
    frozen.update({str(inputs/path):value for key in ('assets','configs') for path,value in stage[key].items()})
    frozen[str(inputs/'manifest.json')] = sha(inputs/'manifest.json')
    record = dict(status='WAITING', task='consequence-evaluator', run_id=output.name, pid=os.getpid(),
                  parent_pid=parent_pid, physical_gpu=a.gpu, sources=frozen, seed=a.seed,
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                  scope='new parent_s1 scratch PPO; native smoke first; further six-expert training pending parent evaluation',
                  wait_budget_s=3600, fit_budget_s=3600, output_budget_bytes=5*2**30)
    path = output/'run_manifest.json'
    write(path,record)
    started = time.monotonic()
    process = None
    scratch = ROOT/'tmp/consequence-baseline-rebuild'
    scratch.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch), TORCH_EXTENSIONS_DIR=str(scratch/'torch-extensions'),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    try:
        while cmdline.is_file():
            if time.monotonic()-started > 3600:
                raise TimeoutError('fixed wait deadline')
            time.sleep(20)
        terminal = json.loads(a.wait_status.read_text())
        if terminal['status'] != 'COMPLETED':
            raise RuntimeError('PointWorld did not complete; inspect its terminal state before reuse')
        occupied = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
                                           '--format=csv,noheader'],text=True).strip()
        if occupied:
            raise RuntimeError('GPU occupied after parent completion: '+occupied)
        for name, digest in frozen.items():
            if sha(name) != digest:
                raise RuntimeError('queued input/source drift: '+name)
        for phase, epochs, seed, count, anneal in [('smoke',2,17,8,[]),
                ('parent_s1',200,a.seed,64,['--anneal-start','40','--anneal-end','80'])]:
            occupied = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
                                               '--format=csv,noheader'],text=True).strip()
            if occupied:
                raise RuntimeError('GPU occupied before '+phase+': '+occupied)
            command = [sys.executable,str(TASK/'tools/run/rebuild_train.py'),
                '--output',str(output/phase),'--gpu',str(a.gpu),'--from-scratch','--source-epoch','0',
                '--target-epoch',str(epochs),'--spec',str(inputs/'specs/parent_s1.json'),
                '--cfg-env',str(inputs/'cfg/inspire_object_balanced.yaml'),
                '--num-envs',str(count),'--minibatch-size','256','--seed',str(seed),
                '--save-frequency','2' if phase=='smoke' else '20',*anneal]
            record.update(status=phase.upper(), command=command)
            write(path,record)
            with (output/(phase+'.log')).open('w',buffering=1) as log:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record['child_pid']=process.pid
                write(path,record)
                phase_started=time.monotonic()
                while process.poll() is None:
                    limit=300 if phase=='smoke' else 3600
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
