"""Evaluate one frozen self-trained parent from full frame0 native starts."""
import argparse
import hashlib
import json
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


class QualificationDeadline(BaseException):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=290)
    p.add_argument('--seconds', type=int, default=900)
    a = p.parse_args()
    output, run = a.output.resolve(), a.run_dir.resolve()
    if (not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists()
            or not is_within(run, ROOT/'outputs/consequence-evaluator')
            or not 1 <= a.seconds <= 900 or a.gpu < 0 or a.seed < 0):
        p.error('fresh task-owned output, native run, GPU and <=900s required')
    trained = json.loads((run/'run_manifest.json').read_text())
    config = json.loads((run/'config.json').read_text())
    if (trained.get('run_status') != 'COMPLETED' or trained.get('cm_enabled') is not False
            or trained.get('initialization') != 'random_scratch'):
        raise ValueError('completed self-trained Cm-off scratch parent required')
    checkpoint = Path(trained['checkpoint'])
    if sha(checkpoint) != trained['checkpoint_sha256']:
        raise ValueError('parent checkpoint identity changed')
    occupied = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                  '--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied before native initialization: '+occupied)
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('disk below20GiB reserve')
    native_root = ROOT/'third_party/DExplore/dexplore'
    files = [Path(__file__), run/'run_manifest.json', run/'config.json', checkpoint,
             Path(config['input_manifest']), Path(config['cfg_env']),
             native_root/'data/cfg/train/rlg/inspire.yaml',
             *sorted(p for p in (native_root/'data/assets').rglob('*') if p.is_file()),
             *sorted((TASK/'src/consequence_evaluator').glob('*.py')),
             *sorted(native_root.rglob('*.py')),
             ROOT/'src/task/CmResidual/tools/dexplore_ddp_rank_bootstrap.py']
    inputs = json.loads(Path(config['input_manifest']).read_text())
    for item in inputs['motions']:
        path = Path(item['path'])/'interaction_hand_inspire.pt'
        if sha(path) != item['tensor_sha256']:
            raise ValueError('native motion identity changed')
        files.append(path)
    frozen = {str(path.resolve()):sha(path) for path in files}
    output.mkdir(parents=True)
    scratch = ROOT/'tmp/consequence-parent-qualification'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch/'torch-extensions'),
                      PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(native_root), str(ROOT), str(ROOT/'src/task/CmResidual/tools')]
    # Install NumPy aliases without Torch, then let the native player load Isaac.
    import dexplore_ddp_rank_bootstrap
    import evaluate as native
    from env.tasks.base_dexplore_task import DexploreTask
    from consequence_evaluator.qualification import qualify_transitions
    import torch
    base = native.EvalPlayer
    class FullStartPlayer(base):
        def run(self):
            task = self.env.task
            if abs(task.dt-1/30) > 1e-8 or task.num_envs != 64 or self.is_rnn:
                raise ValueError('fixed native30Hz/64env/nonrecurrent protocol required')
            task._state_init = DexploreTask.StateInit.Start
            task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False
            task._enable_early_termination = False
            super().run()
    native.EvalPlayer = FullStartPlayer
    argv = ['--task','Dexplore_Inspire','--cfg_env',config['cfg_env'],
            '--cfg_train',str(native_root/'data/cfg/train/rlg/inspire.yaml'),
            '--motion_file',config['motion_root'],'--checkpoint',str(checkpoint),
            '--disable-early-termination','--headless','--sim_device','cuda:0',
            '--rl_device','cuda:0','--pipeline','gpu','--graphics_device_id','0',
            '--num_envs','64','--seed',str(a.seed),'--output',str(output/'native-results.json'),
            '--output_path',str(output/'native-runtime'),
            '--transition-output',str(output/'transitions.pt')]
    manifest = dict(status='RUNNING', task='consequence-evaluator', run_id=output.name,
                    pid=os.getpid(), physical_gpu=a.gpu, seed=a.seed, episodes=64,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    training_commit=trained['git_commit'], checkpoint_sha256=sha(checkpoint),
                    sources=frozen, seconds_budget=a.seconds, output_budget_bytes=2**30,
                    full_frame0=True, early_termination_disabled=True, command=argv)
    write(output/'run_manifest.json',manifest)
    started, old_argv, old_cwd = time.monotonic(), sys.argv, Path.cwd()
    old_signal = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise QualificationDeadline('fixed native qualification deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.seconds)
    try:
        sys.argv=[sys.argv[0],*argv]
        os.chdir(ROOT/'third_party/DExplore')
        native.main()
        payload = torch.load(output/'transitions.pt',map_location='cpu',weights_only=False)
        episodes = json.loads((output/'native-results.json').read_text())['per_episode']
        result=qualify_transitions(payload,episodes)
        if any(sha(path)!=value for path,value in frozen.items()):
            raise RuntimeError('qualification source/input drift')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) > 2**30:
            raise RuntimeError('qualification output exceeded1GiB')
        result.update(transitions_sha256=sha(output/'transitions.pt'),
                      native_results_sha256=sha(output/'native-results.json'))
        write(output/'qualification.json',result)
        manifest.update(status='COMPLETED',data_readiness_pass=result['data_readiness_pass'],
                        qualified_episodes=result['qualified_episodes'])
        print(json.dumps({k:manifest[k] for k in ('status','data_readiness_pass','qualified_episodes')}))
    except BaseException as error:
        manifest.update(status='TIMED_OUT' if isinstance(error,QualificationDeadline) else 'FAILED',error=repr(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,old_signal)
        sys.argv=old_argv
        os.chdir(old_cwd)
        manifest['elapsed_s']=time.monotonic()-started
        write(output/'run_manifest.json',manifest)


if __name__ == '__main__':
    main()
