"""Screen an explicitly identified official data generator against our actor.

Uses the existing native full-frame0 readiness metric, not evaluator labels.
No weight updates and no admission of official weights as our final policy.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
OFFICIAL = Path('/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth')
OFFICIAL_SHA = '8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553'
RUNTIME = Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train')
ISAAC = Path('/home2/wyy/isaac-gym/isaacgym/python')
PACKAGES = RUNTIME/'lib/python3.8/site-packages'


class Deadline(BaseException):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def install_legacy_player_compat(native, common_player, base_player):
    """Translate the repo's newer player entry to the archived1.1.4 contract.

    The legacy BasePlayer already applies external observation RMS. Keeping
    our newer compatibility override would incorrectly normalize twice.
    """
    def create_player(runner):
        return runner.player_factory.create(runner.algo_name, params=runner.config)
    native.Runner.create_player = create_player
    common_player.CommonPlayer._preproc_obs = base_player.BasePlayer._preproc_obs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor', choices=('official', 'self-trained'), required=True)
    p.add_argument('--run-dir', type=Path, required=True, help='owned frozen actor input configuration')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, default=239)
    p.add_argument('--seconds', type=int, default=900)
    a = p.parse_args()
    sys.path.insert(0, str(TASK/'src'))
    from consequence_evaluator.contracts import is_within
    from consequence_evaluator.provenance import self_trained_ancestry
    output, run = a.output.resolve(), a.run_dir.resolve()
    if (not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists()
            or not is_within(run, ROOT/'outputs/consequence-evaluator')
            or not 1 <= a.seconds <= 900 or a.gpu < 0 or not 100 <= a.seed <= 299
            or Path(sys.prefix).resolve() != RUNTIME.resolve()):
        p.error('fresh task output, owned input run, dedicated runtime and bounded Probe required')
    config = json.loads((run/'config.json').read_text())
    trained = json.loads((run/'run_manifest.json').read_text())
    ancestry = self_trained_ancestry(run, ROOT/'outputs/consequence-evaluator')
    checkpoint = OFFICIAL if a.actor == 'official' else Path(trained['checkpoint'])
    checkpoint_sha = OFFICIAL_SHA if a.actor == 'official' else trained['checkpoint_sha256']
    if sha(checkpoint) != checkpoint_sha:
        raise ValueError('checkpoint identity changed')
    motion_root = Path(config['motion_root'])
    motions = sorted(motion_root.glob('*/interaction_hand_inspire.pt'))
    if len(motions) != 1:
        raise ValueError('screen requires exactly one frozen airplane motion')
    input_manifest = json.loads(Path(config['input_manifest']).read_text())
    if (len(input_manifest['motions']) != 1
            or input_manifest['motions'][0]['tensor_sha256'] != sha(motions[0])):
        raise ValueError('input motion manifest identity mismatch')
    occupied = subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
                  '--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied before initialization: '+occupied)
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('disk below20GiB reserve')
    native_root = ROOT/'third_party/DExplore/dexplore'
    helper = ROOT/'src/task/cm-interaction-oracle/tools'
    # The preserved runtime includes a mutually compatible Torch/rl_games/
    # torch_cluster stack. PYTHONNOUSERSITE prevents newer user packages
    # shadowing it; neither installed environment is modified.
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ISAAC),str(native_root),str(ROOT),
                   str(ROOT/'src/task/CmResidual/tools'),str(helper)]
    scratch = ROOT/'tmp/consequence-official-generator'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),
                      TORCH_EXTENSIONS_DIR=str(scratch/'torch201-extensions'),
                      PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
    files = [Path(__file__), checkpoint,run/'run_manifest.json',run/'config.json',
             Path(config['input_manifest']),Path(config['cfg_env']),*motions,
             native_root/'data/cfg/train/rlg/inspire.yaml',
             *sorted(native_root.rglob('*.py')),
             *sorted(p for p in (native_root/'data/assets').rglob('*') if p.is_file()),
             *sorted((TASK/'src/consequence_evaluator').glob('*.py')),
             *sorted((PACKAGES/'rl_games').rglob('*.py')),
             ROOT/'src/task/CmResidual/tools/dexplore_ddp_rank_bootstrap.py']
    frozen = {str(path.absolute()): sha(path) for path in files}
    frozen.update(ancestry)
    output.mkdir(parents=True)
    manifest = dict(status='INITIALIZING',task='consequence-evaluator',run_id=output.name,
                    experiment_id='P-20261008-official-generator-screen',
                    work_version='official-generator-screen',actor=a.actor,
                    source_actor_role='official_data_generator_screen' if a.actor=='official' else 'self_trained_control',
                    training_allowed=False,pid=os.getpid(),physical_gpu=a.gpu,seed=a.seed,
                    checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
                    episodes=64,full_frame0=True,early_termination_disabled=True,
                    seconds_budget=a.seconds,output_budget_bytes=2**30,sources=frozen,
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(output/'run_manifest.json',manifest)
    started = time.monotonic()
    def deadline(signum, frame):
        raise Deadline('fixed generator-screen deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(a.seconds)
    old_cwd,old_argv = Path.cwd(),sys.argv
    try:
        import dexplore_ddp_rank_bootstrap
        import evaluate as native
        import torch
        if (torch.__version__ != '2.0.1+cu118'
                or not is_within(Path(torch.__file__),RUNTIME)):
            raise RuntimeError('archived Torch2.0.1+cu118 runtime was shadowed')
        import rl_games
        if (importlib.metadata.version('rl_games') != '1.1.4'
                or not is_within(Path(rl_games.__file__),RUNTIME)):
            raise RuntimeError('archived rl_games1.1.4 runtime was shadowed')
        from consequence_evaluator.native_reset import install_reset_patch
        from consequence_evaluator.qualification import qualify_transitions
        from env.tasks.base_dexplore_task import DexploreTask
        from learning import common_player
        from rl_games.common import player as base_player
        install_legacy_player_compat(native,common_player,base_player)
        install_reset_patch()
        base = native.EvalPlayer
        class FullStartPlayer(base):
            def run(self):
                self.is_deterministic = self.is_determenistic
                task = self.env.task
                if abs(task.dt-1/30)>1e-8 or task.num_envs != 64 or self.is_rnn:
                    raise ValueError('fixed30Hz/64env/nonrecurrent screen required')
                task._state_init = DexploreTask.StateInit.Start
                task._hybrid_init_prob = 1.
                task._adaptive_kappa_enabled = False
                task._enable_early_termination = False
                super().run()
        native.EvalPlayer = FullStartPlayer
        manifest['runtime'] = dict(python=sys.executable,torch=torch.__version__,
                                   torch_path=torch.__file__,rl_games=importlib.metadata.version('rl_games'),
                                   rl_games_path=rl_games.__file__,
                                   numpy=importlib.metadata.version('numpy'))
        argv = ['--task','Dexplore_Inspire','--cfg_env',config['cfg_env'],
                '--cfg_train',str(native_root/'data/cfg/train/rlg/inspire.yaml'),
                '--motion_file',str(motion_root),'--checkpoint',str(checkpoint),
                '--disable-early-termination','--headless','--sim_device','cuda:0',
                '--rl_device','cuda:0','--pipeline','gpu','--graphics_device_id','0',
                '--num_envs','64','--seed',str(a.seed),'--output',str(output/'native-results.json'),
                '--output_path',str(output/'native-runtime'),'--transition-output',str(output/'transitions.pt')]
        manifest.update(status='RUNNING',command=argv,
                        player_compatibility='legacy_flat_config_single_external_observation_rms_deterministic_alias',
                        reset_contract='batched_actor_roots_urdf_fk_no_extra_physics_step')
        write(output/'run_manifest.json',manifest)
        os.chdir(ROOT/'third_party/DExplore')
        sys.argv=[str(Path(__file__)),*argv]
        native.main()
        payload = torch.load(output/'transitions.pt',map_location='cpu',weights_only=False)
        episodes = json.loads((output/'native-results.json').read_text())['per_episode']
        result = qualify_transitions(payload,episodes)
        result.update(actor=a.actor,checkpoint_sha256=checkpoint_sha,
                      limitation='single-seed generator readiness screen using native force proxies; not evaluator labels, formal validation or policy utility')
        if any(sha(path)!=value for path,value in frozen.items()):
            raise RuntimeError('screen source/input drift')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) > 2**30:
            raise RuntimeError('screen output exceeded1GiB')
        result.update(transitions_sha256=sha(output/'transitions.pt'),
                      native_results_sha256=sha(output/'native-results.json'))
        write(output/'qualification.json',result)
        manifest.update(status='COMPLETED',qualified_episodes=result['qualified_episodes'],
                        data_readiness_pass=result['data_readiness_pass'])
        print(json.dumps({k:manifest[k] for k in ('actor','status','qualified_episodes')}),flush=True)
    except BaseException as error:
        manifest.update(status='TIMED_OUT' if isinstance(error,Deadline) else 'FAILED',error=repr(error))
        raise
    finally:
        signal.alarm(0)
        os.chdir(old_cwd)
        sys.argv=old_argv
        manifest['elapsed_s']=time.monotonic()-started
        write(output/'run_manifest.json',manifest)


if __name__ == '__main__':
    main()
