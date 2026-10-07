"""Bounded six-expert collection: phase-targeted smooth residuals, no forks.

Asset and GPU checks run before importing Isaac Gym/Torch. Real integration
requires the fixed six self-trained checkpoints and native motion/configs.
"""
import argparse
import hashlib
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))

import numpy as np
from consequence_evaluator.collection import Episode, Perturbations, PHASES
from consequence_evaluator.contracts import is_within, EPISODE_SCHEMA, ACTION_SEMANTICS, HAND_LINKS
from consequence_evaluator.provenance import self_trained_ancestry


class CollectionDeadline(BaseException):
    """Escape third-party Exception handlers when the fixed budget expires."""


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, data):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route-config', type=Path, required=True)
    p.add_argument('--asset-root', type=Path, default=ROOT)
    p.add_argument('--motions', type=Path, required=True)
    p.add_argument('--cfg-env', type=Path, required=True)
    p.add_argument('--cfg-train', type=Path, required=True)
    p.add_argument('--observation-router-model', type=Path)
    p.add_argument('--observation-router-sha256')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    p.add_argument('--waves', type=int, default=2)
    p.add_argument('--num-envs', type=int, default=24)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--max-steps', type=int, default=1200)
    p.add_argument('--amplitude', type=float, default=.08)
    a = p.parse_args()
    if (not 1 <= a.waves <= 4 or not 6 <= a.num_envs <= 64 or not 1 <= a.seconds <= 900
            or not 24 <= a.max_steps <= 1200 or not 0 < a.amplitude <= .2):
        p.error('bounded collection: <=4waves, <=64envs, <=900s, <=1200steps')
    output = a.output.resolve()
    if not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists():
        p.error('fresh task-owned output directory required')
    config = json.loads(a.route_config.read_text())
    if len(config['experts']) != 6 or len({s['sha256'] for s in config['experts'].values()})!=6:
        raise ValueError('fixed six-expert route required')
    frozen = {str(a.route_config.resolve()): digest(a.route_config),
              str(Path(__file__).resolve()): digest(__file__)}
    for spec in config['experts'].values():
        path = (a.asset_root/spec['checkpoint']).resolve()
        if not path.is_file():
            raise FileNotFoundError('missing self-trained expert: ' + str(path))
        if digest(path) != spec['sha256']:
            raise ValueError('expert checkpoint identity mismatch: ' + str(path))
        spec['checkpoint'] = str(path)
        frozen[str(path)] = spec['sha256']
        source_run=Path(spec['training_run']).resolve()
        ancestral=self_trained_ancestry(source_run,ROOT/'outputs/consequence-evaluator')
        trained=json.loads((source_run/'run_manifest.json').read_text())
        if Path(trained['checkpoint']).resolve()!=path:
            raise ValueError('expert route does not match its owned training endpoint')
        frozen.update(ancestral)
    if not a.motions.is_dir():
        raise FileNotFoundError(a.motions)
    motion_files = sorted(a.motions.rglob('*.pt')) + sorted(a.motions.rglob('*.npy'))
    if not motion_files:
        raise ValueError('native motion directory has no .pt/.npy inputs')
    files = [a.cfg_env, a.cfg_train, *motion_files,
             *sorted((TASK/'src/consequence_evaluator').glob('*.py')),
             ROOT/'third_party/DExplore/dexplore/evaluate.py',
             ROOT/'third_party/DExplore/dexplore/evaluate_object_router.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/base_task.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/vec_task.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/vec_task_wrappers.py',
             ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py',
             ROOT/'third_party/DExplore/dexplore/learning/dexplore_players.py',
             *sorted(p for p in (ROOT/'third_party/DExplore/dexplore/data/assets').rglob('*') if p.is_file())]
    for path in files:
        frozen[str(path.resolve())] = digest(path)
    if bool(a.observation_router_model) != bool(a.observation_router_sha256):
        p.error('observation router path/hash must be supplied together')
    if a.observation_router_model:
        if digest(a.observation_router_model) != a.observation_router_sha256:
            raise ValueError('observation router identity changed')
        frozen[str(a.observation_router_model.resolve())] = a.observation_router_sha256
    occupied = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied before model/Isaac initialization: ' + occupied.replace('\n', ', '))
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('free disk below20GiB collection reserve')
    scratch = ROOT/'tmp/consequence-evaluator-collect'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='2', TORCH_EXTENSIONS_DIR=str(scratch/'torch-extensions'))
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT/'third_party/DExplore/dexplore'), str(ROOT),
                   str(ROOT/'src/task/cm-interaction-oracle/src')]
    # Importing the native router imports Isaac Gym before Torch.
    import evaluate_object_router as router
    import torch
    from consequence_evaluator.physical_geometry import PhysicalGeometry
    from env.tasks.base_dexplore_task import DexploreTask
    from consequence_evaluator.native_reset import install_reset_patch
    install_reset_patch()
    from oracle_y_utility import align_native_reference_tables
    original_reset = DexploreTask._reset_ref_state_init
    def aligned_reset(task, ids):
        align_native_reference_tables(task)
        return original_reset(task, ids)
    DexploreTask._reset_ref_state_init = aligned_reset
    router.CONFIG = config
    router.MODEL_PATH = a.observation_router_model.resolve() if a.observation_router_model else None
    output.mkdir(parents=True)
    (output/'diagnostics').mkdir()
    started = time.monotonic()
    manifest = dict(schema=EPISODE_SCHEMA, status='RUNNING',
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                    work_version='consequence-evaluator-ref2', run_id=output.name, pid=os.getpid(),
                    task='consequence-evaluator', rollout_kind='continuous', training_allowed=True,
                    fps=30, units='m', horizon=24, execution_horizon=8, seed=a.seed, split=a.split,
                    physical_gpu=a.gpu, sources=frozen, seconds_budget=a.seconds, waves=a.waves,
                    max_steps=a.max_steps,action_semantics=ACTION_SEMANTICS,
                    hand_keypoint_links=list(HAND_LINKS),
                    num_envs=a.num_envs, amplitude=a.amplitude, episodes=[],
                    route_mode='observation' if router.MODEL_PATH else 'fixed_object_route',
                    progress_labels='unknown and masked pending reliable expert annotation',
                    phase_definitions='approach;contact;3-contact-step-grasp;3cm-lift;5-held-step-hold',
                    perturbation_channels='native independent controls; coupled distal residuals zero',
                    diagnostic_only_fields=['contact','contact_valid','actual_residual','clipped','q','hand_root'])
    manifest['contact_semantics'] = 'native_hand_and_object_net_force_proxy'
    write(output/'manifest.json', manifest)
    def check():
        if time.monotonic()-started >= a.seconds:
            raise TimeoutError('fixed collection deadline')
        if sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) > 2*2**30:
            raise RuntimeError('collection output exceeded2GiB')
    class Collector(router.RoutedPlayer):
        @torch.no_grad()
        def run(self):
            task = self.env.task
            if abs(task.dt-1/30) > 1e-8 or task.num_envs != a.num_envs or self.is_rnn:
                raise ValueError('native clock/env count/recurrent policy contract unsupported')
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            task._state_init = DexploreTask.StateInit.Start
            # In this native implementation probability1 selects frame0.
            task._hybrid_init_prob = 1.
            native_pre_physics = task.pre_physics_step
            executed_control = None
            def capture_executed(actions):
                nonlocal executed_control
                # VecTask clipping and domain action noise have already run.
                # Capture before Inspire converts fingers/PD targets in-place.
                executed_control = actions.detach().cpu().numpy().copy()
                return native_pre_physics(actions)
            task.pre_physics_step = capture_executed
            full = torch.arange(task.num_envs, device=task.device)
            geometry=PhysicalGeometry(task,ROOT/'third_party/DExplore/dexplore/data/assets')
            def physical():
                hand = (task._contact_forces[:,task._contact_body_ids].norm(dim=-1)>.1).any(-1)
                obj = task._tar_contact_forces.norm(dim=-1)>.1
                return task._target_states.detach().cpu().numpy().copy(), (hand&obj).cpu().numpy()
            def kinematics():
                points,gap=geometry.measure(task)
                return dict(q=task._dof_pos.detach().cpu().numpy().copy(),
                            hand_keypoints=points.cpu().numpy(),surface_gap=gap.cpu().numpy(),
                            hand_root=task._humanoid_root_states.detach().cpu().numpy().copy())
            for wave in range(a.waves):
                check()
                if self.observation_router is not None:
                    self.initial_expert_names = None
                obs = self.env_reset(full)
                if (task.start_times != 0).any():
                    raise ValueError('full reference-start episodes required')
                states, contact = physical()
                history = obs['obs'].detach().cpu().numpy().copy()
                manifest['history_contract'] = dict(source='raw native policy observation; per-expert RMS remains inside player',
                    shape=list(history.shape[1:]), player_sha256=frozen[str((ROOT/'third_party/DExplore/dexplore/evaluate.py').resolve())])
                measured = kinematics()
                episodes = [Episode(history[i], states[i], contact[i], a.max_steps,
                                    {name:values[i] for name,values in measured.items()}) for i in range(a.num_envs)]
                perturb = Perturbations(a.num_envs, a.seed+wave, a.amplitude, wave)
                active = np.ones(a.num_envs, dtype=bool)
                initial_height = states[:,2].copy()
                motion = task.data_id.cpu().numpy().copy()
                object_names = [task.object_name[int(task.object_id[int(index)])] for index in motion]
                for tick in range(a.max_steps):
                    check()
                    base = self.get_action(obs, True).clamp(-1,1)
                    base_control = base.detach().cpu().numpy().copy()
                    remaining = torch.minimum(task.max_episode_length[task.data_id]-1-task.progress_buf,
                                              task.rollout_length-1-(task.progress_buf-task.start_times)).cpu().numpy()
                    control, phases, diagnostic = perturb.apply(base_control, states[:,2], contact,
                                                                 initial_height, tick, remaining, active)
                    planned,plan_known=perturb.known_plan(tick)
                    control[~active] = 0
                    # Native Inspire mutates its action tensor in PD conversion.
                    # Keep the logged normalized control separate from that buffer.
                    action = torch.tensor(control, device=self.device)
                    executed_control = None
                    obs, _, done, info = self.env_step(self.env, action)
                    if executed_control is None or executed_control.shape != control.shape:
                        raise ValueError('native pre-physics command capture did not run')
                    states, contact = physical()
                    history = obs['obs'].detach().cpu().numpy().copy()
                    measured = kinematics()
                    ended = done.detach().cpu().numpy().reshape(-1).astype(bool)
                    self._post_step(info)
                    for env in np.flatnonzero(active):
                        episodes[env].append(executed_control[env], phases[env], history[env], states[env], contact[env],
                                             executed_control[env]-base_control[env],
                                             diagnostic['clipped'][env] | (np.abs(executed_control[env]-control[env])>1e-7), ended[env],
                                             {name:values[env] for name,values in measured.items()},
                                             plan=planned[env],plan_known=plan_known[env])
                    active &= ~ended
                    if not active.any():
                        break
                if active.any():
                    raise RuntimeError('incomplete native episodes at fixed step cap; do not pad/reset')
                for env, episode in enumerate(episodes):
                    identity = f's{a.seed}_w{wave}_e{env}_{object_names[env]}'
                    path = output/(identity+'.npz')
                    np.savez_compressed(path, **episode.arrays())
                    sidecar = output/'diagnostics'/(identity+'.npz')
                    np.savez_compressed(sidecar, **episode.diagnostics())
                    manifest['episodes'].append(dict(episode=identity, split_group=f'source_seed:{a.seed}',
                        split=a.split, task=object_names[env], quality='unlabeled', path=path.name,
                        sha256=digest(path), diagnostics=str(sidecar.relative_to(output)),
                        diagnostics_sha256=digest(sidecar), motion_id=int(motion[env]),
                        motion=Path(task.motion_file[int(motion[env])]).name,
                        expert= self.expert_names[int(self.last_teacher_choice[env])],
                        assigned_phase='clean' if perturb.assignment[env]==0 else PHASES[perturb.assignment[env]-1],
                        perturbation_tick=int(perturb.started[env]), steps=len(episode.actions)))
                write(output/'manifest.json', manifest)
    native = ['--task','Dexplore_Inspire','--cfg_env',str(a.cfg_env.resolve()),
              '--cfg_train',str(a.cfg_train.resolve()),'--motion_file',str(a.motions.resolve()),
              '--checkpoint',config['experts'][config['default_expert']]['checkpoint'],
              '--headless','--sim_device','cuda:0','--rl_device','cuda:0','--pipeline','gpu',
              '--graphics_device_id','0','--num_threads','1','--num_envs',str(a.num_envs),
              '--seed',str(a.seed),'--output',str(output/'native-player-unused.json'),
              '--output_path',str(output/'native-runtime'),'--disable-early-termination']
    original_argv = sys.argv
    sys.argv = [sys.argv[0], *native]
    router.original.EvalPlayer = Collector
    original_cwd = Path.cwd()
    previous_signal = signal.getsignal(signal.SIGALRM)
    def deadline(signum, frame):
        raise CollectionDeadline('fixed native collection deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(a.seconds)
    try:
        os.chdir(ROOT/'third_party/DExplore')
        router.original.main()
        if any(digest(path)!=expected for path,expected in frozen.items()):
            raise RuntimeError('collection source/input drift')
        manifest['status']='COMPLETED'
    except BaseException as error:
        manifest.update(status='TIMED_OUT' if isinstance(error,(TimeoutError,CollectionDeadline)) else 'FAILED', error=repr(error))
        raise
    finally:
        sys.argv=original_argv
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_signal)
        os.chdir(original_cwd)
        manifest['elapsed_s']=time.monotonic()-started
        write(output/'manifest.json', manifest)


if __name__ == '__main__':
    main()
