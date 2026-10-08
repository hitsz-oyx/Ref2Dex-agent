"""Bounded six-expert collection: phase-targeted smooth residuals, no forks.

Asset and GPU checks run before importing Isaac Gym/Torch. Real integration
requires the fixed six self-trained checkpoints and native motion/configs.
"""
import argparse
import importlib.util
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
from consequence_evaluator.collection import Episode, Perturbations, PhaseAssignments, PHASES
from consequence_evaluator.contracts import (is_within, EPISODE_SCHEMA, ACTION_SEMANTICS,
                                             HAND_LINKS, MIN_PREFERENCE_PAIRS)
from consequence_evaluator.provenance import self_trained_ancestry
from consequence_evaluator.value_outcomes import RAW_SCHEMA as VALUE_EPISODE_SCHEMA, validate_plan_execution


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


def native_motion_files(root):
    """Freeze the immediate native sequence directories, including owned links."""
    root = Path(root)
    files = []
    for folder in sorted(root.iterdir()):
        if folder.is_symlink() and not folder.exists():
            raise FileNotFoundError('broken native motion link: ' + str(folder))
        if not folder.is_dir():
            continue
        tensor = folder/'interaction_hand_inspire.pt'
        if not tensor.is_file():
            raise FileNotFoundError('missing native sequence tensor: ' + str(tensor))
        files.append(tensor.absolute())
    if not files or len({path.resolve() for path in files}) != len(files):
        raise ValueError('native motion directory needs distinct sequence tensors')
    return files


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
    p.add_argument('--audit-only', action='store_true',
                   help='collect observational diagnostics; output can never be used for fitting')
    p.add_argument('--value-outcomes', action='store_true',
                   help='independent ref4 airplane outcome dataset; does not relax the old production schema')
    p.add_argument('--official-generator',action='store_true',
                   help='explicit pinned official source, only for the independent value dataset')
    p.add_argument('--clean-only',action='store_true',help='value-data nominal failure audit before fitting perturbation distributions')
    p.add_argument('--target-phase', choices=('all', *PHASES), default='all',
                   help='audit-only clean/target-stage allocation; all retains the production schedule')
    a = p.parse_args()
    if a.value_outcomes and a.audit_only:
        p.error('value outcomes and engineering-only collection are distinct modes')
    if (a.official_generator or a.clean_only) and not a.value_outcomes:
        p.error('official source and nominal pilot are exclusive to the independent value contract')
    if a.official_generator and a.observation_router_model:
        p.error('official generator requires a fixed controller')
    if a.target_phase != 'all' and not (a.audit_only or a.value_outcomes):
        p.error('focused stage allocation requires --audit-only')
    if (not 1 <= a.waves <= 4 or not 6 <= a.num_envs <= 64 or not 1 <= a.seconds <= 900
            or not 24 <= a.max_steps <= 1200 or not 0 < a.amplitude <= .2):
        p.error('bounded collection: <=4waves, <=64envs, <=900s, <=1200steps')
    output = a.output.resolve()
    if not is_within(output, ROOT/'outputs/consequence-evaluator') or output.exists():
        p.error('fresh task-owned output directory required')
    config = json.loads(a.route_config.read_text())
    baseline='official_inspire' if a.official_generator else 'airplane_base'
    official_entry=None
    if a.official_generator:
        path=TASK/'tools/audit/probe_official_generator.py'
        spec=importlib.util.spec_from_file_location('official_generator_entry',path)
        official_entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(official_entry)
        if (set(config['experts'])!={'official_inspire'} or config.get('default_expert')!=baseline
                or Path(sys.prefix).resolve()!=official_entry.RUNTIME.resolve()):
            raise ValueError('explicit single official route and complete archived runtime required')
        source=config['experts'][baseline]
        if (Path(source['checkpoint']).resolve()!=official_entry.OFFICIAL.resolve()
                or source['sha256']!=official_entry.OFFICIAL_SHA):
            raise ValueError('official source is not the user-authorized hash-pinned teacher')
        screen=json.loads(Path(config['generator_screen']).read_text())
        if (screen['checkpoint_sha256']!=source['sha256'] or screen['actor']!='official'
                or sum(r['maximum_held_frames']>=45 for r in screen['per_episode'])<32):
            raise ValueError('official source lacks the declared physical grasp coverage screen')
    elif len(config['experts']) != 6 or len({s['sha256'] for s in config['experts'].values()})!=6:
        raise ValueError('fixed six-expert route required')
    if a.value_outcomes:
        if (config.get('object_route',{}).get('airplane') != baseline
                or (not a.official_generator and config['experts'][baseline].get('data_readiness_pass') is not True)):
            raise ValueError('ref4 requires a qualified fixed airplane baseline')
    if not (a.audit_only or a.value_outcomes) and (config.get('training_allowed') is not True
            or config.get('all_experts_operationally_qualified') is not True):
        raise ValueError('expert route is observational-only; all six experts must be operationally qualified before collection')
    frozen = {str(a.route_config.resolve()): digest(a.route_config),
              str(Path(__file__).resolve()): digest(__file__)}
    for path, expected in config.get('sources', {}).items():
        if (not is_within(Path(path).resolve(), ROOT/'outputs/consequence-evaluator')
                or digest(path) != expected):
            raise ValueError('generated route evidence identity mismatch: '+str(path))
        frozen[path] = expected
    for spec in config['experts'].values():
        path = (a.asset_root/spec['checkpoint']).resolve()
        if not path.is_file():
            raise FileNotFoundError('missing self-trained expert: ' + str(path))
        if digest(path) != spec['sha256']:
            raise ValueError('expert checkpoint identity mismatch: ' + str(path))
        spec['checkpoint'] = str(path)
        frozen[str(path)] = spec['sha256']
        if a.official_generator:
            continue
        source_run=Path(spec['training_run']).resolve()
        ancestral=self_trained_ancestry(source_run,ROOT/'outputs/consequence-evaluator')
        trained=json.loads((source_run/'run_manifest.json').read_text())
        if Path(trained['checkpoint']).resolve()!=path:
            raise ValueError('expert route does not match its owned training endpoint')
        frozen.update(ancestral)
    if not a.motions.is_dir():
        raise FileNotFoundError(a.motions)
    motion_files = native_motion_files(a.motions)
    if a.value_outcomes and len(motion_files) != 1:
        raise ValueError('first ref4 outcome Probe requires one fixed airplane motion')
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
             ROOT/'third_party/DExplore/dexplore/learning/common_player.py',
             ROOT/'src/task/CmResidual/dexplore_cm_geometry.py',
             ROOT/'src/task/CmResidual/v118_planner.py',
             ROOT/'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
             ROOT/'third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py',
             *sorted(p for p in (ROOT/'third_party/DExplore/dexplore/data/assets').rglob('*') if p.is_file())]
    if a.official_generator:
        files += [TASK/'tools/audit/probe_official_generator.py',Path(config['generator_screen']),
                  *sorted((official_entry.PACKAGES/'rl_games').rglob('*.py'))]
    for path in files:
        # Keep runtime aliases: retargeting a sequence/assets link must also
        # change the checked input, even when its previous target still exists.
        frozen[str(path.absolute())] = digest(path)
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
                      OMP_NUM_THREADS='2', TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/consequence-official-generator/torch201-extensions' if a.official_generator else scratch/'torch-extensions'))
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(ROOT/'third_party/DExplore/dexplore'), str(ROOT),
                   str(ROOT/'src/task/cm-interaction-oracle/src')]
    if a.official_generator:
        # Optional router/geometry dependencies missing in the preserved venv
        # are fallback-only; its compatible Torch/NumPy/rl_games stay first.
        sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    # Importing the native router imports Isaac Gym before Torch.
    import evaluate_object_router as router
    import torch
    if a.official_generator:
        from learning import common_player
        from rl_games.common import player as base_player
        official_entry.install_legacy_player_compat(router.original,common_player,base_player)
        if torch.__version__!='2.0.1+cu118':
            raise ValueError('official runtime Torch version changed')
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
    manifest = dict(schema=VALUE_EPISODE_SCHEMA if a.value_outcomes else EPISODE_SCHEMA, status='RUNNING',
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                    work_version='consequence-value-ref4' if a.value_outcomes else 'consequence-evaluator-ref2', run_id=output.name, pid=os.getpid(),
                    task='consequence-evaluator', rollout_kind='continuous', training_allowed=not a.audit_only,
                    audit_only=a.audit_only, target_phase=a.target_phase,
                    route_sha256=digest(a.route_config),
                    all_experts_operationally_qualified=config.get('all_experts_operationally_qualified') is True,
                    fps=30, units='m', horizon=24, execution_horizon=24 if a.value_outcomes else 8, seed=a.seed, split=a.split,
                    physical_gpu=a.gpu, sources=frozen, seconds_budget=a.seconds, waves=a.waves,
                    max_steps=a.max_steps,action_semantics=ACTION_SEMANTICS,
                    hand_keypoint_links=list(HAND_LINKS),
                    minimum_pair_coverage=MIN_PREFERENCE_PAIRS,
                    pair_coverage_required=not a.value_outcomes, value_outcomes=a.value_outcomes,
                    num_envs=a.num_envs, amplitude=a.amplitude, episodes=[],
                    route_mode='observation' if router.MODEL_PATH else 'fixed_object_route',
                    progress_labels='unknown and masked pending reliable expert annotation',
                    phase_definitions='approach;contact;3-contact-step-grasp;3cm-lift;5-held-step-hold',
                    perturbation_channels='native independent controls; coupled distal residuals zero',
                    diagnostic_only_fields=['contact','contact_valid','actual_residual','clipped','q','hand_root'])
    manifest['contact_semantics'] = 'native_hand_and_object_net_force_proxy'
    if a.value_outcomes:
        manifest.update(value_task='airplane_grasp_lift_controlled_place', baseline_expert=baseline,
            source_actor_role='official_data_collector' if a.official_generator else 'self_trained_generator',
            task_success_rule='full-reference-grasp-lift-controlled-place-stage-margin-v1',
            user_task_definition='retain complete reference; controlled normal return is successful',
            clean_only=a.clean_only,required_expert_qualified=not a.official_generator,
            fixed_continuation='same frozen expert after one immutable24step requested residual plan',
            labels_pending=True, outcome_semantics='whole-task outcome after decision-known plan and fixed-policy suffix through first episode end')
        manifest['diagnostic_only_fields'] += ['reference_object_pose','object_velocity','support_gap','table_footprint']
    write(output/'manifest.json', manifest)
    last_resource_check = [0.]
    def check():
        if time.monotonic()-started >= a.seconds:
            raise TimeoutError('fixed collection deadline')
        if sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) > 2*2**30:
            raise RuntimeError('collection output exceeded2GiB')
        if a.value_outcomes and time.monotonic()-last_resource_check[0] >= 15:
            last_resource_check[0]=time.monotonic()
            if shutil.disk_usage(ROOT).free < 20*2**30:
                raise RuntimeError('20GiB reserve violated')
            pids=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
                                           '--format=csv,noheader'],text=True).strip().splitlines()
            if any(int(pid)!=os.getpid() for pid in pids if pid.strip()):
                raise RuntimeError('foreign GPU compute process appeared')
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
            if a.value_outcomes:
                from consequence_evaluator.value_geometry import TableSupport
                support_geometry=TableSupport(ROOT/'third_party/DExplore/dexplore/data/assets',task.device)
            def physical():
                hand = (task._contact_forces[:,task._contact_body_ids].norm(dim=-1)>.1).any(-1)
                obj = task._tar_contact_forces.norm(dim=-1)>.1
                return task._target_states.detach().cpu().numpy().copy(), (hand&obj).cpu().numpy()
            def kinematics():
                points,gap=geometry.measure(task)
                measured=dict(q=task._dof_pos.detach().cpu().numpy().copy(),
                            hand_keypoints=points.cpu().numpy(),surface_gap=gap.cpu().numpy(),
                            hand_root=task._humanoid_root_states.detach().cpu().numpy().copy())
                if a.value_outcomes:
                    support,footprint=support_geometry.measure(task,geometry)
                    index=task.progress_buf.clamp_max(task.hoi_data.shape[1]-1)
                    reference=task.hoi_data[task.data_id,index]
                    from consequence_evaluator.collection import pose_matrix
                    ref_state=np.zeros((task.num_envs,13),np.float32)
                    ref_state[:,:3]=reference[:,106:109].cpu().numpy()
                    ref_state[:,3:7]=reference[:,109:113].cpu().numpy()
                    measured.update(reference_object_pose=pose_matrix(ref_state),
                                    object_velocity=task._target_states[:,7:13].cpu().numpy().copy(),
                                    support_gap=support.cpu().numpy(),table_footprint=footprint.cpu().numpy())
                return measured
            selected_phases = PHASES if a.target_phase == 'all' else (a.target_phase,)
            assignments = PhaseAssignments(a.seed, phases=selected_phases)
            for wave in range(a.waves):
                check()
                if self.observation_router is not None:
                    self.initial_expert_names = None
                obs = self.env_reset(full)
                if self.get_batch_size(obs['obs'],1)!=task.num_envs:
                    raise ValueError('native player observation batch differs from env count')
                if (task.start_times != 0).any():
                    raise ValueError('full reference-start episodes required')
                states, contact = physical()
                history = obs['obs'].detach().cpu().numpy().copy()
                manifest['history_contract'] = dict(source='raw native policy observation; per-expert RMS remains inside player',
                    shape=list(history.shape[1:]), player_sha256=frozen[str((ROOT/'third_party/DExplore/dexplore/evaluate.py').resolve())])
                measured = kinematics()
                episodes = [Episode(history[i], states[i], contact[i], a.max_steps,
                                    {name:values[i] for name,values in measured.items()}) for i in range(a.num_envs)]
                active = np.ones(a.num_envs, dtype=bool)
                initial_height = states[:,2].copy()
                motion = task.data_id.cpu().numpy().copy()
                perturb = Perturbations(a.num_envs, a.seed+wave, a.amplitude, wave,
                                        assignment=np.zeros(a.num_envs,np.int64) if a.clean_only else assignments.assign(motion))
                object_names = [task.object_name[int(task.object_id[int(index)])] for index in motion]
                if a.value_outcomes and set(object_names) != {'airplane'}:
                    raise ValueError('ref4 first Probe supports airplane only')
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
                    # Native tensor-pipeline env_step returns a raw observation,
                    # unlike env_reset. Wrap without resetting or normalizing it.
                    if not isinstance(obs,dict):
                        obs={'obs':obs}
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
                    packet=episode.arrays()
                    if a.value_outcomes:
                        if self.expert_names[int(self.last_teacher_choice[env])] != baseline:
                            raise ValueError('fixed continuation expert changed')
                        validate_plan_execution(packet,dict(perturbation_tick=int(perturb.started[env])))
                    identity = f's{a.seed}_w{wave}_e{env}_{object_names[env]}'
                    path = output/(identity+'.npz')
                    np.savez_compressed(path, **packet)
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
                coverage = {}
                for record in manifest['episodes']:
                    key = record['expert']+'|'+record['motion']
                    group = coverage.setdefault(key, {'assigned': {}, 'triggered': {}})
                    phase = record['assigned_phase']
                    group['assigned'][phase] = group['assigned'].get(phase,0)+1
                    if record['perturbation_tick'] >= 0:
                        group['triggered'][phase] = group['triggered'].get(phase,0)+1
                manifest['assignment_semantics'] = 'seeded per-current-motion clean/target-phase rotation across waves'
                manifest['assignment_phases'] = ['clean', *selected_phases]
                manifest['phase_coverage'] = coverage
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
