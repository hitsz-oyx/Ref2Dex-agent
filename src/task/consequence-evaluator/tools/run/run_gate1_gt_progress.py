"""Fresh complete-prefix native replay, causal GT phase value, real mixed controls.

The fixed first Probe intervenes only at four early-contact decisions, then
continues the self-trained actor to the full reference end. No mid-state restore.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]; ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.contracts import HAND_LINKS, K, K_EXEC, is_within
from consequence_evaluator.gate1 import (SEEDS, QUERY_TICKS, CANDIDATES, candidate_plan,
    choose_candidate, episode_outcome, paired_counts, legacy_batched_actor_action,
    legacy_group_actor_action)
from consequence_evaluator.provenance import self_trained_ancestry, sha
from consequence_evaluator.twin import (REQUIRED_NATIVE_STATE_KEYS, capture_native_rng,
    fingerprint, restore_native_rng)
from consequence_evaluator.native_backend import BACKENDS, canonical_device, resolve_legacy_backend

NATIVE_PYTHON = '/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


def load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def save(path, value):
    with Path(path).open('xb') as stream:
        pickle.dump(value, stream, protocol=4)


def ensure_free(gpu):
    pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip()
    if pids:
        raise RuntimeError('GPU is occupied: ' + pids)


def native_worker(a):
    ensure_free(a.gpu)
    if a.worker_output is None:
        raise ValueError('native worker requires --worker-output')
    a.worker_output.parent.mkdir(parents=True, exist_ok=True)
    backend = a.native_backend
    helper = TASK / 'tools/audit/probe_official_generator.py'
    spec = importlib.util.spec_from_file_location('official_runtime_compat', helper)
    compat = importlib.util.module_from_spec(spec); spec.loader.exec_module(compat)
    if Path(sys.prefix).resolve() != compat.RUNTIME:
        raise RuntimeError('native worker requires the verified archived runtime')
    scratch = ROOT / 'tmp/consequence-official-generator'; scratch.mkdir(exist_ok=True, parents=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(ROOT / 'tmp'),
        TORCH_EXTENSIONS_DIR=str(scratch / 'torch201-extensions'), OMP_NUM_THREADS='2')
    native_root = ROOT / 'third_party/DExplore/dexplore'
    sys.path[:0] = [str(compat.ISAAC), str(native_root), str(ROOT / 'src/task/CmResidual/tools'),
                   str(ROOT / 'src/task/cm-interaction-oracle/src')]
    sys.path.append('/home2/wyy/.local/lib/python3.8/site-packages')
    # Bootstrap installs old NumPy aliases; evaluate imports Gym before Torch.
    import dexplore_ddp_rank_bootstrap
    import evaluate as native
    import torch
    from learning import common_player
    from rl_games.common import player as base_player
    from env.tasks.base_dexplore_task import DexploreTask
    from consequence_evaluator.native_reset import install_reset_patch
    from consequence_evaluator.physical_geometry import PhysicalGeometry
    from consequence_evaluator.value_geometry import TableSupport
    from consequence_evaluator.collection import pose_matrix
    from oracle_y_utility import align_native_reference_tables
    if torch.__version__ != '2.0.1+cu118':
        raise RuntimeError('archived Torch was shadowed')
    compat.install_legacy_player_compat(native, common_player, base_player)
    install_reset_patch(); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    original_reset = DexploreTask._reset_ref_state_init
    def aligned_reset(task, ids):
        align_native_reference_tables(task); return original_reset(task, ids)
    DexploreTask._reset_ref_state_init = aligned_reset
    properties_spec = importlib.util.spec_from_file_location('native_property_codec', TASK / 'tools/audit/native_twin_probe.py')
    properties_module = importlib.util.module_from_spec(properties_spec); properties_spec.loader.exec_module(properties_module)
    normalize_properties = properties_module.normalize_properties
    config = json.loads((a.run_dir / 'config.json').read_text())
    trained = json.loads((a.run_dir / 'run_manifest.json').read_text())
    cfg_env = Path(config['cfg_env']).resolve()
    if a.engineering_env_spacing is not None:
        text = cfg_env.read_text()
        replacement, count = re.subn(
            r'(?m)^(\s*envSpacing:\s*)[-+0-9.eE]+(\s*(?:#.*)?$)',
            r'\g<1>%s\g<2>' % format(a.engineering_env_spacing, '.9g'), text)
        if count != 1:
            raise ValueError('engineering env-spacing override expected one envSpacing entry, got %d' % count)
        spacing_root = ROOT / 'tmp/consequence-official-generator'
        spacing_root.mkdir(exist_ok=True, parents=True)
        cfg_env = spacing_root / ('inspire-engineering-spacing-%s.yaml' %
                                  format(a.engineering_env_spacing, '.9g').replace('.', '_'))
        cfg_env.write_text(replacement)
    expected = load(a.prefix) if a.prefix else None
    replay_chunk_packet = load(a.action_chunk_replay) if a.action_chunk_replay else None
    query = a.query_tick
    if expected is not None and query > len(expected['actions']):
        raise ValueError('missing committed prefix controls')
    base = native.EvalPlayer
    started = time.monotonic()

    def engineering_group_run(player, task):
        """Run baseline/zero/+/- in one synchronous native-pipeline process.

        Isaac Gym reports separate environment origins, but DExplore's native
        tensors and task geometry are already expressed in each environment's
        local frame. The packet retains the reported origins as provenance and
        compares those local tensors directly. The first two environments are
        both baseline controls; their prefix is measured with strict hashes and
        retained drift diagnostics through the complete bounded window.
        """
        count = int(a.engineering_group_envs)
        actor_copies = int(a.engineering_actor_copies)
        if actor_copies < 1:
            raise ValueError('engineering group actor copies must be positive')
        if count * actor_copies != 256:
            raise ValueError('engineering group must keep the fixed 256-row actor contract')
        if player.is_rnn or task.num_envs != count or abs(task.dt - 1 / 30) > 1e-8 \
                or abs(task.sim_params.dt - 1 / 60) > 1e-8 or task.control_freq_inv != 2:
            raise ValueError('fixed nonrecurrent synchronous group contract required')
        if int(task.num_motions) != 1:
            raise ValueError('engineering group requires one repeated motion across all env roles')
        zero_left, zero_right = map(int, a.zero_env_pair)
        positive_env, negative_env = map(int, a.candidate_env_pair)
        candidate_envs = (positive_env, negative_env)
        if (not 0 <= zero_left < count or not 0 <= zero_right < count
                or not 0 <= positive_env < count or not 0 <= negative_env < count
                or zero_left == zero_right or positive_env == negative_env
                or set((zero_left, zero_right)) & set(candidate_envs)):
            raise ValueError('zero and candidate env pairs must be distinct in-range roles')
        prefix_actions = None
        if a.action_chunk_roles == 'candidate' and expected is not None:
            prefix_actions = np.asarray(expected.get('actions'))
            if prefix_actions.ndim == 3:
                prefix_actions = prefix_actions[:, 0]
            if prefix_actions.ndim != 2 or prefix_actions.shape[1] != 18 or len(prefix_actions) < 48:
                raise ValueError('candidate group prefix must contain a native [T,18] action stream')
        task._state_init = DexploreTask.StateInit.Start; task._hybrid_init_prob = 1.
        task._adaptive_kappa_enabled = False; task._enable_early_termination = False
        if task.dr_randomizations or task._motion_sampler is not None or task.projtype != 'None':
            raise ValueError('randomized/sampler/projectile replay unsupported')
        geometry = PhysicalGeometry(task, native_root / 'data/assets')
        table = TableSupport(native_root / 'data/assets', task.device)
        proposal_model = None
        proposal_replay = None
        proposal_mean = proposal_scale = None
        proposal_period = None
        proposal_roles = a.action_chunk_roles if a.action_chunk_checkpoint is not None else None
        if a.action_chunk_replay is not None:
            proposal_roles = a.action_chunk_roles
        proposal_trace = []
        proposal_ticks = []
        proposal_inputs = []
        requested_controls = []
        active_chunk_counts = []
        proposal_executor = None
        # The historical group probe used tick48, but exposing the query
        # boundary lets an engineering probe test a prefix that ends before
        # the first native contact-cache divergence.  A zero CLI value keeps
        # the established tick48 contract.
        query = int(a.query_tick or 48)
        if a.action_chunk_checkpoint is not None:
            from consequence_evaluator.action_chunk import (EXECUTED_ACTION_SEMANTICS,
                NativeActionChunkProposal, ActionChunkExecutor)
            payload = torch.load(a.action_chunk_checkpoint, map_location='cpu')
            if (payload.get('schema') != 'ref2dex.consequence-evaluator.native-action-chunks.v1'
                    or payload.get('chunk') != K or payload.get('action_dim') != 18
                    or payload.get('history_dim') != int(task.num_obs)
                    or payload.get('executed_action_semantics') != EXECUTED_ACTION_SEMANTICS):
                raise ValueError('native action-chunk checkpoint contract mismatch')
            standardizer = payload.get('history_standardizer')
            if (not isinstance(standardizer, dict) or len(standardizer.get('mean', [])) != int(task.num_obs)
                    or len(standardizer.get('scale', [])) != int(task.num_obs)):
                raise ValueError('native action-chunk history standardizer mismatch')
            proposal_mean = torch.as_tensor(standardizer['mean'], dtype=torch.float32, device=player.device)
            proposal_scale = torch.as_tensor(standardizer['scale'], dtype=torch.float32, device=player.device)
            proposal_clip = standardizer.get('clip')
            if (proposal_mean.ndim != 1 or proposal_scale.ndim != 1
                    or proposal_mean.numel() != int(task.num_obs) or proposal_scale.numel() != int(task.num_obs)
                    or not torch.isfinite(proposal_mean).all() or not torch.isfinite(proposal_scale).all()
                    or (proposal_scale <= 0).any()
                    or (proposal_clip is not None and (not np.isfinite(proposal_clip) or float(proposal_clip) <= 0))):
                raise ValueError('native action-chunk history standardizer is invalid')
            proposal_model = NativeActionChunkProposal(
                int(payload['history_dim']), width=int(payload['width']), layers=int(payload['layers'])).to(player.device)
            proposal_model.load_state_dict(payload['state_dict'], strict=True); proposal_model.eval()
            proposal_executor = ActionChunkExecutor(a.action_chunk_mode)
            proposal_period = proposal_executor.period
        if replay_chunk_packet is not None:
            role_names = list(replay_chunk_packet.get('role_names') or [])
            if (replay_chunk_packet.get('engineering_only') is not True
                    or replay_chunk_packet.get('action_chunk_mode') != 'open_loop24'
                    or replay_chunk_packet.get('group_mode') != 'synchronous_same_process_act_behavior'
                    or replay_chunk_packet.get('source_prefix_role') != 'reactive_teacher'
                    or 'act_chunk' not in role_names):
                raise ValueError('native action-chunk replay packet contract mismatch')
            chunks = np.asarray(replay_chunk_packet.get('proposal_chunks'))
            replay_index = int(replay_chunk_packet.get('proposal_index', query // 24))
            role_index = role_names.index('act_chunk')
            if (chunks.ndim != 4 or chunks.shape[1] <= role_index or chunks.shape[2:] != (24, 18)
                    or replay_index >= chunks.shape[0]
                    or not np.isfinite(chunks).all() or np.abs(chunks).max() > 1 + 1e-6):
                raise ValueError('native action-chunk replay proposal shape/range mismatch')
            proposal_replay = torch.as_tensor(chunks[replay_index, role_index], dtype=torch.float32,
                                               device=player.device).unsqueeze(0)
            proposal_period = 24
        env_ids = torch.arange(count, device=task.device)
        obs = player.env_reset(env_ids)
        player.get_batch_size(obs['obs'], count)
        if task.gym.get_frame_count(task.sim) != 0 or (task.start_times != 0).any():
            raise ValueError('fresh unstepped group reset required')
        for name in ('data_id', 'progress_buf', 'start_times', 'ref_index'):
            value = getattr(task, name).detach().cpu().numpy()
            if not np.array_equal(value, np.broadcast_to(value[0], value.shape)):
                raise ValueError('group semantic initial state differs for %s' % name)
        total = int(task.max_episode_length[task.data_id[0]]) - 1
        if total != 542:
            raise ValueError('frozen full543state reference required')
        origins = []
        for env in task.envs:
            origin = task.gym.get_env_origin(env)
            origins.append((origin.x, origin.y, origin.z))
        origins = torch.as_tensor(origins, dtype=torch.float32, device=task.device)
        initial_points, initial_gap = geometry.measure(task)
        initial_support, initial_footprint = table.measure(task, geometry)
        initial_history = obs['obs'].detach()
        initial_semantic_gap = dict(
            object_state_max_abs=float((task._target_states - task._target_states[0]).abs().max().item()),
            dof_position_max_abs=float((task._dof_pos - task._dof_pos[0]).abs().max().item()),
            dof_velocity_max_abs=float((task._dof_vel - task._dof_vel[0]).abs().max().item()),
            hand_keypoints_max_abs=float((initial_points - initial_points[0]).abs().max().item()),
            surface_gap_max_abs=float((initial_gap - initial_gap[0]).abs().max().item()),
            support_gap_max_abs=float((initial_support - initial_support[0]).abs().max().item()),
            history_max_abs=float((initial_history - initial_history[0]).abs().max().item()),
            contact_force_max_abs=float((task._contact_forces - task._contact_forces[0]).abs().max().item()),
            object_contact_force_max_abs=float((task._tar_contact_forces - task._tar_contact_forces[0]).abs().max().item()),
            footprint_mismatch_count=int((initial_footprint != initial_footprint[0]).sum().item()))
        initial_semantic_exact = bool(
            max(value for key, value in initial_semantic_gap.items() if key.endswith('_max_abs')) <= 1e-7
            and initial_semantic_gap['footprint_mismatch_count'] == 0)
        if not initial_semantic_exact:
            raise ValueError('group semantic initial state differs: %r' % initial_semantic_gap)

        physics = {'sim_params': normalize_properties(task.gym.get_sim_params(task.sim)), 'actors': []}
        for env in task.envs:
            actors = []
            for actor_index in range(task.gym.get_actor_count(env)):
                handle = task.gym.get_actor_handle(env, actor_index)
                actors.append(dict(name=task.gym.get_actor_name(env, actor_index),
                    shape=normalize_properties(task.gym.get_actor_rigid_shape_properties(env, handle)),
                    body=normalize_properties(task.gym.get_actor_rigid_body_properties(env, handle)),
                    dof=normalize_properties(task.gym.get_actor_dof_properties(env, handle))))
            physics['actors'].append(actors)
        controller = dict(model={k: v.cpu().numpy().copy() for k, v in player.model.state_dict().items()},
            normalize_input=bool(player.normalize_input), checkpoint_sha256=trained['checkpoint_sha256'])
        for key, normalizer in [('rms', getattr(player, 'running_mean_std', None)),
                                ('amp_rms', getattr(player, '_amp_input_mean_std', None))]:
            controller[key] = None if normalizer is None else {k: v.cpu().numpy().copy() for k, v in normalizer.state_dict().items()}
        params = task.gym.get_sim_params(task.sim)
        actual_backend = dict(name=backend.name, sim_device=backend.sim_device,
            pipeline='gpu' if bool(params.use_gpu_pipeline) else 'cpu',
            physx_use_gpu=bool(params.physx.use_gpu), physx_num_threads=int(params.physx.num_threads),
            tensor_device=canonical_device(task.device), actor_device=canonical_device(player.device))
        expected_backend = backend.as_dict()
        for key in ('pipeline', 'physx_use_gpu', 'physx_num_threads', 'tensor_device', 'actor_device'):
            if actual_backend[key] != expected_backend[key]:
                raise ValueError('native backend contract drift for %s: expected %r, got %r' %
                                 (key, expected_backend[key], actual_backend[key]))
        identity = dict(physics_hash=fingerprint(physics), controller_hash=fingerprint(controller), backend=actual_backend)
        identity['environment'] = dict(cfg_env_sha256=sha(cfg_env),
                                       env_spacing_override=a.engineering_env_spacing)
        identity['query_tick'] = query
        identity['actor_execution'] = dict(
            layout='environment_rows_then_fixed_copies',
            copies=actor_copies,
            total_rows=count * actor_copies,
        )
        retarget = None
        if a.retargeter_checkpoint is not None:
            from consequence_evaluator.retarget_execution import GTRetargetExecution
            retarget = GTRetargetExecution(a.retargeter_checkpoint, a.retargeter_source,
                                         task, player, identity)
        elif a.object_relative_source is not None:
            from consequence_evaluator.object_relative_servo import ObjectRelativeGTExecution
            retarget = ObjectRelativeGTExecution(a.object_relative_source, task, player,
                                                identity, a.object_relative_anchor, a.object_relative_layout,
                                                preload_path=a.object_relative_preload, inverse_path=a.object_relative_inverse,
                                                pd_inverse_path=a.object_relative_pd_inverse)
        if replay_chunk_packet is not None:
            replay_identity = replay_chunk_packet.get('replay_identity')
            if (not isinstance(replay_identity, dict)
                    or replay_identity.get('controller_hash') != identity['controller_hash']
                    or replay_identity.get('physics_hash') != identity['physics_hash']
                    or replay_identity.get('backend') != identity['backend']
                    or replay_identity.get('environment') != identity['environment']
                    or replay_identity.get('query_tick') != identity['query_tick']
                    or replay_identity.get('actor_execution') != identity['actor_execution']
                    or replay_chunk_packet.get('seed') != a.seed):
                raise ValueError('native action-chunk replay provenance does not match current actor/backend')
            if (a.prefix is None or replay_chunk_packet.get('source_prefix_packet_sha256') != sha(a.prefix)):
                raise ValueError('native action-chunk replay prefix does not match the supplied control prefix')
        if prefix_actions is not None:
            prefix_identity = expected.get('replay_identity') if isinstance(expected, dict) else None
            if (not isinstance(expected, dict) or expected.get('engineering_only') is not True
                    or expected.get('seed') != a.seed or not isinstance(prefix_identity, dict)
                    or prefix_identity.get('controller_hash') != identity['controller_hash']
                    or prefix_identity.get('physics_hash') != identity['physics_hash']
                    or prefix_identity.get('backend') != identity['backend']
                    or prefix_identity.get('environment') != identity['environment']
                    or prefix_identity.get('query_tick') != identity['query_tick']
                    or prefix_identity.get('actor_execution') != identity['actor_execution']):
                raise ValueError('candidate prefix provenance does not match current native actor/backend')
            if (not np.isfinite(prefix_actions).all() or np.abs(prefix_actions).max() > 1 + 1e-6):
                raise ValueError('candidate prefix contains invalid native controls')
        state_keys = sorted(REQUIRED_NATIVE_STATE_KEYS | {k for k in ('reset_buf', 'actions', 'real_pd_tar')
                                                           if isinstance(getattr(task, k, None), torch.Tensor)})
        fields = ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap', 'table_footprint',
                  'dof_position', 'dof_velocity', 'object_velocity', 'history',
                  'native_contact_forces', 'native_object_contact_forces')
        if a.object_relative_source is not None:
            fields += ('native_rigid_body_states',)
        rows = {k: [] for k in fields}; actions=[]; ended=[]; hashes=[]; canonical_hashes=[]
        state_field_hashes=[]; canonical_state_field_hashes=[]; rng_hashes=[]

        origin_np = origins.detach().cpu().numpy()
        def env_value(key, value, index):
            if key == '_root_states' and value.ndim >= 2 and value.shape[0] != count:
                return value.reshape(count, -1, value.shape[-1])[index].copy()
            if key == '_dof_state' and value.ndim >= 2 and value.shape[0] != count:
                return value.reshape(count, -1, value.shape[-1])[index].copy()
            if key == '_rigid_body_state' and value.ndim >= 2 and value.shape[0] != count:
                return value.reshape(count, -1, value.shape[-1])[index].copy()
            return value[index].copy()

        def canonical_state(state, index):
            # DExplore's native state views are already environment-local (the
            # flattened tensors are only reshaped here). Keep Gym origins as
            # provenance rather than subtracting them a second time.
            result = {}
            for key, value in state.items():
                item = env_value(key, value, index)
                result[key] = item
            return result

        def observe():
            points, gap = geometry.measure(task); support, footprint = table.measure(task, geometry)
            object_pose = pose_matrix(task._target_states.cpu().numpy())
            points_np = points.detach().cpu().numpy().copy()
            values = dict(object_pose=object_pose, hand_keypoints=points_np,
                surface_gap=gap.cpu().numpy().copy(), support_gap=support.cpu().numpy().copy(),
                table_footprint=footprint.cpu().numpy().copy(),
                dof_position=task._dof_pos.cpu().numpy().copy(),
                dof_velocity=task._dof_vel.cpu().numpy().copy(),
                object_velocity=task._target_states[:, 7:13].cpu().numpy().copy(),
                history=obs['obs'].cpu().numpy().copy(),
                native_contact_forces=task._contact_forces.cpu().numpy().copy(),
                native_object_contact_forces=task._tar_contact_forces.cpu().numpy().copy())
            if a.object_relative_source is not None:
                values['native_rigid_body_states'] = task._rigid_body_state.view(count, -1, 13).cpu().numpy().copy()
            for key, value in values.items(): rows[key].append(np.asarray(value).copy())
            state = {key: getattr(task, key).cpu().numpy().copy() for key in state_keys}
            per_state=[]; per_canonical=[]; per_fields=[]; per_canonical_fields=[]
            for index in range(count):
                raw = {key: env_value(key, value, index) for key, value in state.items()}
                local = canonical_state(state, index)
                per_state.append(fingerprint(raw)); per_canonical.append(fingerprint(local))
                per_fields.append({key: fingerprint(value) for key, value in raw.items()})
                per_canonical_fields.append({key: fingerprint(value) for key, value in local.items()})
            hashes.append(per_state); canonical_hashes.append(per_canonical)
            state_field_hashes.append(per_fields); canonical_state_field_hashes.append(per_canonical_fields)
            rng_hashes.append(fingerprint(capture_native_rng(torch)))
            index = len(hashes) - 1
            if task.gym.get_frame_count(task.sim) != index * task.control_freq_inv:
                raise ValueError('unexpected simulator step count at group tick%d' % index)

        executed = None; original_pre = task.pre_physics_step
        def capture(control):
            nonlocal executed
            executed = control.detach().cpu().numpy().copy()
            return original_pre(control)
        task.pre_physics_step = capture
        observe()
        horizon = 24
        positive = torch.as_tensor(candidate_plan(1), device=player.device)
        negative = torch.as_tensor(candidate_plan(2), device=player.device)
        stop = a.engineering_steps or query + horizon
        if stop > total:
            raise ValueError('group probe exceeds frozen reference')
        proposal_enabled = proposal_model is not None or proposal_replay is not None
        for tick in range(stop):
            if time.monotonic() - started > 150:
                raise TimeoutError('bounded synchronous group deadline')
            if retarget is not None:
                teacher_controls = legacy_group_actor_action(player, obs, copies=actor_copies).clamp(-1, 1)
                points, _ = geometry.measure(task)
                control = retarget.control(tick, task, points, teacher_controls)
            elif proposal_enabled and proposal_roles == 'candidate' and tick < query:
                # Preserve the native policy's approach/contact prefix.  Only
                # the 24-step scoring window is frozen into a nominal chunk.
                if prefix_actions is not None:
                    control = torch.as_tensor(prefix_actions[tick], dtype=torch.float32,
                                              device=player.device).view(1, -1).expand(count, -1).clone()
                else:
                    actor_actions = legacy_group_actor_action(player, obs, copies=actor_copies).clamp(-1, 1)
                    control = actor_actions[zero_left:zero_left + 1].expand(count, -1).clone()
            elif proposal_enabled:
                # Generate a complete chunk only at the declared replanning
                # boundary.  Every intervening control is read from that
                # frozen chunk; no future observation is fed back mid-chunk.
                proposal_boundary = tick == query if proposal_roles == 'candidate' else tick % proposal_period == 0
                if proposal_boundary:
                    # Candidate mode deliberately uses one proposal generated
                    # from env0 and broadcasts that frozen nominal chunk to
                    # all roles.  This keeps the zero pair on one action
                    # stream and makes the only post-query intervention the
                    # requested residual candidate.
                    if proposal_replay is not None:
                        proposal_cache = proposal_replay
                    else:
                        proposal_input = (obs['obs'][:1] - proposal_mean) / proposal_scale \
                            if proposal_roles == 'candidate' else (obs['obs'] - proposal_mean) / proposal_scale
                        if proposal_clip is not None:
                            proposal_input = proposal_input.clamp(-float(proposal_clip), float(proposal_clip))
                        proposal_cache = proposal_model(proposal_input).clamp(-1, 1)
                    proposal_trace.append(proposal_cache.detach().cpu().numpy().copy())
                    proposal_ticks.append(tick)
                    if proposal_model is not None:
                        proposal_inputs.append(obs['obs'].detach().cpu().numpy().copy())
                    if proposal_roles == 'behavior':
                        proposal_executor.add(tick, proposal_cache)
                if 'proposal_cache' not in locals():
                    raise RuntimeError('action-chunk cache was not initialized')
                chunk_offset = tick - query if proposal_roles == 'candidate' else tick % proposal_period
                if proposal_roles == 'candidate':
                    control = proposal_cache[0, chunk_offset].expand(count, -1).clone()
                else:
                    proposal_control = proposal_executor.action(tick)
                    actor_actions = legacy_group_actor_action(player, obs, copies=actor_copies).clamp(-1, 1)
                    base_action = actor_actions[:1].expand(count, -1).clone()
                    control = base_action.clone()
                    control[1] = proposal_control[1]
                    # Keep the repeated ACT arm on the same frozen chunk. The
                    # earlier behavior probe used an independently inferred
                    # env3 chunk, which mixed policy sensitivity into its
                    # solver-drift diagnostic.
                    control[3] = proposal_control[1]
                    active_chunk_counts.append(proposal_executor.active_count)
            else:
                # Keep baseline, zero-repeat, and both candidate arms on the same
                # control stream through the query. This removes tiny row-wise
                # GEMM differences and makes the group test an execution-contract probe.
                actor_actions = legacy_group_actor_action(player, obs, copies=actor_copies).clamp(-1, 1)
                base_action = actor_actions[zero_left:zero_left + 1].expand(count, -1).clone()
                control = base_action.clone()
            if retarget is None and (not proposal_enabled or proposal_roles == 'candidate') and query <= tick < query + horizon:
                offset = tick - query
                control[positive_env] = (control[positive_env] + positive[offset]).clamp(-1, 1)
                control[negative_env] = (control[negative_env] + negative[offset]).clamp(-1, 1)
            requested_controls.append(control.detach().cpu().numpy().copy())
            obs, _, done, info = player.env_step(player.env, control.clone())
            if not isinstance(obs, dict): obs = {'obs': obs}
            player._post_step(info)
            if executed is None or not np.isfinite(executed).all() or np.abs(executed).max() > 1 + 1e-6:
                raise ValueError('nonfinite/out-of-bounds group control')
            actions.append(executed.copy()); ended.append(done.detach().cpu().numpy().reshape(-1).astype(bool))
            observe()
            if done.any() and tick < stop - 1:
                raise ValueError('group episode terminated before requested window')

        packets = {key: np.asarray(value) for key, value in rows.items()}
        packets.update(actions=np.asarray(actions), done=np.asarray(ended), state_hashes=hashes,
            canonical_state_hashes=canonical_hashes, state_field_hashes=state_field_hashes,
            canonical_state_field_hashes=canonical_state_field_hashes, rng_hashes=rng_hashes,
            timestamps=np.arange(stop + 1) / 30)
        if retarget is not None:
            packet = retarget.finish(packets, identity, requested_controls, dict(
                seed=a.seed, source_backend=actual_backend, initial_semantic_gap=initial_semantic_gap,
                initial_semantic_exact=initial_semantic_exact, env_origins=origin_np.tolist(),
                actor_inference_batch=count * actor_copies, actor_inference_copies=actor_copies))
            save(a.worker_output, packet)
            write(a.worker_output.with_suffix('.json'), dict(status='COMPLETED', steps=stop,
                elapsed_s=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                metrics=packet['metrics'], role_outcomes=packet['role_outcomes'], gate=packet['gate'],
                clipped_coordinate_counts_total=np.asarray(packet['clipped_coordinate_counts']).sum(0).tolist(),
                commanded_target_max_abs_error=np.asarray(packet['commanded_target_max_abs_error']).max(0).tolist()))
            return
        if proposal_enabled and proposal_roles == 'behavior':
            # This is a behavior-only packet: role pairs are reactive teacher /
            # repeated teacher and open-loop ACT / repeated ACT.  It deliberately
            # bypasses the zero/candidate calibration fields below.
            role_names = ['reactive_teacher', 'act_chunk', 'reactive_repeat', 'act_chunk_repeat']
            pair_drift = {}
            for left, right, name in ((0, 2, 'reactive_pair'), (1, 3, 'act_pair')):
                pair_drift[name] = {}
                for field in fields:
                    value = np.asarray(packets[field], dtype=np.float64)
                    delta = np.abs(value[:, left] - value[:, right]).reshape(len(value), -1)
                    pair_drift[name][field] = dict(
                        max_abs=float(delta.max()) if delta.size else 0.0,
                        p95_abs=float(np.quantile(delta, .95)) if delta.size else 0.0,
                        prequery_max_abs=float(delta[:query + 1].max()) if delta[:query + 1].size else 0.0)
            role_outcomes = {}
            for index, name in enumerate(role_names):
                role_packet = {key: value[:, index] for key, value in packets.items() if key in fields}
                role_packet['timestamps'] = packets['timestamps']
                role_outcomes[name] = episode_outcome(role_packet) if stop >= 45 else None
            packet = dict(**packets, proposal_chunks=np.asarray(proposal_trace), seed=a.seed,
                          proposal_query_ticks=np.asarray(proposal_ticks),
                          proposal_input_history=np.asarray(proposal_inputs),
                          requested_controls=np.asarray(requested_controls),
                          active_chunk_counts=np.asarray(active_chunk_counts),
                          role_names=role_names, group_mode='synchronous_same_process_act_behavior',
                          action_chunk_checkpoint=str(a.action_chunk_checkpoint),
                          action_chunk_checkpoint_sha256=sha(a.action_chunk_checkpoint),
                          action_chunk_mode=a.action_chunk_mode,
                          action_chunk_replan_period=proposal_period,
                          action_chunk_semantics=(
                              'causal overlapping native18 chunks; exponential weights oldest-first'
                              if a.action_chunk_mode in ('overlap8', 'temporal1') else
                              'one-shot native18 action chunk; no mid-chunk observation feedback'),
                          temporal_aggregation_decay=proposal_executor.decay,
                          temporal_aggregation_order='oldest_covering_prediction_first',
                          training_allowed=False,
                          replay_identity=identity, canonical_state_keys=state_keys,
                          actor_inference_batch=count * actor_copies,
                          actor_inference_copies=actor_copies, engineering_only=True,
                          group_envs=count, initial_semantic_gap=initial_semantic_gap,
                          initial_semantic_exact=initial_semantic_exact, env_origins=origin_np.tolist(),
                          source_backend=actual_backend, pair_drift=pair_drift,
                          role_outcomes=role_outcomes,
                          role_max_lift_m={name: float(np.max(packets['object_pose'][:, i, 2, 3]
                                                             - packets['object_pose'][0, i, 2, 3]))
                                          for i, name in enumerate(role_names)})
            save(a.worker_output, packet)
            write(a.worker_output.with_suffix('.json'), dict(
                status='COMPLETED', steps=stop, group_envs=count, engineering_only=True,
                action_chunk_mode=a.action_chunk_mode, role_names=role_names,
                elapsed_s=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                initial_semantic_gap=initial_semantic_gap, initial_semantic_exact=initial_semantic_exact,
                pair_drift=pair_drift, role_outcomes=role_outcomes,
                role_max_lift_m=packet['role_max_lift_m']))
            return
        # env0 baseline and env1 zero-repeat must share the same local state,
        # contact/history and control at every tick; candidates are allowed to diverge.
        exact_keys = ['canonical_state_hashes', 'canonical_state_field_hashes', 'actions', 'done',
                      'object_pose', 'hand_keypoints', 'surface_gap', 'support_gap', 'table_footprint',
                      'dof_position', 'dof_velocity', 'object_velocity', 'history',
                      'native_contact_forces', 'native_object_contact_forces']
        mismatches=[]
        for key in exact_keys:
            value=packets[key]
            for tick in range(len(value)):
                if not np.array_equal(value[tick][zero_left], value[tick][zero_right]):
                    mismatches.append(dict(field=key, tick=tick)); break
        if mismatches:
            diagnostics=[]
            for mismatch in mismatches[:12]:
                key, tick = mismatch['field'], mismatch['tick']
                value = packets[key]
                left, right = value[tick][zero_left], value[tick][zero_right]
                detail = dict(field=key, tick=tick)
                if isinstance(left, dict) and isinstance(right, dict):
                    detail['different_keys'] = [name for name in left if left.get(name) != right.get(name)]
                else:
                    try:
                        left_array = np.asarray(left); right_array = np.asarray(right)
                        detail['shapes'] = [list(left_array.shape), list(right_array.shape)]
                        if left_array.dtype.kind in 'biufc' and right_array.dtype.kind in 'biufc':
                            detail['max_abs'] = float(np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64))))
                            detail['different_indices'] = np.argwhere(left_array != right_array)[:10].tolist()
                    except (TypeError, ValueError):
                        detail['left_repr'] = repr(left)[:200]; detail['right_repr'] = repr(right)[:200]
                diagnostics.append(detail)
            save(a.worker_output.with_suffix('.mismatch.pkl'), packets)
            write(a.worker_output.with_suffix('.mismatch.json'), dict(mismatches=mismatches, diagnostics=diagnostics,
                origins=origin_np.tolist()))
            # Parallel environments are a solver-schedule probe, not a claim
            # that two different world instances are bitwise twins. Preserve
            # the drift diagnostics and continue to measure whether the
            # native-policy behavior is restored. Fresh-process exact replay
            # is still required by the production Gate1 worker.
        baseline_packet={key: value[:,zero_left] for key,value in packets.items() if key in fields}
        baseline_packet['timestamps']=packets['timestamps']
        outcome=episode_outcome(baseline_packet) if stop >= 45 else None
        calibration_fields = ('object_pose', 'hand_keypoints', 'dof_position', 'dof_velocity',
                              'object_velocity', 'history', 'native_contact_forces',
                              'native_object_contact_forces')

        def delta_summary(delta, tick_offset=0):
            """Summarize pair drift without hiding a rare contact impulse."""
            array = np.asarray(delta, dtype=np.float64)
            if array.size == 0:
                return dict(max_abs=0., p95_abs=0., p99_abs=0.,
                            peak_tick=None, first_nonzero_tick=None,
                            tick_max_abs=[])
            flat = np.abs(array).reshape(array.shape[0], -1)
            tick_max = flat.max(axis=1)
            nonzero = np.flatnonzero(tick_max > 1e-12)
            return dict(
                max_abs=float(tick_max.max()),
                p95_abs=float(np.quantile(flat.reshape(-1), .95)),
                p99_abs=float(np.quantile(flat.reshape(-1), .99)),
                peak_tick=int(tick_offset + np.argmax(tick_max)),
                first_nonzero_tick=(int(tick_offset + nonzero[0]) if nonzero.size else None),
                tick_max_abs=tick_max.tolist())

        zero_noise = {}; candidate_effect = {}
        # State packets have one more sample than the executed-control packet:
        # include the state at t+24 when a 72-step bounded run ends there.
        effect_stop = min(len(packets['object_pose']), query + horizon + 1)
        for key in calibration_fields:
            value = np.asarray(packets[key], dtype=np.float64)
            zero = value[:query + 1, zero_right] - value[:query + 1, zero_left]
            zero_summary = delta_summary(zero)
            zero_flat = np.abs(zero).reshape(-1)
            zero_max = zero_summary['max_abs']
            zero_p95 = zero_summary['p95_abs']
            zero_post = value[query + 1:effect_stop, zero_right] - value[query + 1:effect_stop, zero_left]
            zero_displacement = ((value[query + 1:effect_stop, zero_right] - value[query, zero_right])
                                 - (value[query + 1:effect_stop, zero_left] - value[query, zero_left]))
            zero_post_summary = delta_summary(zero_post, query + 1)
            zero_disp_summary = delta_summary(zero_displacement, query + 1)
            zero_post_flat = np.abs(zero_post).reshape(-1)
            zero_disp_flat = np.abs(zero_displacement).reshape(-1)
            zero_noise[key] = dict(
                prequery_max_abs=zero_max, prequery_p95_abs=zero_p95,
                prequery_p99_abs=zero_summary['p99_abs'],
                prequery_peak_tick=zero_summary['peak_tick'],
                prequery_first_nonzero_tick=zero_summary['first_nonzero_tick'],
                prequery_tick_max_abs=zero_summary['tick_max_abs'],
                postquery_max_abs=zero_post_summary['max_abs'],
                postquery_p95_abs=zero_post_summary['p95_abs'],
                postquery_p99_abs=zero_post_summary['p99_abs'],
                postquery_peak_tick=zero_post_summary['peak_tick'],
                postquery_displacement_max_abs=zero_disp_summary['max_abs'],
                postquery_displacement_p95_abs=zero_disp_summary['p95_abs'],
                postquery_displacement_p99_abs=zero_disp_summary['p99_abs'],
                postquery_displacement_peak_tick=zero_disp_summary['peak_tick'],
                postquery_displacement_tick_max_abs=zero_disp_summary['tick_max_abs'])
            candidate_effect[key] = {}
            for env_index, name in ((positive_env, 'positive'), (negative_env, 'negative')):
                query_offset = value[query, env_index] - value[query, zero_left]
                raw = value[query + 1:effect_stop, env_index] - value[query + 1:effect_stop, zero_left]
                incremental = ((value[query + 1:effect_stop, env_index] - value[query, env_index])
                               - (value[query + 1:effect_stop, zero_left] - value[query, zero_left]))
                effect_vs_zero = incremental - zero_displacement
                raw_summary = delta_summary(raw, query + 1)
                inc_summary = delta_summary(incremental, query + 1)
                effect_summary = delta_summary(effect_vs_zero, query + 1)
                raw_flat = np.abs(raw).reshape(-1); inc_flat = np.abs(incremental).reshape(-1)
                effect_flat = np.abs(effect_vs_zero).reshape(-1)
                inc_max = inc_summary['max_abs']
                candidate_effect[key][name] = dict(
                    raw_max_abs=raw_summary['max_abs'],
                    raw_p99_abs=raw_summary['p99_abs'], raw_peak_tick=raw_summary['peak_tick'],
                    incremental_max_abs=inc_max,
                    incremental_p95_abs=inc_summary['p95_abs'], incremental_p99_abs=inc_summary['p99_abs'],
                    incremental_peak_tick=inc_summary['peak_tick'],
                    effect_vs_zero_max_abs=effect_summary['max_abs'],
                    effect_vs_zero_p95_abs=effect_summary['p95_abs'],
                    effect_vs_zero_p99_abs=effect_summary['p99_abs'],
                    effect_vs_zero_peak_tick=effect_summary['peak_tick'],
                    effect_vs_zero_to_prequery_max_ratio=(float(effect_flat.max()) / zero_max if zero_max > 0 else None),
                    effect_vs_zero_to_prequery_p95_ratio=(float(np.quantile(effect_flat, .95)) / zero_p95 if zero_p95 > 0 else None),
                    effect_vs_zero_to_postquery_displacement_max_ratio=(float(effect_flat.max()) / zero_noise[key]['postquery_displacement_max_abs']
                        if zero_noise[key]['postquery_displacement_max_abs'] > 0 else None),
                    effect_vs_zero_to_postquery_displacement_p95_ratio=(float(np.quantile(effect_flat, .95)) / zero_noise[key]['postquery_displacement_p95_abs']
                        if zero_noise[key]['postquery_displacement_p95_abs'] > 0 else None),
                    query_offset_max_abs=float(np.abs(query_offset).max()) if np.size(query_offset) else 0.0)
        action_calibration = {}
        for env_index, name in ((positive_env, 'positive'), (negative_env, 'negative')):
            delta = (packets['actions'][query:effect_stop, env_index].astype(np.float64)
                     - packets['actions'][query:effect_stop, zero_left])
            delta_abs = np.abs(delta).reshape(-1)
            step_delta = np.abs(delta).max(axis=-1)
            action_calibration[name] = dict(max_abs=float(delta_abs.max()), p95_abs=float(np.quantile(delta_abs, .95)),
                                            nonzero_steps=int(np.count_nonzero(step_delta > 1e-7)),
                                            nonzero_elements=int(np.count_nonzero(delta_abs > 1e-7)))
        zero_pair_thresholds = dict(
            object_pose=1e-2, hand_keypoints=1e-2, dof_position=5e-2,
            dof_velocity=5., object_velocity=5., history=5.,
            native_contact_forces=5., native_object_contact_forces=5.)
        zero_pair_p95_thresholds = dict(
            object_pose=1e-3, hand_keypoints=2e-3, dof_position=2e-3,
            dof_velocity=5e-2, object_velocity=5e-2, history=5e-2,
            native_contact_forces=5., native_object_contact_forces=5.)
        zero_pair_gate = dict(
            max_thresholds=zero_pair_thresholds,
            p95_thresholds=zero_pair_p95_thresholds,
            prequery_max_abs={key: zero_noise[key]['prequery_max_abs'] for key in calibration_fields},
            prequery_p95_abs={key: zero_noise[key]['prequery_p95_abs'] for key in calibration_fields},
            max_passed=all(zero_noise[key]['prequery_max_abs'] <= limit
                           for key, limit in zero_pair_thresholds.items()),
            p95_passed=all(zero_noise[key]['prequery_p95_abs'] <= limit
                           for key, limit in zero_pair_p95_thresholds.items()),
            # Candidate calibration uses a robust p95 floor. The max values
            # remain visible because a contact impulse can be a real diagnostic
            # even when it should not invalidate the whole paired window.
            passed=all(zero_noise[key]['prequery_p95_abs'] <= limit
                       for key, limit in zero_pair_p95_thresholds.items()),
            rule='prequery zero-pair p95 must remain below field-specific physical tolerance; max is diagnostic')
        zero_role_indices = [index for index in range(count) if index not in candidate_envs]
        pairwise_zero_noise = {}
        for key in calibration_fields:
            value = np.asarray(packets[key], dtype=np.float64)
            pairs = []
            for left_offset, left in enumerate(zero_role_indices):
                for right in zero_role_indices[left_offset + 1:]:
                    delta = np.abs(value[:query + 1, left] - value[:query + 1, right]).reshape(-1)
                    pairs.append(dict(left=left, right=right,
                                      max_abs=float(delta.max()) if delta.size else 0.,
                                      p95_abs=float(np.quantile(delta, .95)) if delta.size else 0.))
            pairwise_zero_noise[key] = dict(
                zero_roles=zero_role_indices, pair_count=len(pairs), pairs=pairs,
                best_p95_abs=min((item['p95_abs'] for item in pairs), default=0.),
                median_p95_abs=float(np.median([item['p95_abs'] for item in pairs])) if pairs else 0.,
                worst_p95_abs=max((item['p95_abs'] for item in pairs), default=0.))
        common_pairs = []
        for left_offset, left in enumerate(zero_role_indices):
            for right in zero_role_indices[left_offset + 1:]:
                p95_by_field = {}
                for key in calibration_fields:
                    match = next(item for item in pairwise_zero_noise[key]['pairs']
                                 if item['left'] == left and item['right'] == right)
                    p95_by_field[key] = match['p95_abs']
                threshold_ratios = {
                    key: (p95_by_field[key] / zero_pair_p95_thresholds[key]
                          if zero_pair_p95_thresholds[key] > 0 else float('inf'))
                    for key in calibration_fields}
                common_pairs.append(dict(
                    left=left, right=right, p95_abs=p95_by_field,
                    max_p95_threshold_ratio=max(threshold_ratios.values()),
                    passed=all(threshold_ratios[key] <= 1. for key in calibration_fields)))
        best_common_pair = min(common_pairs, key=lambda item: item['max_p95_threshold_ratio'], default=None)
        common_pair_scores = [item['max_p95_threshold_ratio'] for item in common_pairs]
        passing_pair_count = sum(1 for item in common_pairs if item['passed'])
        pair_pass_fraction = (float(passing_pair_count) / len(common_pairs)
                              if common_pairs else 0.)
        pairwise_zero_gate = dict(
            zero_roles=zero_role_indices,
            p95_thresholds=zero_pair_p95_thresholds,
            best_fieldwise_p95={key: value['best_p95_abs'] for key, value in pairwise_zero_noise.items()},
            best_common_pair=best_common_pair,
            passing_pair_count=passing_pair_count, pair_count=len(common_pairs),
            pair_pass_fraction=pair_pass_fraction, required_pair_pass_fraction=.8,
            pair_score_quantiles=(dict(zip(('p50', 'p80', 'p90', 'p95', 'max'),
                                           [float(value) for value in np.quantile(
                                               common_pair_scores, [.5, .8, .9, .95, 1.])]))
                                  if common_pair_scores else {}),
            passed=bool(common_pairs and pair_pass_fraction >= .8),
            rule='at least 80% of common zero-zero pairs must meet every field p95 tolerance; fieldwise minima are diagnostic')
        candidate_margin = dict()
        for name in ('positive', 'negative'):
            candidate_margin[name] = {
                key: candidate_effect[key][name]['effect_vs_zero_to_postquery_displacement_p95_ratio']
                for key in calibration_fields}
        effect_margin_gate = dict(
            minimum_ratio=1., ratios=candidate_margin,
            passed=all(ratio is None or ratio >= 1.
                       for values in candidate_margin.values() for ratio in values.values()),
            rule='candidate-vs-zero p95 effect must reach the post-query zero displacement p95')
        behavior_gate = dict(
            evaluated=bool(stop >= total),
            passed=bool(stop >= total and outcome is not None
                        and outcome['maximum_held_frames'] >= 45
                        and outcome['controlled_final_place']),
            rule='candidate nominal must retain the full native GPU hold behavior before calibration')
        # This packet is engineering-only.  A failed calibration must never be
        # represented as valid merely because it used the reactive (rather than
        # ACT proposal) execution path; downstream audits use this field as a
        # promotion guard before any candidate value is considered.
        candidate_calibration_valid = bool(
            zero_pair_gate['passed'] and pairwise_zero_gate['passed']
            and effect_margin_gate['passed'] and behavior_gate['passed'])
        candidate_roles = ['baseline', 'zero_repeat', 'positive', 'negative']
        role_indices = dict(baseline=zero_left, zero_repeat=zero_right,
                            positive=positive_env, negative=negative_env)
        role_names = ['zero_role_%d' % index for index in range(count)]
        for name, index in role_indices.items():
            role_names[index] = name
        if proposal_enabled:
            nominal_text = ('recorded native nominal chunk' if a.action_chunk_replay is not None
                            else 'env0 ACT open-loop24 proposal')
            candidate_plan_semantics = (
                'baseline/zero/positive/negative; %s is broadcast to all envs, '
                'then positive/negative residuals fork at tick%d for24 steps' % (nominal_text, query))
        else:
            candidate_plan_semantics = (
                'baseline/zero/positive/negative; env0 actor control is broadcast to all envs, '
                'then positive/negative fork at tick%d for24 steps' % query)
        packet=dict(**packets, proposal_chunks=np.asarray(proposal_trace) if proposal_enabled else None,
            seed=a.seed, query_tick=query, candidate_roles=candidate_roles,
            candidate_env_pair=[positive_env, negative_env], role_indices=role_indices,
            role_index_map=role_indices, baseline_env_index=zero_left,
            zero_repeat_env_index=zero_right,
            candidate_env_indices=[positive_env, negative_env],
            role_names=role_names,
            candidate_plan_semantics=candidate_plan_semantics,
            group_mode=('synchronous_same_process_act_candidate_noise'
                        if proposal_enabled else 'synchronous_same_process_noise_calibration'),
            action_chunk_checkpoint=(str(a.action_chunk_checkpoint) if a.action_chunk_checkpoint is not None else None),
            action_chunk_checkpoint_sha256=(sha(a.action_chunk_checkpoint) if a.action_chunk_checkpoint is not None else None),
            action_chunk_replay_source=(str(a.action_chunk_replay) if a.action_chunk_replay is not None else None),
            action_chunk_replay_source_sha256=(sha(a.action_chunk_replay) if a.action_chunk_replay is not None else None),
            action_chunk_mode=(a.action_chunk_mode if proposal_enabled else None),
            action_chunk_roles=(a.action_chunk_roles if proposal_enabled else None),
            action_chunk_replan_period=(proposal_period if proposal_enabled else None),
            action_chunk_semantics=(
                'one-shot native18 action chunk; env0 proposal is frozen and broadcast; candidate residual is added only at query'
                if proposal_enabled else None),
            action_chunk_prefix_source=(str(a.prefix) if prefix_actions is not None else None),
            action_chunk_prefix_source_sha256=(sha(a.prefix) if prefix_actions is not None else None),
            replay_identity=identity, canonical_state_keys=state_keys,
            actor_inference_batch=(1 if proposal_enabled else count * actor_copies),
            actor_inference_copies=(None if proposal_enabled else actor_copies),
            engineering_only=True, group_envs=count, group_prefix_exact=(not mismatches),
            zero_env_pair=[zero_left, zero_right],
            control_prefix_exact=bool(np.array_equal(packets['actions'][:,zero_left], packets['actions'][:,zero_right])
                                      and np.array_equal(packets['done'][:,zero_left], packets['done'][:,zero_right])),
            exact_fields=exact_keys,
            rng_semantics='one process-global RNG trace shared by all envs; zero-pair calibration is state/observation based',
            calibration_window=dict(zero_prequery_ticks=[0, query], candidate_postquery_ticks=[query + 1, effect_stop - 1]),
            zero_noise_calibration=zero_noise, candidate_effect_calibration=candidate_effect,
            action_calibration=action_calibration,
            zero_pair_gate=zero_pair_gate, effect_margin_gate=effect_margin_gate,
            zero_role_indices=zero_role_indices, pairwise_zero_noise=pairwise_zero_noise,
            pairwise_zero_gate=pairwise_zero_gate,
            behavior_gate=behavior_gate, candidate_calibration_valid=candidate_calibration_valid,
            baseline_outcome=outcome, baseline_max_lift_m=float(np.max(
                packets['object_pose'][:,zero_left,2,3] - packets['object_pose'][0,zero_left,2,3])),
            zero_repeat_max_lift_m=float(np.max(packets['object_pose'][:,zero_right,2,3]
                                                - packets['object_pose'][0,zero_right,2,3])),
            initial_semantic_gap=initial_semantic_gap,
            initial_semantic_exact=initial_semantic_exact,
            env_origins=origin_np.tolist(),
            source_backend=actual_backend)
        save(a.worker_output, packet)
        write(a.worker_output.with_suffix('.json'), dict(status='COMPLETED', steps=stop, group_envs=count,
            baseline_zero_exact=not mismatches, mismatches=mismatches, engineering_only=True,
            group_mode=packet['group_mode'], action_chunk_roles=packet['action_chunk_roles'],
            actor_inference_batch=packet['actor_inference_batch'],
            actor_inference_copies=packet['actor_inference_copies'],
            elapsed_s=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(), outcome=outcome,
            baseline_max_lift_m=packet['baseline_max_lift_m'], zero_noise_calibration=zero_noise,
            candidate_effect_calibration=candidate_effect, action_calibration=action_calibration,
            zero_pair_gate=zero_pair_gate, effect_margin_gate=effect_margin_gate,
            zero_role_indices=zero_role_indices, pairwise_zero_noise=pairwise_zero_noise,
            pairwise_zero_gate=pairwise_zero_gate,
            behavior_gate=behavior_gate, candidate_calibration_valid=candidate_calibration_valid,
            initial_semantic_gap=initial_semantic_gap, initial_semantic_exact=initial_semantic_exact,
            zero_env_pair=[zero_left, zero_right],
            candidate_env_pair=[positive_env, negative_env], role_indices=role_indices,
            role_index_map=role_indices, baseline_env_index=zero_left,
            zero_repeat_env_index=zero_right,
            candidate_env_indices=[positive_env, negative_env],
            role_names=role_names,
            action_chunk_prefix_source=packet['action_chunk_prefix_source'],
            action_chunk_prefix_source_sha256=packet['action_chunk_prefix_source_sha256'],
            action_chunk_replay_source=packet['action_chunk_replay_source'],
            action_chunk_replay_source_sha256=packet['action_chunk_replay_source_sha256'],
            replay_identity=identity,
            role_max_lift_m={str(index): float(np.max(
                packets['object_pose'][:, index, 2, 3] - packets['object_pose'][0, index, 2, 3]))
                             for index in range(count)}))

    def engineering_serial_replay_run(player, task):
        """Replay frozen controls after one reactive teacher run.

        This is an engineering diagnostic for reset/cache repeatability.  It
        deliberately does not claim a fresh-simulator twin: PhysX solver
        caches are not serializable and the simulator frame counter remains
        process-local across resets.  The optional cluster schedule keeps the
        reactive teacher separate from frozen zero and candidate arms so its
        policy-driven trajectory is never used as a counterfactual zero.
        """
        if backend.name != 'gpu_physx_gpu_pipeline':
            raise ValueError('serial replay is defined only for native GPU PhysX/GPU pipeline')
        if (player.is_rnn or task.num_envs != 1 or abs(task.dt - 1 / 30) > 1e-8
                or abs(task.sim_params.dt - 1 / 60) > 1e-8 or task.control_freq_inv != 2):
            raise ValueError('serial replay requires one fixed nonrecurrent native environment')
        if int(task.num_motions) != 1:
            raise ValueError('serial replay requires one repeated motion')
        task._state_init = DexploreTask.StateInit.Start; task._hybrid_init_prob = 1.
        task._adaptive_kappa_enabled = False; task._enable_early_termination = False
        if task.dr_randomizations or task._motion_sampler is not None or task.projtype != 'None':
            raise ValueError('randomized/sampler/projectile serial replay unsupported')
        geometry = PhysicalGeometry(task, native_root / 'data/assets')
        table = TableSupport(native_root / 'data/assets', task.device)
        env_ids = torch.arange(1, device=task.device)
        horizon = 24; query = 48; stop = a.engineering_steps or query + horizon
        if stop < query + horizon or stop > 542:
            raise ValueError('serial replay requires query tick48 plus24 steps and <=542 steps')
        if a.engineering_serial_cluster and stop != 72:
            raise ValueError('serial cluster is fixed to the 72-step short-window contract')
        total = int(task.max_episode_length[task.data_id[0]]) - 1
        if total != 542:
            raise ValueError('frozen full543state reference required')
        # Keep the required native inventory and add task/controller buffers whose
        # reset semantics can affect the first post-reset step.  This remains an
        # engineering trace; it does not claim that every PhysX cache is visible.
        mutable_names = REQUIRED_NATIVE_STATE_KEYS | {
            'reset_buf', 'actions', 'real_pd_tar', '_dof_force_tensor', 'rew_buf',
            '_reset_ig', 'metric_1', 'metric_2', '_curr_ref_obs', 'ref_index',
            'curr_obj_points', '_kappa',
        }
        missing = sorted(name for name in REQUIRED_NATIVE_STATE_KEYS
                         if not isinstance(getattr(task, name, None), torch.Tensor))
        if missing:
            raise ValueError('serial replay missing required task tensors: %s' % missing)
        state_keys = sorted(name for name in mutable_names
                            if isinstance(getattr(task, name, None), torch.Tensor))
        fields = ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap', 'table_footprint',
                  'dof_position', 'dof_velocity', 'object_velocity', 'history',
                  'native_contact_forces', 'native_object_contact_forces')
        original_pre = task.pre_physics_step; executed = None; obs = None

        physics = {'sim_params': normalize_properties(task.gym.get_sim_params(task.sim)), 'actors': []}
        for env in task.envs:
            actors = []
            for actor_index in range(task.gym.get_actor_count(env)):
                handle = task.gym.get_actor_handle(env, actor_index)
                actors.append(dict(name=task.gym.get_actor_name(env, actor_index),
                    shape=normalize_properties(task.gym.get_actor_rigid_shape_properties(env, handle)),
                    body=normalize_properties(task.gym.get_actor_rigid_body_properties(env, handle)),
                    dof=normalize_properties(task.gym.get_actor_dof_properties(env, handle))))
            physics['actors'].append(actors)
        controller = dict(model={k: v.cpu().numpy().copy() for k, v in player.model.state_dict().items()},
            normalize_input=bool(player.normalize_input), checkpoint_sha256=trained['checkpoint_sha256'])
        for key, normalizer in [('rms', getattr(player, 'running_mean_std', None)),
                                ('amp_rms', getattr(player, '_amp_input_mean_std', None))]:
            controller[key] = None if normalizer is None else {
                k: v.cpu().numpy().copy() for k, v in normalizer.state_dict().items()}
        params = task.gym.get_sim_params(task.sim)
        actual_backend = dict(name=backend.name, sim_device=backend.sim_device,
            pipeline='gpu' if bool(params.use_gpu_pipeline) else 'cpu',
            physx_use_gpu=bool(params.physx.use_gpu),
            physx_num_threads=int(params.physx.num_threads),
            tensor_device=canonical_device(task.device), actor_device=canonical_device(player.device))
        expected_backend = backend.as_dict()
        for key in ('pipeline', 'physx_use_gpu', 'physx_num_threads', 'tensor_device', 'actor_device'):
            if actual_backend[key] != expected_backend[key]:
                raise ValueError('native backend contract drift for %s: expected %r, got %r' %
                                 (key, expected_backend[key], actual_backend[key]))
        identity = dict(physics_hash=fingerprint(physics), controller_hash=fingerprint(controller),
                        backend=actual_backend)
        if int(task.gym.get_frame_count(task.sim)) != 0:
            raise ValueError('serial replay requires a fresh unstepped simulator')
        reset_rng_anchor = None

        def capture(control):
            nonlocal executed
            executed = control.detach().cpu().numpy().copy()
            return original_pre(control)

        task.pre_physics_step = capture
        def reset_episode():
            nonlocal obs, reset_rng_anchor
            if reset_rng_anchor is None:
                reset_rng_anchor = capture_native_rng(torch)
            else:
                restore_native_rng(reset_rng_anchor, torch)
            pre_frame = int(task.gym.get_frame_count(task.sim))
            pre_rng_hash = fingerprint(capture_native_rng(torch))
            obs = player.env_reset(env_ids)
            player.get_batch_size(obs['obs'], 1)
            if task.start_times[0].item() != 0 or task.progress_buf[0].item() != 0:
                raise ValueError('serial reset did not return to reference tick0')
            frame0 = int(task.gym.get_frame_count(task.sim))
            if frame0 != pre_frame:
                raise ValueError('serial reset advanced the simulator frame counter')
            state = {key: getattr(task, key).cpu().numpy().copy() for key in state_keys}
            points, gap = geometry.measure(task); support, footprint = table.measure(task, geometry)
            values = dict(object_pose=pose_matrix(task._target_states[0].cpu().numpy()),
                          hand_keypoints=points[0].cpu().numpy().copy(),
                          surface_gap=gap[0].cpu().numpy().copy(),
                          support_gap=support[0].cpu().numpy().copy(),
                          table_footprint=footprint[0].cpu().numpy().copy(),
                          dof_position=task._dof_pos[0].cpu().numpy().copy(),
                          dof_velocity=task._dof_vel[0].cpu().numpy().copy(),
                          object_velocity=task._target_states[0, 7:13].cpu().numpy().copy(),
                          history=obs['obs'][0].cpu().numpy().copy(),
                          native_contact_forces=task._contact_forces[0].cpu().numpy().copy(),
                          native_object_contact_forces=task._tar_contact_forces[0].cpu().numpy().copy())
            post_rng_hash = fingerprint(capture_native_rng(torch))
            return dict(pre_frame=pre_frame, frame=frame0, state=state, values=values,
                        pre_rng_hash=pre_rng_hash, post_rng_hash=post_rng_hash)

        def run_episode(name, replay_actions=None, candidate=None):
            nonlocal obs, executed
            reset = reset_episode()
            frame0, initial_state = reset['frame'], reset['state']
            initial_values, initial_rng = reset['values'], reset['post_rng_hash']
            rows = {key: [np.asarray(value).copy()] for key, value in initial_values.items()}
            states = [initial_state]; state_hashes = [fingerprint(initial_state)]
            rng_hashes = [initial_rng]; actions = []; ended = []
            for tick in range(stop):
                if time.monotonic() - started > 240:
                    raise TimeoutError('serial replay deadline')
                if replay_actions is None:
                    control = legacy_batched_actor_action(player, obs).clamp(-1, 1)
                else:
                    control = torch.as_tensor(replay_actions[tick], dtype=torch.float32,
                                              device=player.device).view(1, -1)
                if candidate in (1, 2) and query <= tick < query + horizon:
                    residual = torch.as_tensor(candidate_plan(candidate)[tick - query],
                                               device=player.device).view(1, -1)
                    control = (control + residual).clamp(-1, 1)
                executed = None
                obs, _, done, info = player.env_step(player.env, control.clone())
                if not isinstance(obs, dict): obs = {'obs': obs}
                player._post_step(info)
                if executed is None or executed.shape != (1, 18) or not np.isfinite(executed).all():
                    raise ValueError('serial replay captured invalid executed control')
                actions.append(executed[0].copy()); ended.append(bool(done[0].item()))
                points, gap = geometry.measure(task); support, footprint = table.measure(task, geometry)
                values = dict(object_pose=pose_matrix(task._target_states[0].cpu().numpy()),
                              hand_keypoints=points[0].cpu().numpy().copy(),
                              surface_gap=gap[0].cpu().numpy().copy(),
                              support_gap=support[0].cpu().numpy().copy(),
                              table_footprint=footprint[0].cpu().numpy().copy(),
                              dof_position=task._dof_pos[0].cpu().numpy().copy(),
                              dof_velocity=task._dof_vel[0].cpu().numpy().copy(),
                              object_velocity=task._target_states[0, 7:13].cpu().numpy().copy(),
                              history=obs['obs'][0].cpu().numpy().copy(),
                              native_contact_forces=task._contact_forces[0].cpu().numpy().copy(),
                              native_object_contact_forces=task._tar_contact_forces[0].cpu().numpy().copy())
                for key, value in values.items(): rows[key].append(np.asarray(value).copy())
                state = {key: getattr(task, key).cpu().numpy().copy() for key in state_keys}
                states.append(state); state_hashes.append(fingerprint(state))
                rng_hashes.append(fingerprint(capture_native_rng(torch)))
                if int(task.gym.get_frame_count(task.sim)) != frame0 + (tick + 1) * task.control_freq_inv:
                    raise ValueError('serial replay frame counter advanced unexpectedly')
                if done.any() and tick < stop - 1:
                    raise ValueError('serial replay episode terminated before requested window')
            packet = {key: np.asarray(value) for key, value in rows.items()}
            packet.update(actions=np.asarray(actions), done=np.asarray(ended), state_traces=states,
                          state_hashes=state_hashes, rng_hashes=rng_hashes,
                          timestamps=np.arange(stop + 1) / 30, role=name,
                          reset_frame=frame0, pre_reset_frame=reset['pre_frame'],
                          reset_state=initial_state, reset_rng_hash=initial_rng,
                          pre_reset_rng_hash=reset['pre_rng_hash'],
                          post_reset_rng_hash=reset['post_rng_hash'])
            packet['outcome'] = episode_outcome(packet)
            return packet

        teacher_role = 'reactive_teacher' if a.engineering_serial_cluster else 'baseline'
        baseline = run_episode(teacher_role)
        baseline_actions = baseline['actions'].copy()
        if a.engineering_serial_cluster:
            # The teacher is a behavior screen and action-stream source.  All
            # subsequent arms replay that exact stream; candidate arms modify
            # only the query window.  Interleaving reduces a simple warm-cache
            # confound without pretending that reset restores hidden PhysX state.
            arm_specs = [('frozen_zero_1', None), ('positive_1', 1),
                         ('frozen_zero_2', None), ('negative_1', 2),
                         ('frozen_zero_3', None), ('negative_2', 2),
                         ('frozen_zero_4', None), ('positive_2', 1),
                         ('frozen_zero_5', None)]
        else:
            arm_specs = ([('zero_repeat_1', None), ('zero_repeat_2', None),
                          ('positive', 1), ('negative', 2)]
                         if a.engineering_serial_order == 'zero-first' else
                         [('positive', 1), ('negative', 2),
                          ('zero_repeat_1', None), ('zero_repeat_2', None)])
        repeats = [run_episode(name, baseline_actions, candidate) for name, candidate in arm_specs]
        execution_packets = [baseline] + repeats
        execution_order = [packet['role'] for packet in execution_packets]
        for execution_index, packet in enumerate(execution_packets):
            packet['execution_index'] = execution_index
        by_role = {packet['role']: packet for packet in execution_packets}
        canonical_roles = (['reactive_teacher'] + [name for name, _ in arm_specs]
                           if a.engineering_serial_cluster else
                           ['baseline', 'zero_repeat_1', 'zero_repeat_2', 'positive', 'negative'])
        all_packets = [by_role[role] for role in canonical_roles]
        reset_reference = baseline['reset_state']
        reset_diagnostics = []
        for packet in all_packets:
            differences = {}
            for key, expected in reset_reference.items():
                actual = packet['reset_state'][key]
                differences[key] = float(np.max(np.abs(np.asarray(actual, dtype=np.float64)
                                                       - np.asarray(expected, dtype=np.float64))))
            reset_diagnostics.append(dict(role=packet['role'], pre_reset_frame=packet['pre_reset_frame'],
                                          reset_frame=packet['reset_frame'],
                                          reset_frame_unchanged=packet['pre_reset_frame'] == packet['reset_frame'],
                                          pre_reset_rng_hash=packet['pre_reset_rng_hash'],
                                          post_reset_rng_hash=packet['post_reset_rng_hash'],
                                          reset_rng_equal=packet['reset_rng_hash'] == baseline['reset_rng_hash'],
                                          state_max_abs=differences, state_exact=all(value == 0 for value in differences.values())))
        def trace_diff(left, right):
            if len(left) != len(right):
                return dict(lengths=(len(left), len(right)))
            first = None; max_by_key = {}
            for tick, (left_state, right_state) in enumerate(zip(left, right)):
                for key in sorted(set(left_state) | set(right_state)):
                    if key not in left_state or key not in right_state:
                        max_by_key[key] = float('inf')
                        first = first or dict(tick=tick, field=key)
                        continue
                    delta = np.asarray(left_state[key], dtype=np.float64) - np.asarray(right_state[key], dtype=np.float64)
                    value = float(np.max(np.abs(delta))) if delta.size else 0.
                    max_by_key[key] = max(max_by_key.get(key, 0.), value)
                    if value != 0. and first is None:
                        first = dict(tick=tick, field=key, max_abs=value)
            return dict(first_divergence=first, max_abs_by_field=max_by_key,
                        exact=first is None)
        zero_roles = ([name for name, candidate in arm_specs if candidate is None]
                      if a.engineering_serial_cluster else
                      ['zero_repeat_1', 'zero_repeat_2'])
        candidate_roles = ([name for name, candidate in arm_specs if candidate is not None]
                           if a.engineering_serial_cluster else ['positive', 'negative'])
        zero1, zero2 = by_role[zero_roles[0]], by_role[zero_roles[1]]
        repeat_mismatch = []
        for key in fields + ('actions', 'done'):
            if not np.array_equal(baseline[key], zero1[key]):
                repeat_mismatch.append(key)
        teacher_mismatch = list(repeat_mismatch)
        teacher_state_diff = trace_diff(baseline['state_traces'], zero1['state_traces'])
        zero_state_diff = trace_diff(zero1['state_traces'], zero2['state_traces'])
        if a.engineering_serial_cluster:
            repeat_mismatch = []
            for key in fields + ('actions', 'done'):
                if not np.array_equal(zero1[key], zero2[key]):
                    repeat_mismatch.append(key)
            if not zero_state_diff['exact']:
                repeat_mismatch.append('state_traces')
            if zero1['rng_hashes'] != zero2['rng_hashes']:
                repeat_mismatch.append('rng_hashes')
            state_diff = zero_state_diff
        else:
            if not teacher_state_diff['exact']:
                repeat_mismatch.append('state_traces')
            if baseline['rng_hashes'] != zero1['rng_hashes']:
                repeat_mismatch.append('rng_hashes')
            state_diff = teacher_state_diff
        control_prefix_exact = all(np.array_equal(baseline_actions[:query], packet['actions'][:query])
                                   for packet in repeats)
        def action_delta_summary(packet):
            delta = np.asarray(packet['actions']) - baseline_actions
            nonzero = np.flatnonzero(np.any(delta != 0., axis=1))
            return dict(max_abs=float(np.max(np.abs(delta))),
                        l2=float(np.linalg.norm(delta)),
                        first_nonzero_tick=None if len(nonzero) == 0 else int(nonzero[0]))
        candidate_deltas = {packet['role']: action_delta_summary(packet) for packet in all_packets}
        schema = ('ref2dex.consequence-gate1.gpu-serial-cluster.v1'
                  if a.engineering_serial_cluster else
                  'ref2dex.consequence-gate1.gpu-serial-replay.v1')
        result = dict(schema=schema, engineering_only=True,
                      training_allowed=False, serial_replay=True,
                      serial_cluster=bool(a.engineering_serial_cluster),
                      group_mode=('same_process_same_env_reset_frozen_control_cluster'
                                  if a.engineering_serial_cluster else 'same_process_same_env_reset'),
                      roles=canonical_roles, execution_order=execution_order, group_envs=1,
                      query_tick=query, horizon=horizon, steps=stop, seed=a.seed,
                      outcome_complete=bool(stop >= 542),
                      source_backend=actual_backend, replay_identity=identity,
                      reset_diagnostics=reset_diagnostics,
                      baseline_repeat_mismatches=None if a.engineering_serial_cluster else repeat_mismatch,
                      teacher_vs_zero_mismatches=teacher_mismatch if a.engineering_serial_cluster else None,
                      teacher_vs_zero_state_diff=teacher_state_diff if a.engineering_serial_cluster else None,
                      zero_repeat_state_diff=state_diff, zero_pair_state_diff=zero_state_diff,
                      baseline_repeat_rng_exact=(zero1['rng_hashes'] == zero2['rng_hashes']
                                                 if a.engineering_serial_cluster
                                                 else baseline['rng_hashes'] == zero1['rng_hashes']),
                      serial_arm_order=None if a.engineering_serial_cluster else a.engineering_serial_order,
                      control_prefix_exact=control_prefix_exact,
                      candidate_plan_sha256={str(candidate): fingerprint(candidate_plan(candidate)) for candidate in (0, 1, 2)},
                      candidate_actual_action_delta=candidate_deltas,
                      baseline_outcome=None if a.engineering_serial_cluster else baseline['outcome'],
                      teacher_outcome=baseline['outcome'],
                      role_outcomes=(None if a.engineering_serial_cluster
                                     else {p['role']: p['outcome'] for p in all_packets}),
                      teacher_role=teacher_role, zero_roles=zero_roles, candidate_roles=candidate_roles,
                      replay_source='reactive_teacher_executed_actions' if a.engineering_serial_cluster else 'baseline_executed_actions',
                      teacher_action_sha256=fingerprint(baseline_actions),
                      schedule_sha256=fingerprint(execution_order),
                      candidate_effect='candidate residuals applied to recorded teacher controls at tick48 for24 steps',
                      note='engineering reset/cache diagnostic; serial reset is not a fresh-simulator twin')
        def requested_trace(role):
            trace = np.zeros_like(baseline['actions'])
            candidate = (1 if role.startswith('positive') else
                         2 if role.startswith('negative') else None)
            if candidate is not None:
                trace[query:query + horizon] = candidate_plan(candidate)
            return trace
        packet = dict(schema=result['schema'], engineering_only=True, training_allowed=False,
                      **{key: np.stack([p[key] for p in all_packets]) for key in fields + ('actions', 'done')},
                      state_traces=[p['state_traces'] for p in all_packets],
                      state_hashes=[p['state_hashes'] for p in all_packets],
                      rng_hashes=[p['rng_hashes'] for p in all_packets],
                      timestamps=baseline['timestamps'],
                      requested_residual=np.stack([requested_trace(p['role']) for p in all_packets]),
                      roles=result['roles'], query_tick=query, horizon=horizon, steps=stop,
                      seed=a.seed, source_backend=actual_backend, replay_identity=identity,
                      outcome_complete=bool(stop >= 542),
                      serial_cluster=bool(a.engineering_serial_cluster),
                      group_mode=result['group_mode'],
                      reset_diagnostics=reset_diagnostics,
                      baseline_repeat_mismatches=None if a.engineering_serial_cluster else repeat_mismatch,
                      teacher_vs_zero_mismatches=teacher_mismatch if a.engineering_serial_cluster else None,
                      teacher_vs_zero_state_diff=teacher_state_diff if a.engineering_serial_cluster else None,
                      zero_repeat_state_diff=state_diff, zero_pair_state_diff=zero_state_diff,
                      control_prefix_exact=control_prefix_exact,
                      serial_arm_order=None if a.engineering_serial_cluster else a.engineering_serial_order,
                      execution_order=execution_order, teacher_role=teacher_role,
                      zero_roles=zero_roles, candidate_roles=candidate_roles,
                      replay_source=result['replay_source'], schedule_sha256=result['schedule_sha256'],
                      teacher_action_sha256=result['teacher_action_sha256'],
                      teacher_outcome=result['teacher_outcome'], role_outcomes=result['role_outcomes'])
        save(a.worker_output, packet)
        write(a.worker_output.with_suffix('.json'), result)

    def engineering_single_act_run(player, task):
        """Screen one native GPU episode with one frozen open-loop ACT chunk.

        This is deliberately a behavior-only diagnostic.  The reactive native
        actor supplies controls through tick47.  At tick48 the current native
        observation is transformed once and decoded by the frozen action-chunk
        checkpoint; the resulting 24 controls are then fed directly through
        the native pre-physics boundary without reading a later observation.
        No candidate residual, second environment, or serial reset is involved.
        """
        if backend.name != 'gpu_physx_gpu_pipeline':
            raise ValueError('single ACT behavior requires native GPU PhysX/GPU pipeline')
        if (player.is_rnn or task.num_envs != 1 or abs(task.dt - 1 / 30) > 1e-8
                or abs(task.sim_params.dt - 1 / 60) > 1e-8 or task.control_freq_inv != 2):
            raise ValueError('single ACT behavior requires one fixed nonrecurrent native environment')
        if int(task.num_motions) != 1:
            raise ValueError('single ACT behavior requires one repeated motion')
        if a.engineering_steps != 72 or not a.finish:
            raise ValueError('single ACT behavior requires the fixed 72-step finished screen')
        if a.query_tick not in (0, 48):
            raise ValueError('single ACT behavior requires the established query tick48 contract')
        if a.action_chunk_checkpoint is None or a.action_chunk_replay is not None:
            raise ValueError('single ACT behavior requires a checkpoint and no replay packet')
        query = 48
        horizon = 24
        stop = 72

        task._state_init = DexploreTask.StateInit.Start
        task._hybrid_init_prob = 1.
        task._adaptive_kappa_enabled = False
        task._enable_early_termination = False
        if task.dr_randomizations or task._motion_sampler is not None or task.projtype != 'None':
            raise ValueError('randomized/sampler/projectile single ACT behavior unsupported')
        geometry = PhysicalGeometry(task, native_root / 'data/assets')
        table = TableSupport(native_root / 'data/assets', task.device)

        from consequence_evaluator.action_chunk import EXECUTED_ACTION_SEMANTICS, NativeActionChunkProposal
        payload = torch.load(a.action_chunk_checkpoint, map_location='cpu')
        if (payload.get('schema') != 'ref2dex.consequence-evaluator.native-action-chunks.v1'
                or payload.get('chunk') != K or payload.get('action_dim') != 18
                or payload.get('history_dim') != int(task.num_obs)
                or payload.get('executed_action_semantics') != EXECUTED_ACTION_SEMANTICS):
            raise ValueError('native action-chunk checkpoint contract mismatch')
        checkpoint_manifest = a.action_chunk_checkpoint.parent / 'manifest.json'
        if not checkpoint_manifest.exists():
            raise ValueError('single ACT behavior requires a sibling proposal manifest')
        proposal_manifest = json.loads(checkpoint_manifest.read_text())
        manifest_checkpoint = proposal_manifest.get('checkpoint')
        if (manifest_checkpoint is None
                or Path(manifest_checkpoint).resolve() != a.action_chunk_checkpoint.resolve()
                or proposal_manifest.get('checkpoint_sha256') != sha(a.action_chunk_checkpoint)
                or proposal_manifest.get('engineering_only') is not True
                or proposal_manifest.get('training_allowed') is not False):
            raise ValueError('single ACT behavior proposal manifest does not authenticate the checkpoint')
        standardizer = payload.get('history_standardizer')
        if (not isinstance(standardizer, dict)
                or len(standardizer.get('mean', [])) != int(task.num_obs)
                or len(standardizer.get('scale', [])) != int(task.num_obs)):
            raise ValueError('native action-chunk history standardizer mismatch')
        proposal_mean = torch.as_tensor(standardizer['mean'], dtype=torch.float32, device=player.device)
        proposal_scale = torch.as_tensor(standardizer['scale'], dtype=torch.float32, device=player.device)
        proposal_clip = standardizer.get('clip')
        if (proposal_mean.ndim != 1 or proposal_scale.ndim != 1
                or proposal_mean.numel() != int(task.num_obs)
                or proposal_scale.numel() != int(task.num_obs)
                or not torch.isfinite(proposal_mean).all()
                or not torch.isfinite(proposal_scale).all()
                or (proposal_scale <= 0).any()
                or (proposal_clip is not None
                    and (not np.isfinite(proposal_clip) or float(proposal_clip) <= 0))):
            raise ValueError('native action-chunk history standardizer is invalid')
        proposal_model = NativeActionChunkProposal(
            int(payload['history_dim']), width=int(payload['width']),
            layers=int(payload['layers'])).to(player.device)
        proposal_model.load_state_dict(payload['state_dict'], strict=True)
        proposal_model.eval()

        env_ids = torch.arange(1, device=task.device)
        obs = player.env_reset(env_ids)
        player.get_batch_size(obs['obs'], 1)
        if task.gym.get_frame_count(task.sim) != 0 or (task.start_times != 0).any():
            raise ValueError('fresh unstepped single ACT reset required')
        total = int(task.max_episode_length[task.data_id[0]]) - 1
        if total != 542:
            raise ValueError('frozen full543state reference required')

        physics = {'sim_params': normalize_properties(task.gym.get_sim_params(task.sim)), 'actors': []}
        for env in task.envs:
            actors = []
            for actor_index in range(task.gym.get_actor_count(env)):
                handle = task.gym.get_actor_handle(env, actor_index)
                actors.append(dict(
                    name=task.gym.get_actor_name(env, actor_index),
                    shape=normalize_properties(task.gym.get_actor_rigid_shape_properties(env, handle)),
                    body=normalize_properties(task.gym.get_actor_rigid_body_properties(env, handle)),
                    dof=normalize_properties(task.gym.get_actor_dof_properties(env, handle))))
            physics['actors'].append(actors)
        controller = dict(
            model={k: v.cpu().numpy().copy() for k, v in player.model.state_dict().items()},
            normalize_input=bool(player.normalize_input),
            checkpoint_sha256=trained['checkpoint_sha256'])
        for key, normalizer in [('rms', getattr(player, 'running_mean_std', None)),
                                ('amp_rms', getattr(player, '_amp_input_mean_std', None))]:
            controller[key] = None if normalizer is None else {
                k: v.cpu().numpy().copy() for k, v in normalizer.state_dict().items()}
        params = task.gym.get_sim_params(task.sim)
        actual_backend = dict(
            name=backend.name, sim_device=backend.sim_device,
            pipeline='gpu' if bool(params.use_gpu_pipeline) else 'cpu',
            physx_use_gpu=bool(params.physx.use_gpu),
            physx_num_threads=int(params.physx.num_threads),
            tensor_device=canonical_device(task.device),
            actor_device=canonical_device(player.device))
        expected_backend = backend.as_dict()
        for key in ('pipeline', 'physx_use_gpu', 'physx_num_threads', 'tensor_device', 'actor_device'):
            if actual_backend[key] != expected_backend[key]:
                raise ValueError('native backend contract drift for %s: expected %r, got %r' %
                                 (key, expected_backend[key], actual_backend[key]))
        identity = dict(
            physics_hash=fingerprint(physics), controller_hash=fingerprint(controller),
            backend=actual_backend,
            environment=dict(cfg_env_sha256=sha(cfg_env), env_spacing_override=None),
            query_tick=query,
            actor_execution=dict(layout='single_environment_64row_inference', copies=64, total_rows=64))

        state_keys = sorted(REQUIRED_NATIVE_STATE_KEYS | {
            key for key in ('reset_buf', 'actions', 'real_pd_tar')
            if isinstance(getattr(task, key, None), torch.Tensor)})
        fields = ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap',
                  'table_footprint', 'dof_position', 'dof_velocity', 'object_velocity',
                  'history', 'native_contact_forces', 'native_object_contact_forces')
        rows = {key: [] for key in fields}
        actions = []
        requested_controls = []
        ended = []
        state_hashes = []
        state_field_hashes = []
        rng_hashes = []

        def observe():
            points, gap = geometry.measure(task)
            support, footprint = table.measure(task, geometry)
            values = dict(
                object_pose=pose_matrix(task._target_states[0].cpu().numpy()),
                hand_keypoints=points[0].cpu().numpy().copy(),
                surface_gap=gap[0].cpu().numpy().copy(),
                support_gap=support[0].cpu().numpy().copy(),
                table_footprint=footprint[0].cpu().numpy().copy(),
                dof_position=task._dof_pos[0].cpu().numpy().copy(),
                dof_velocity=task._dof_vel[0].cpu().numpy().copy(),
                object_velocity=task._target_states[0, 7:13].cpu().numpy().copy(),
                history=obs['obs'][0].cpu().numpy().copy(),
                native_contact_forces=task._contact_forces[0].cpu().numpy().copy(),
                native_object_contact_forces=task._tar_contact_forces[0].cpu().numpy().copy())
            for key, value in values.items():
                rows[key].append(np.asarray(value).copy())
            state = {key: getattr(task, key).cpu().numpy().copy() for key in state_keys}
            state_hashes.append(fingerprint(state))
            state_field_hashes.append({key: fingerprint(value) for key, value in state.items()})
            rng_hashes.append(fingerprint(capture_native_rng(torch)))
            index = len(state_hashes) - 1
            if task.gym.get_frame_count(task.sim) != index * task.control_freq_inv:
                raise ValueError('unexpected simulator step count at single ACT tick%d' % index)

        executed = None
        original_pre = task.pre_physics_step

        def capture(control):
            nonlocal executed
            executed = control.detach().cpu().numpy().copy()
            return original_pre(control)

        task.pre_physics_step = capture
        observe()
        proposal_cache = None
        proposal_input = None
        proposal_trace = []
        for tick in range(stop):
            if time.monotonic() - started > 150:
                raise TimeoutError('bounded single ACT behavior deadline')
            if tick < query:
                control = legacy_batched_actor_action(player, obs).clamp(-1, 1)
            elif tick == query:
                proposal_input = (obs['obs'] - proposal_mean) / proposal_scale
                if proposal_clip is not None:
                    proposal_input = proposal_input.clamp(-float(proposal_clip), float(proposal_clip))
                proposal_cache = proposal_model(proposal_input).clamp(-1, 1)
                proposal_trace.append(proposal_cache.detach().cpu().numpy().copy())
                control = proposal_cache[:, 0]
            else:
                if proposal_cache is None:
                    raise RuntimeError('single ACT action chunk was not initialized at tick48')
                control = proposal_cache[:, tick - query]
            requested = control.detach().cpu().numpy()
            if requested.shape != (1, 18):
                raise ValueError('single ACT requested control shape changed')
            requested_controls.append(requested[0].copy())
            obs, _, done, info = player.env_step(player.env, control.clone())
            if not isinstance(obs, dict):
                obs = {'obs': obs}
            player._post_step(info)
            if (executed is None or executed.shape != (1, 18)
                    or not np.isfinite(executed).all() or np.abs(executed).max() > 1 + 1e-6):
                raise ValueError('nonfinite/out-of-bounds single ACT native command')
            actions.append(executed[0].copy())
            ended.append(bool(done[0].item()))
            observe()
            if done.any() and tick < stop - 1:
                raise ValueError('single ACT episode terminated before the 72-step screen')

        packet = {key: np.asarray(value) for key, value in rows.items()}
        requested_controls = np.asarray(requested_controls)
        captured_actions = np.asarray(actions)
        action_delta = np.abs(captured_actions - requested_controls)
        action_delta_max = float(action_delta.max()) if action_delta.size else 0.0
        if not np.isfinite(action_delta).all() or action_delta_max > 1e-7:
            raise ValueError('native captured actions differ from requested single ACT controls')
        packet.update(
            actions=captured_actions, requested_controls=requested_controls,
            native_action_max_abs_delta=action_delta_max,
            native_action_contract_exact=True, done=np.asarray(ended), state_hashes=state_hashes,
            state_field_hashes=state_field_hashes, rng_hashes=rng_hashes,
            timestamps=np.arange(stop + 1) / 30,
            schema='ref2dex.consequence-gate1.native-single-act-behavior.v1',
            engineering_only=True, training_allowed=False,
            single_act_behavior=True, group_envs=1,
            role='single_env_open_loop24_act',
            prefix_role='reactive_native_policy', query_tick=query, horizon=horizon, steps=stop,
            prefix_semantics='reactive native actor controls through tick47',
            action_chunk_mode='open_loop24',
            action_chunk_semantics='one-shot native18 chunk at tick48; no future observation feedback',
            proposal_generated_tick=query, proposal_model_calls=1,
            future_observation_feedback=False,
            frozen_action_ticks=[query, query + horizon - 1],
            action_chunk_checkpoint=str(a.action_chunk_checkpoint),
            action_chunk_checkpoint_sha256=sha(a.action_chunk_checkpoint),
            proposal_chunks=np.asarray(proposal_trace)[:, 0],
            proposal_chunk=proposal_cache[0].detach().cpu().numpy().copy(),
            proposal_input=proposal_input.detach().cpu().numpy().copy(),
            replay_identity=identity, canonical_state_keys=state_keys,
            actor_inference_batch=64, actor_inference_copies=64,
            source_backend=actual_backend,
            prefix_action_sha256=fingerprint(captured_actions[:query]),
            rng_semantics='native torch/python/numpy RNG captured per tick; no future observation feedback',
            outcome_complete=False)
        packet['outcome'] = episode_outcome(packet)
        packet['max_lift_m'] = float(np.max(
            packet['object_pose'][:, 2, 3] - packet['object_pose'][0, 2, 3]))
        save(a.worker_output, packet)
        write(a.worker_output.with_suffix('.json'), dict(
            status='COMPLETED', schema=packet['schema'], engineering_only=True,
            single_act_behavior=True, role=packet['role'], steps=stop,
            query_tick=query, horizon=horizon, action_chunk_mode='open_loop24',
            proposal_generated_tick=query, proposal_model_calls=1,
            future_observation_feedback=False,
            action_chunk_checkpoint=str(a.action_chunk_checkpoint),
            action_chunk_checkpoint_sha256=packet['action_chunk_checkpoint_sha256'],
            native_action_max_abs_delta=packet['native_action_max_abs_delta'],
            native_action_contract_exact=True,
            elapsed_s=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            outcome=packet['outcome'], max_lift_m=packet['max_lift_m'],
            replay_identity=identity))

    class GatePlayer(base):
        def restore(self, filename):
            value = native.torch_ext.load_checkpoint(filename)
            self._gate_restored_model_hash = fingerprint(compat.model_state(value['model']))
            self.model.load_state_dict(compat.model_state(value['model']), strict=True)
            if self.normalize_input:
                self.running_mean_std.load_state_dict(value['running_mean_std'], strict=True)
            if self._normalize_amp_input:
                self._amp_input_mean_std.load_state_dict(value['amp_input_mean_std'], strict=True)

        @torch.no_grad()
        def run(self):
            self.is_deterministic = self.is_determenistic
            torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
            if fingerprint(self.model.state_dict()) != self._gate_restored_model_hash:
                raise ValueError('actor weights changed after checkpoint restore')
            task = self.env.task
            if a.engineering_single_act:
                engineering_single_act_run(self, task)
                return
            if a.engineering_serial_replay or a.engineering_serial_cluster:
                engineering_serial_replay_run(self, task)
                return
            if a.engineering_group_envs:
                engineering_group_run(self, task)
                return
            if (self.is_rnn or task.num_envs != 1 or abs(task.dt - 1 / 30) > 1e-8
                    or abs(task.sim_params.dt - 1 / 60) > 1e-8 or task.control_freq_inv != 2):
                raise ValueError('fixed single-environment30Hz deterministic policy required')
            task._state_init = DexploreTask.StateInit.Start; task._hybrid_init_prob = 1.
            task._adaptive_kappa_enabled = False; task._enable_early_termination = False
            if task.dr_randomizations or task._motion_sampler is not None or task.projtype != 'None':
                raise ValueError('randomized/sampler/projectile replay unsupported')
            geometry = PhysicalGeometry(task, native_root / 'data/assets')
            table = TableSupport(native_root / 'data/assets', task.device)
            obs = self.env_reset(torch.arange(1, device=task.device))
            self.get_batch_size(obs['obs'], 1)
            if task.gym.get_frame_count(task.sim) != 0 or (task.start_times != 0).any():
                raise ValueError('fresh unstepped full-frame0 reset required')
            total = int(task.max_episode_length[task.data_id[0]]) - 1
            if total != 542:
                raise ValueError('frozen full543state reference required')
            physics = {'sim_params': normalize_properties(task.gym.get_sim_params(task.sim)), 'actors': []}
            for env in task.envs:
                actors = []
                for actor_index in range(task.gym.get_actor_count(env)):
                    handle = task.gym.get_actor_handle(env, actor_index)
                    actors.append(dict(name=task.gym.get_actor_name(env, actor_index),
                        shape=normalize_properties(task.gym.get_actor_rigid_shape_properties(env, handle)),
                        body=normalize_properties(task.gym.get_actor_rigid_body_properties(env, handle)),
                        dof=normalize_properties(task.gym.get_actor_dof_properties(env, handle))))
                physics['actors'].append(actors)
            controller = dict(model={k: v.cpu().numpy().copy() for k, v in self.model.state_dict().items()},
                normalize_input=bool(self.normalize_input), checkpoint_sha256=trained['checkpoint_sha256'])
            for key, normalizer in [('rms', getattr(self, 'running_mean_std', None)),
                                    ('amp_rms', getattr(self, '_amp_input_mean_std', None))]:
                controller[key] = None if normalizer is None else {k: v.cpu().numpy().copy() for k, v in normalizer.state_dict().items()}
            params = task.gym.get_sim_params(task.sim)
            actual_backend = dict(
                name=backend.name,
                sim_device=backend.sim_device,
                pipeline='gpu' if bool(params.use_gpu_pipeline) else 'cpu',
                physx_use_gpu=bool(params.physx.use_gpu),
                physx_num_threads=int(params.physx.num_threads),
                tensor_device=canonical_device(task.device),
                actor_device=canonical_device(self.device),
            )
            expected_backend = backend.as_dict()
            for key in ('pipeline', 'physx_use_gpu', 'physx_num_threads', 'tensor_device', 'actor_device'):
                if actual_backend[key] != expected_backend[key]:
                    raise ValueError('native backend contract drift for %s: expected %r, got %r' %
                                     (key, expected_backend[key], actual_backend[key]))
            identity = dict(physics_hash=fingerprint(physics), controller_hash=fingerprint(controller),
                backend=actual_backend)
            if expected is not None and expected['replay_identity'] != identity:
                raise ValueError('native physics/controller identity changed')
            state_keys = sorted(REQUIRED_NATIVE_STATE_KEYS | {k for k in ('reset_buf', 'actions', 'real_pd_tar') if isinstance(getattr(task, k, None), torch.Tensor)})
            fields = ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap',
                      'table_footprint', 'object_velocity', 'history', 'native_contact_forces', 'native_object_contact_forces')
            rows = {k: [] for k in fields}; replay_mismatches = []; hashes = []; state_field_hashes = []; rng_hashes = []; actions = []; ended = []
            def observe():
                points, gap = geometry.measure(task); support, footprint = table.measure(task, geometry)
                values = dict(object_pose=pose_matrix(task._target_states[0].cpu().numpy()),
                    hand_keypoints=points[0].cpu().numpy(), surface_gap=gap[0].cpu().numpy(),
                    support_gap=support[0].cpu().numpy(), table_footprint=footprint[0].cpu().numpy(),
                    object_velocity=task._target_states[0, 7:13].cpu().numpy(),
                    history=obs['obs'][0].cpu().numpy(),
                    native_contact_forces=task._contact_forces[0].cpu().numpy(),
                    native_object_contact_forces=task._tar_contact_forces[0].cpu().numpy())
                for k, v in values.items(): rows[k].append(np.asarray(v).copy())
                state = {k: getattr(task, k).cpu().numpy().copy() for k in state_keys}
                state_field_hashes.append({k: fingerprint(v) for k,v in state.items()})
                hashes.append(fingerprint(state)); rng_hashes.append(fingerprint(capture_native_rng(torch)))
                index = len(hashes) - 1
                if task.gym.get_frame_count(task.sim) != index * task.control_freq_inv:
                    raise ValueError('unexpected simulator step count at tick%d' % index)
                if expected is not None and index <= query:
                    if hashes[-1] != expected['state_hashes'][index] or rng_hashes[-1] != expected['rng_hashes'][index]:
                        diagnostic = dict(tick=index, state_equal=hashes[-1]==expected['state_hashes'][index], rng_equal=rng_hashes[-1]==expected['rng_hashes'][index],
                            changed_state_fields=[k for k,h in state_field_hashes[-1].items() if 'state_field_hashes' in expected and h!=expected['state_field_hashes'][index][k]],
                            measurement_max_abs_errors={k:float(np.max(np.abs(np.asarray(rows[k][-1],dtype=float)-np.asarray(expected[k][index],dtype=float)))) for k in fields})
                        force_only = (a.diagnostic_force_cache and diagnostic['rng_equal'] and
                            bool(diagnostic['changed_state_fields']) and
                            set(diagnostic['changed_state_fields']) <= {'_contact_forces', '_tar_contact_forces'} and
                            all(v == 0 for k,v in diagnostic['measurement_max_abs_errors'].items() if k not in ('native_contact_forces','native_object_contact_forces')))
                        if force_only:
                            replay_mismatches.append(diagnostic)
                        else:
                            save(a.worker_output.with_suffix('.mismatch.pkl'), {k:np.asarray(v) for k,v in rows.items()})
                            write(a.worker_output.with_suffix('.mismatch.json'), diagnostic)
                            raise ValueError('declared native/RNG prefix replay mismatch at tick%d: %s' % (index, diagnostic))
                    for k in fields:
                        if a.diagnostic_force_cache and k in ('native_contact_forces','native_object_contact_forces'):
                            continue
                        if not np.array_equal(rows[k][-1], expected[k][index]):
                            raise ValueError('measured prefix replay differs at tick%d: %s' % (index, k))
            executed = None; original_pre = task.pre_physics_step
            def capture(control):
                nonlocal executed
                executed = control.detach().cpu().numpy().copy()
                return original_pre(control)
            task.pre_physics_step = capture
            observe(); plan = candidate_plan(a.candidate)
            stop = (a.engineering_steps or total) if a.finish else query + K
            for tick in range(stop):
                if time.monotonic() - started > 150:
                    raise TimeoutError('bounded native worker deadline')
                control = legacy_batched_actor_action(self, obs).clamp(-1, 1)
                if tick < query:
                    control = torch.as_tensor(expected['actions'][tick:tick + 1], device=self.device)
                elif not a.finish:
                    control = (control + torch.as_tensor(plan[tick - query], device=self.device)).clamp(-1, 1)
                obs, _, done, info = self.env_step(self.env, control.clone())
                if not isinstance(obs, dict): obs = {'obs': obs}
                self._post_step(info)
                actual = executed[0].copy()
                if not np.isfinite(actual).all() or np.abs(actual).max() > 1 + 1e-6:
                    raise ValueError('nonfinite/out-of-bounds native command')
                if tick < query and not np.array_equal(actual, expected['actions'][tick]):
                    raise ValueError('executed prefix control changed')
                if tick < query and bool(done[0]) != bool(expected['done'][tick]):
                    raise ValueError('prefix done signal changed')
                actions.append(actual); ended.append(bool(done[0])); observe()
                if done.any() and tick < stop - 1:
                    raise ValueError('native episode terminated before full requested window')
            packet = {k: np.asarray(v) for k, v in rows.items()}
            packet.update(actions=np.asarray(actions), done=np.asarray(ended),
                state_hashes=hashes, state_field_hashes=state_field_hashes, rng_hashes=rng_hashes, timestamps=np.arange(stop + 1) / 30,
                seed=a.seed, query_tick=query, candidate=a.candidate, residual_plan=plan,
                checkpoint_sha256=trained['checkpoint_sha256'], fresh_prefix_replay=True,
                replay_identity=identity, canonical_state_keys=state_keys, actor_inference_batch=64,
                engineering_only=bool(a.diagnostic_force_cache or a.engineering_steps or (expected is not None and expected.get('engineering_only'))), replay_mismatches=replay_mismatches)
            save(a.worker_output, packet)
            write(a.worker_output.with_suffix('.json'), dict(status='COMPLETED', steps=stop,
                prefix_steps=query, all_prefix_native_rng_and_measurements_exact=not replay_mismatches,
                engineering_only=bool(a.diagnostic_force_cache or a.engineering_steps or (expected is not None and expected.get('engineering_only'))), replay_mismatches=replay_mismatches,
                elapsed_s=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                outcome=episode_outcome(packet) if a.finish else None))
    native.EvalPlayer = GatePlayer
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', str(cfg_env),
        '--cfg_train', str(native_root / 'data/cfg/train/rlg/inspire.yaml'),
        '--motion_file', config['motion_root'], '--checkpoint', trained['checkpoint'],
        '--headless', '--disable-early-termination', '--num_envs', str(a.engineering_group_envs or 1), '--seed', str(a.seed),
        *backend.argv(), '--graphics_device_id', '0',
        '--output', str(a.worker_output.with_suffix('.unused.json')),
        '--output_path', str(a.worker_output.with_suffix('.runtime'))]
    os.chdir(ROOT / 'third_party/DExplore'); native.main()


def score_worker(a):
    ensure_free(a.gpu); os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from consequence_evaluator.reference_bank import load_reference_features, BANK_SCHEMA
    from consequence_evaluator.reference_progress import trajectory_features
    from consequence_evaluator.temporal_phase import LearnedReferenceProgress, ENCODER_SCHEMA
    torch.set_num_threads(1)
    meta = json.loads((a.reference / 'manifest.json').read_text())
    trained = json.loads((a.encoder / 'manifest.json').read_text())
    checkpoint = Path(trained['checkpoint'])
    if (meta['schema'] != BANK_SCHEMA or trained['schema'] != ENCODER_SCHEMA
            or trained['status'] != 'COMPLETED' or trained['reference_sha256'] != meta['reference_sha256']
            or sha(checkpoint) != trained['checkpoint_sha256']):
        raise ValueError('exact frozen physical-bank phase encoder required')
    if any(sha(p) != h for p, h in {**meta['sources'], **trained['sources']}.items()):
        raise ValueError('frozen phase model/reference source drift')
    references, _ = load_reference_features(a.reference, meta)
    matcher = LearnedReferenceProgress(references, torch.load(checkpoint, map_location='cpu', weights_only=False), 'cuda:0')
    packets = [load(p) for p in a.score_inputs]
    if any(p.get('engineering_only') for p in packets):
        raise ValueError('engineering traces cannot enter GT scoring')
    values = []; progress_start = []
    for packet in packets:
        trace = matcher.align(trajectory_features(packet['object_pose'], packet['hand_keypoints'], packet['timestamps']))
        tick = packet['query_tick']; end = tick + K
        if len(packet['actions']) != end:
            raise ValueError('candidate GT future must end exactly at t+24')
        values.append(float(trace['progress'][end] - trace['progress'][tick]))
        progress_start.append(float(trace['progress'][tick]))
    if max(progress_start) - min(progress_start) > 1e-9:
        raise ValueError('candidate prefixes have different current progress')
    write(a.score_output, dict(values=values, progress_start=progress_start,
        choice=choose_candidate(values), model_role='phase-only; no evaluator',
        horizon=K, execution_horizon=K_EXEC, no_candidate_future_after_horizon=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--encoder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    backend_choices = tuple(sorted(BACKENDS))
    p.add_argument('--backend', choices=backend_choices,
                   help='native execution contract; host/gpu_physx_cpu_pipeline keeps GPU PhysX with CPU tensor exchange')
    p.add_argument('--physics-device', choices=backend_choices, default=None,
                   help='deprecated alias for --backend; retained for old commands')
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--episodes', type=int, default=4)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--worker-output', type=Path)
    p.add_argument('--prefix', type=Path)
    p.add_argument('--query-tick', type=int, default=0)
    p.add_argument('--seed', type=int, default=282)
    p.add_argument('--candidate', type=int, default=0)
    p.add_argument('--finish', action='store_true')
    p.add_argument('--engineering-steps', type=int, default=0, help='native worker only: bounded diagnostic baseline, never GT scoring')
    p.add_argument('--engineering-single-act', action='store_true',
                   help='native worker only: one-env reactive-prefix/open-loop24 ACT behavior screen, never GT scoring')
    p.add_argument('--engineering-group-envs', type=int, default=0,
                   help='native worker only: synchronous same-process group (baseline/zero/+/-), never GT scoring')
    p.add_argument('--engineering-actor-copies', type=int, default=64,
                   help='engineering group only: fixed actor copies per env (4x64 is the legacy 256-row contract)')
    p.add_argument('--engineering-env-spacing', type=float, default=None,
                   help='engineering group only: override envSpacing to reduce world-origin float drift')
    p.add_argument('--engineering-serial-replay', action='store_true',
                   help='native worker only: same-process one-env reset/replay diagnostic, never GT scoring')
    p.add_argument('--engineering-serial-cluster', action='store_true',
                   help='native worker only: teacher plus interleaved frozen-control serial cluster, never GT scoring')
    p.add_argument('--engineering-serial-order', choices=('zero-first', 'candidate-first'), default='zero-first',
                   help='serial replay arm order after baseline; candidate-first is a warm-cache order probe')
    p.add_argument('--zero-env-pair', type=int, nargs=2, default=(0, 1), metavar=('LEFT', 'RIGHT'),
                   help='engineering group only: the two non-candidate env roles used for zero-noise calibration')
    p.add_argument('--candidate-env-pair', type=int, nargs=2, default=(2, 3), metavar=('POSITIVE', 'NEGATIVE'),
                   help='engineering group only: environment roles receiving the positive/negative residuals')
    p.add_argument('--diagnostic-force-cache', action='store_true', help='engineering only: inspect derived force differences; all other prefix fields remain strict')
    p.add_argument('--action-chunk-checkpoint', type=Path,
                   help='engineering worker only: frozen native 24-step proposal checkpoint')
    p.add_argument('--retargeter-checkpoint', type=Path, help='engineering only: GT-hand learned retargeter')
    p.add_argument('--retargeter-source', type=Path, help='engineering only: held-out full teacher GT hand packet')
    p.add_argument('--object-relative-source', type=Path, help='engineering only: analytic GT wrist transport teacher source')
    p.add_argument('--object-relative-anchor', choices=('current', 'future'), default='current')
    p.add_argument('--object-relative-layout', choices=('command_vs_measured', 'transport_ablation', 'chunk_alignment', 'finger_preload', 'finger_preload_late', 'wrist_geometry_late', 'geometry_inverse_late', 'geometry_inverse_relative_late', 'geometry_inverse_repeat', 'geometry_load_delta_late', 'geometry_pd_inverse', 'measured_pd_load_memory', 'geometry_pd'), default='command_vs_measured')
    p.add_argument('--object-relative-inverse', type=Path)
    p.add_argument('--object-relative-pd-inverse', type=Path)
    p.add_argument('--object-relative-preload', type=Path, help='engineering only: train-only fixed contact preload statistics')
    p.add_argument('--action-chunk-replay', type=Path,
                   help='engineering candidate worker only: replay a recorded native proposal chunk packet')
    p.add_argument('--action-chunk-mode', choices=('open_loop24', 'receding8', 'receding1', 'overlap8', 'temporal1'), default='open_loop24',
                   help='action-chunk behavior worker replanning schedule')
    p.add_argument('--action-chunk-roles', choices=('behavior', 'candidate'), default='behavior',
                   help='engineering worker role layout: behavior parity or ACT candidate/zero calibration')
    p.add_argument('--score-inputs', type=Path, nargs='+')
    p.add_argument('--score-output', type=Path)
    a = p.parse_args()
    try:
        a.native_backend = resolve_legacy_backend(a.backend, a.physics_device)
    except ValueError as error:
        p.error(str(error))
    for name in ('run_dir', 'reference', 'encoder', 'output'):
        setattr(a, name, getattr(a, name).resolve())
        if not is_within(getattr(a, name), ROOT / 'outputs/consequence-evaluator'):
            p.error('all inputs/outputs must be task-owned')
    for name in ('worker_output', 'prefix', 'score_output'):
        value = getattr(a, name, None)
        if value is not None:
            value = value.resolve()
            if not is_within(value, ROOT / 'outputs/consequence-evaluator'):
                p.error('all optional outputs/inputs must be task-owned')
            setattr(a, name, value)
    for name in ('action_chunk_checkpoint', 'action_chunk_replay', 'retargeter_checkpoint', 'retargeter_source', 'object_relative_source', 'object_relative_preload', 'object_relative_inverse', 'object_relative_pd_inverse'):
        value = getattr(a, name, None)
        if value is not None:
            value = value.resolve()
            if (not is_within(value, ROOT / 'outputs/consequence-evaluator') or not value.exists()):
                p.error('%s must be an existing task-owned output' % name.replace('_', '-'))
            setattr(a, name, value)
    if a.object_relative_layout in ('finger_preload','finger_preload_late','geometry_pd','measured_pd_load_memory') and (
            a.object_relative_source is None or a.object_relative_preload is None):
        p.error('preload diagnostics require oracle source and frozen preload statistics')
    if a.object_relative_layout in ('geometry_inverse_late','geometry_inverse_relative_late','geometry_inverse_repeat','geometry_load_delta_late','geometry_pd_inverse','measured_pd_load_memory') and (
            a.object_relative_source is None or a.object_relative_inverse is None):
        p.error('geometry inverse diagnostic requires source and inverse artifact')
    if a.object_relative_layout in ('geometry_pd_inverse','measured_pd_load_memory') and a.object_relative_pd_inverse is None:
        p.error('cold geometry dynamics requires train-only PD calibration')
    if a.object_relative_source is not None:
        if (not a.worker or not a.finish or a.engineering_group_envs != 4
                or a.engineering_steps != 542 or a.action_chunk_checkpoint is not None
                or a.action_chunk_replay is not None or a.retargeter_checkpoint is not None
                or a.retargeter_source is not None or a.prefix is not None
                or a.native_backend.name != 'gpu_physx_gpu_pipeline'):
            p.error('object-relative oracle requires native GPU/four roles/full542 and excludes ACT/model/prefix')
    if a.retargeter_checkpoint is not None or a.retargeter_source is not None:
        if (a.retargeter_checkpoint is None or a.retargeter_source is None
                or not a.worker or not a.finish or a.engineering_group_envs != 4
                or a.engineering_steps != 542 or a.action_chunk_checkpoint is not None
                or a.action_chunk_replay is not None or a.prefix is not None
                or a.native_backend.name != 'gpu_physx_gpu_pipeline'):
            p.error('GT retarget requires checkpoint+source, native GPU worker, four roles/full542, no ACT/prefix')
    if not 0 <= a.gpu <= 7 or not 1 <= a.episodes <= 4 or not 30 <= a.seconds <= 900:
        p.error('bounded one-GPU/<=4episode/<=900s Probe required')
    if a.engineering_steps and (not a.worker or not a.finish or not 24 <= a.engineering_steps <= 542):
        p.error('engineering step cap requires native baseline worker and24..542steps')
    if a.engineering_group_envs and (not a.worker or not a.finish or not 4 <= a.engineering_group_envs <= 96):
        p.error('engineering group requires native baseline worker and4..96 environments')
    if a.engineering_group_envs and a.engineering_actor_copies < 1:
        p.error('engineering group actor copies must be positive')
    if a.engineering_group_envs and a.engineering_group_envs * a.engineering_actor_copies != 256:
        p.error('engineering group must keep the fixed 256-row actor contract')
    if a.engineering_env_spacing is not None and (
            not a.engineering_group_envs or not np.isfinite(a.engineering_env_spacing)
            or a.engineering_env_spacing <= 0 or a.engineering_env_spacing > 100):
        p.error('engineering env-spacing requires a positive finite synchronous group override <=100')
    if a.worker and (a.query_tick < 0 or a.query_tick > 542 - K):
        p.error('worker query tick must leave a complete 24-step horizon')
    if not a.worker and not a.score_inputs and a.query_tick != 0:
        p.error('--query-tick is restricted to native engineering workers')
    if (a.engineering_env_spacing is not None and a.native_backend.name != 'gpu_physx_gpu_pipeline'):
        p.error('engineering env-spacing is only defined for native GPU PhysX/GPU pipeline')
    if a.engineering_serial_replay and (not a.worker or not a.finish or a.engineering_group_envs):
        p.error('serial replay requires a native worker without a synchronous group')
    if a.engineering_serial_replay and (a.engineering_steps < 72 or a.engineering_steps > 542):
        p.error('serial replay requires query tick48 plus24 steps and <=542 steps')
    if a.engineering_serial_replay and a.query_tick not in (0, 48):
        p.error('serial replay requires the established query tick48 contract')
    if a.engineering_serial_cluster and (a.engineering_serial_replay or not a.worker or not a.finish
                                        or a.engineering_group_envs):
        p.error('serial cluster requires a native worker without group or serial replay mode')
    if a.engineering_serial_cluster and (a.engineering_steps < 72 or a.engineering_steps > 542):
        p.error('serial cluster requires query tick48 plus24 steps and <=542 steps')
    if a.engineering_serial_cluster and a.engineering_steps != 72:
        p.error('serial cluster is fixed to the 72-step short-window contract')
    if a.engineering_serial_cluster and a.query_tick not in (0, 48):
        p.error('serial cluster requires the established query tick48 contract')
    if a.engineering_group_envs and not a.engineering_steps:
        p.error('engineering group requires an explicit bounded --engineering-steps value')
    if a.engineering_group_envs:
        group_query = int(a.query_tick or 48)
        if group_query < 0 or group_query + K > a.engineering_steps:
            p.error('engineering group requires --engineering-steps after query tick plus24 candidate steps')
        if group_query > 542 - K:
            p.error('engineering group query tick must leave a complete 24-step horizon')
    if len(a.zero_env_pair) != 2 or a.zero_env_pair[0] == a.zero_env_pair[1]:
        p.error('--zero-env-pair requires two distinct environment indices')
    if len(a.candidate_env_pair) != 2 or a.candidate_env_pair[0] == a.candidate_env_pair[1]:
        p.error('--candidate-env-pair requires two distinct environment indices')
    if a.engineering_group_envs and any(index < 0 or index >= a.engineering_group_envs
                                        for index in a.zero_env_pair):
        p.error('--zero-env-pair indices must be inside --engineering-group-envs')
    if a.engineering_group_envs and any(index < 0 or index >= a.engineering_group_envs
                                        for index in a.candidate_env_pair):
        p.error('--candidate-env-pair indices must be inside --engineering-group-envs')
    if a.engineering_group_envs and set(a.zero_env_pair) & set(a.candidate_env_pair):
        p.error('--zero-env-pair and --candidate-env-pair must be disjoint')
    if a.diagnostic_force_cache and not a.worker:
        p.error('force-cache diagnostic is only available to a native engineering worker')
    if a.engineering_single_act and (
            not a.worker or not a.finish or a.engineering_group_envs
            or a.engineering_serial_replay or a.engineering_serial_cluster):
        p.error('single ACT behavior requires a finished native worker without group or serial mode')
    if a.engineering_single_act and a.native_backend.name != 'gpu_physx_gpu_pipeline':
        p.error('single ACT behavior requires --backend gpu_physx_gpu_pipeline')
    if a.engineering_single_act and a.engineering_steps != 72:
        p.error('single ACT behavior requires exactly --engineering-steps 72')
    if a.engineering_single_act and a.query_tick not in (0, 48):
        p.error('single ACT behavior requires the established query tick48 contract')
    if a.engineering_single_act and a.prefix is not None:
        p.error('single ACT behavior starts from a fresh prefix and does not accept --prefix')
    if a.engineering_single_act and a.action_chunk_checkpoint is None:
        p.error('single ACT behavior requires --action-chunk-checkpoint')
    if a.engineering_single_act and a.action_chunk_replay is not None:
        p.error('single ACT behavior does not accept --action-chunk-replay')
    if a.engineering_single_act and a.action_chunk_mode != 'open_loop24':
        p.error('single ACT behavior requires --action-chunk-mode open_loop24')
    if a.engineering_single_act and a.action_chunk_roles != 'behavior':
        p.error('single ACT behavior requires --action-chunk-roles behavior')
    if a.engineering_single_act and a.engineering_actor_copies != 64:
        p.error('single ACT behavior requires the verified 64-row actor inference contract')
    if a.engineering_single_act and a.diagnostic_force_cache:
        p.error('single ACT behavior does not accept force-cache diagnostics')
    if (a.action_chunk_checkpoint is not None or a.action_chunk_replay is not None) and (
            not a.worker or not a.finish or not (a.engineering_group_envs or a.engineering_single_act)
            or a.engineering_steps < 24):
        p.error('action-chunk behavior requires a bounded native engineering worker')
    if a.action_chunk_checkpoint is not None and a.action_chunk_replay is not None:
        p.error('choose one action-chunk checkpoint or recorded replay packet')
    if a.action_chunk_roles == 'candidate' and a.action_chunk_checkpoint is None and a.action_chunk_replay is None:
        p.error('action-chunk candidate roles require a checkpoint or recorded replay packet')
    if a.action_chunk_roles == 'candidate' and a.action_chunk_mode != 'open_loop24':
        p.error('action-chunk candidate calibration currently requires --action-chunk-mode open_loop24')
    if a.action_chunk_replay is not None and a.action_chunk_roles != 'candidate':
        p.error('recorded action-chunk replay is only valid for candidate roles')
    if a.action_chunk_roles == 'candidate' and a.engineering_steps < 72:
        p.error('action-chunk candidate calibration requires query tick48 plus24 steps')
    if a.action_chunk_roles == 'candidate' and a.engineering_steps != 72:
        p.error('action-chunk candidate calibration is currently bounded to exactly 72 steps')
    if (a.action_chunk_checkpoint is not None or a.action_chunk_replay is not None) and a.query_tick not in (0, 48):
        p.error('action-chunk group probes require the established query tick48 contract')
    if (a.action_chunk_checkpoint is not None or a.action_chunk_replay is not None) and (
            a.action_chunk_roles == 'behavior' and not a.engineering_single_act
            and a.engineering_group_envs != 4):
        p.error('action-chunk behavior roles require exactly four group environments')
    if a.action_chunk_roles == 'candidate' and a.engineering_group_envs < 4:
        p.error('action-chunk candidate roles require at least four group environments')
    if a.worker:
        native_worker(a); return
    if a.score_inputs:
        score_worker(a); return
    if a.output.exists() or shutil.disk_usage(ROOT).free < 20 * 2**30:
        p.error('fresh owned output and20GiB reserve required')
    ensure_free(a.gpu); frozen = self_trained_ancestry(a.run_dir, ROOT / 'outputs/consequence-evaluator')
    config = json.loads((a.run_dir / 'config.json').read_text())
    native_root = ROOT / 'third_party/DExplore/dexplore'
    files = [Path(__file__).resolve(), TASK / 'docs/experiments/probes/P-20261008-gate1-gt-progress.md',
        a.run_dir / 'config.json', Path(config['input_manifest']),
        Path(config['cfg_env']), TASK / 'tools/audit/probe_official_generator.py',
        TASK / 'tools/audit/native_twin_probe.py',
        ROOT / 'src/task/CmResidual/tools/dexplore_ddp_rank_bootstrap.py',
        ROOT / 'src/task/cm-interaction-oracle/src/oracle_y_utility.py',
        *sorted((TASK / 'src/consequence_evaluator').glob('*.py')),
        *sorted(native_root.rglob('*.py')),
        *sorted(p for p in (native_root / 'data/assets').rglob('*') if p.is_file()),
        *sorted(Path(config['motion_root']).glob('*/interaction_hand_inspire.pt'))]
    for directory in (a.reference, a.encoder):
        meta = json.loads((directory / 'manifest.json').read_text())
        frozen.update(meta['sources']); files.append(directory / 'manifest.json')
        files.append(Path(meta['checkpoint']) if directory == a.encoder else directory / 'reference.npz')
    frozen.update({str(f): sha(f) for f in files})
    if any(sha(k) != v for k, v in frozen.items()):
        raise ValueError('Gate1 input drift')
    a.output.mkdir(parents=True); started = time.monotonic(); pairs = []; events = []
    manifest = dict(schema='ref2dex.consequence-gate1.gt-progress.v1', status='RUNNING',
        experiment_id='P-20261008-gate1-gt-progress', run_id=a.output.name, pid=os.getpid(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        sources=frozen, physical_gpu=a.gpu, physics_device=a.native_backend.sim_device,
        backend=a.native_backend.as_dict(), seconds_budget=a.seconds, training_allowed=False,
        seeds=list(SEEDS[:a.episodes]), candidates=list(CANDIDATES), query_ticks=list(QUERY_TICKS),
        horizon=K, execution_horizon=K_EXEC, scope='four early-contact rolling decisions then full actor continuation',
        actor_role='owned self-trained Cm-off actor, not official reference generator',
        solver='fresh_simulator_complete_executed_prefix_replay; no mid-state restore',
        outcome_metric='recoverable_hold45_controlled_place_settle15', events=events, pairs=pairs)
    def check():
        if time.monotonic() - started >= a.seconds:
            raise TimeoutError('Gate1 campaign deadline')
        if shutil.disk_usage(ROOT).free < 20 * 2**30 or sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file()) > 2**30:
            raise RuntimeError('Gate1 disk budget exceeded')
        if any(sha(k) != v for k, v in frozen.items()):
            raise ValueError('Gate1 source/input drift during execution')
    common = ['--run-dir', str(a.run_dir), '--reference', str(a.reference), '--encoder', str(a.encoder),
              '--output', str(a.output), '--gpu', str(a.gpu), '--backend', a.native_backend.name]
    def launch(name, args, native=True):
        check(); command = [NATIVE_PYTHON if native else sys.executable, str(Path(__file__).resolve()), *common, *args]
        env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',
            TMPDIR=str(ROOT / 'tmp'), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
        if native: env['PYTHONNOUSERSITE'] = '1'
        else: env.pop('PYTHONNOUSERSITE', None)
        with (a.output / (name + '.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=min(160, max(1, a.seconds - (time.monotonic() - started))), check=True)
        check()
        events.append(dict(worker=name, elapsed_s=time.monotonic() - started))
        manifest['elapsed_s'] = time.monotonic() - started
        write(a.output / 'manifest.json', manifest)
        print(json.dumps(events[-1]), flush=True)
    write(a.output / 'manifest.json', manifest)
    try:
        for seed in SEEDS[:a.episodes]:
            baseline_path = a.output / ('s%d-baseline.pkl' % seed)
            launch('s%d-baseline' % seed, ['--worker', '--worker-output', str(baseline_path), '--seed', str(seed), '--finish'])
            baseline = load(baseline_path); prefix_path = baseline_path; decisions = []
            for tick in QUERY_TICKS:
                paths = []
                for candidate in range(3):
                    name = 's%d-t%d-c%d' % (seed, tick, candidate); path = a.output / (name + '.pkl')
                    launch(name, ['--worker', '--worker-output', str(path), '--seed', str(seed),
                        '--prefix', str(prefix_path), '--query-tick', str(tick), '--candidate', str(candidate)])
                    paths.append(path)
                packets = [load(path) for path in paths]
                # First zero branch must reproduce the separately run baseline.
                if tick == QUERY_TICKS[0]:
                    if packets[0]['state_hashes'] != baseline['state_hashes'][:tick + K + 1]:
                        raise ValueError('candidate-zero continuation differs from frozen baseline')
                # Repeated zero branch at the first query protects hidden solver state.
                if seed == SEEDS[0] and tick == QUERY_TICKS[0]:
                    repeat = a.output / 'first-zero-repeat.pkl'
                    launch('first-zero-repeat', ['--worker', '--worker-output', str(repeat), '--seed', str(seed),
                        '--prefix', str(prefix_path), '--query-tick', str(tick)])
                    repeated = load(repeat)
                    if fingerprint(repeated) != fingerprint(packets[0]):
                        raise ValueError('first candidate-zero repeat is not exact')
                score = a.output / ('s%d-t%d-score.json' % (seed, tick))
                launch('s%d-t%d-score' % (seed, tick), ['--score-inputs', *map(str, paths), '--score-output', str(score)], native=False)
                decision = json.loads(score.read_text()); decisions.append(dict(tick=tick, **decision))
                # Only controls from the chosen8step prefix are committed. The
                # NEXT fresh native worker executes this full mixed prefix and
                # verifies every native/RNG/measured state, never joins suffix Z.
                chosen = packets[decision['choice']]; stop = tick + K_EXEC
                committed = {}
                for key, value in chosen.items():
                    if key in ('actions', 'done'): committed[key] = value[:stop].copy()
                    elif key in ('state_hashes', 'state_field_hashes', 'rng_hashes'): committed[key] = value[:stop + 1]
                    elif key in ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap',
                                 'table_footprint', 'object_velocity', 'history', 'native_contact_forces', 'native_object_contact_forces', 'timestamps'):
                        committed[key] = value[:stop + 1].copy()
                    else: committed[key] = value
                prefix_path = a.output / ('s%d-committed-through%d.pkl' % (seed, stop)); save(prefix_path, committed)
            terminal = a.output / ('s%d-rolling.pkl' % seed)
            launch('s%d-rolling' % seed, ['--worker', '--worker-output', str(terminal), '--seed', str(seed),
                '--prefix', str(prefix_path), '--query-tick', str(QUERY_TICKS[-1] + K_EXEC), '--finish'])
            actual = load(terminal)
            pairs.append(dict(seed=seed, baseline=episode_outcome(baseline), rolling=episode_outcome(actual),
                initial_state_hash=baseline['state_hashes'][0], decisions=decisions,
                every_committed_prefix_reexecuted_and_exact=True))
            print(json.dumps(dict(paired=paired_counts(pairs))), flush=True)
        result = paired_counts(pairs)
        unique = len({p['initial_state_hash'] for p in pairs})
        # This panel is a bounded Probe; no formal Gate1 claim from four episodes.
        status = 'PROMISING' if result['rescued'] > result['harmed'] else 'UNCLEAR'
        result.update(probe_status=status, unique_initial_native_states=unique, Gate1_formal_pass=False,
            scope=manifest['scope'], training_allowed=False, pairs=pairs,
            limitation='four early-contact decisions, few deterministic seeds; no whole-episode MPC or formal validation')
        write(a.output / 'result.json', result); manifest.update(status='COMPLETED', result=result)
    except BaseException as error:
        manifest.update(status='FAILED', error=repr(error)); raise
    finally:
        manifest['elapsed_s'] = time.monotonic() - started; write(a.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
