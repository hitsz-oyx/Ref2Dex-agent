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
from consequence_evaluator.twin import REQUIRED_NATIVE_STATE_KEYS, capture_native_rng, fingerprint
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
    expected = load(a.prefix) if a.prefix else None
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
        if player.is_rnn or task.num_envs != count or abs(task.dt - 1 / 30) > 1e-8 \
                or abs(task.sim_params.dt - 1 / 60) > 1e-8 or task.control_freq_inv != 2:
            raise ValueError('fixed nonrecurrent synchronous group contract required')
        if int(task.num_motions) != 1:
            raise ValueError('engineering group requires one repeated motion across all env roles')
        task._state_init = DexploreTask.StateInit.Start; task._hybrid_init_prob = 1.
        task._adaptive_kappa_enabled = False; task._enable_early_termination = False
        if task.dr_randomizations or task._motion_sampler is not None or task.projtype != 'None':
            raise ValueError('randomized/sampler/projectile replay unsupported')
        geometry = PhysicalGeometry(task, native_root / 'data/assets')
        table = TableSupport(native_root / 'data/assets', task.device)
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
        state_keys = sorted(REQUIRED_NATIVE_STATE_KEYS | {k for k in ('reset_buf', 'actions', 'real_pd_tar')
                                                           if isinstance(getattr(task, k, None), torch.Tensor)})
        fields = ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap', 'table_footprint',
                  'dof_position', 'dof_velocity', 'object_velocity', 'history',
                  'native_contact_forces', 'native_object_contact_forces')
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
        query = 48; horizon = 24
        positive = torch.as_tensor(candidate_plan(1), device=player.device)
        negative = torch.as_tensor(candidate_plan(2), device=player.device)
        stop = a.engineering_steps or query + horizon
        if stop > total:
            raise ValueError('group probe exceeds frozen reference')
        for tick in range(stop):
            if time.monotonic() - started > 150:
                raise TimeoutError('bounded synchronous group deadline')
            actor_actions = legacy_group_actor_action(player, obs).clamp(-1, 1)
            # Keep baseline, zero-repeat, and both candidate arms on the same
            # control stream through the query. This removes tiny row-wise GEMM
            # differences and makes the group test an execution-contract probe.
            base_action = actor_actions[:1].expand(count, -1).clone()
            control = base_action.clone()
            if query <= tick < query + horizon:
                offset = tick - query
                control[2] = (control[2] + positive[offset]).clamp(-1, 1)
                control[3] = (control[3] + negative[offset]).clamp(-1, 1)
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
                if not np.array_equal(value[tick][0], value[tick][1]):
                    mismatches.append(dict(field=key, tick=tick)); break
        if mismatches:
            diagnostics=[]
            for mismatch in mismatches[:12]:
                key, tick = mismatch['field'], mismatch['tick']
                value = packets[key]
                left, right = value[tick][0], value[tick][1]
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
        baseline_packet={key: value[:,0] for key,value in packets.items() if key in fields}
        baseline_packet['timestamps']=packets['timestamps']
        outcome=episode_outcome(baseline_packet) if stop >= 45 else None
        calibration_fields = ('object_pose', 'hand_keypoints', 'dof_position', 'dof_velocity',
                              'object_velocity', 'history', 'native_contact_forces',
                              'native_object_contact_forces')
        zero_noise = {}; candidate_effect = {}
        # State packets have one more sample than the executed-control packet:
        # include the state at t+24 when a 72-step bounded run ends there.
        effect_stop = min(len(packets['object_pose']), query + horizon + 1)
        for key in calibration_fields:
            value = np.asarray(packets[key], dtype=np.float64)
            zero = value[:query + 1, 1] - value[:query + 1, 0]
            zero_flat = np.abs(zero).reshape(-1)
            zero_max = float(zero_flat.max()) if zero_flat.size else 0.0
            zero_p95 = float(np.quantile(zero_flat, .95)) if zero_flat.size else 0.0
            zero_post = value[query + 1:effect_stop, 1] - value[query + 1:effect_stop, 0]
            zero_displacement = ((value[query + 1:effect_stop, 1] - value[query, 1])
                                 - (value[query + 1:effect_stop, 0] - value[query, 0]))
            zero_post_flat = np.abs(zero_post).reshape(-1)
            zero_disp_flat = np.abs(zero_displacement).reshape(-1)
            zero_noise[key] = dict(
                prequery_max_abs=zero_max, prequery_p95_abs=zero_p95,
                postquery_max_abs=float(zero_post_flat.max()) if zero_post_flat.size else 0.0,
                postquery_p95_abs=float(np.quantile(zero_post_flat, .95)) if zero_post_flat.size else 0.0,
                postquery_displacement_max_abs=float(zero_disp_flat.max()) if zero_disp_flat.size else 0.0,
                postquery_displacement_p95_abs=float(np.quantile(zero_disp_flat, .95)) if zero_disp_flat.size else 0.0)
            candidate_effect[key] = {}
            for env_index, name in ((2, 'positive'), (3, 'negative')):
                query_offset = value[query, env_index] - value[query, 0]
                raw = value[query + 1:effect_stop, env_index] - value[query + 1:effect_stop, 0]
                incremental = ((value[query + 1:effect_stop, env_index] - value[query, env_index])
                               - (value[query + 1:effect_stop, 0] - value[query, 0]))
                effect_vs_zero = incremental - zero_displacement
                raw_flat = np.abs(raw).reshape(-1); inc_flat = np.abs(incremental).reshape(-1)
                effect_flat = np.abs(effect_vs_zero).reshape(-1)
                inc_max = float(inc_flat.max()) if inc_flat.size else 0.0
                candidate_effect[key][name] = dict(
                    raw_max_abs=float(raw_flat.max()) if raw_flat.size else 0.0,
                    incremental_max_abs=inc_max,
                    incremental_p95_abs=float(np.quantile(inc_flat, .95)) if inc_flat.size else 0.0,
                    effect_vs_zero_max_abs=float(effect_flat.max()) if effect_flat.size else 0.0,
                    effect_vs_zero_p95_abs=float(np.quantile(effect_flat, .95)) if effect_flat.size else 0.0,
                    effect_vs_zero_to_prequery_max_ratio=(float(effect_flat.max()) / zero_max if zero_max > 0 else None),
                    effect_vs_zero_to_prequery_p95_ratio=(float(np.quantile(effect_flat, .95)) / zero_p95 if zero_p95 > 0 else None),
                    effect_vs_zero_to_postquery_displacement_max_ratio=(float(effect_flat.max()) / zero_noise[key]['postquery_displacement_max_abs']
                        if zero_noise[key]['postquery_displacement_max_abs'] > 0 else None),
                    effect_vs_zero_to_postquery_displacement_p95_ratio=(float(np.quantile(effect_flat, .95)) / zero_noise[key]['postquery_displacement_p95_abs']
                        if zero_noise[key]['postquery_displacement_p95_abs'] > 0 else None),
                    query_offset_max_abs=float(np.abs(query_offset).max()) if np.size(query_offset) else 0.0)
        action_calibration = {}
        for env_index, name in ((2, 'positive'), (3, 'negative')):
            delta = packets['actions'][query:effect_stop, env_index].astype(np.float64) - packets['actions'][query:effect_stop, 0]
            delta_abs = np.abs(delta).reshape(-1)
            step_delta = np.abs(delta).max(axis=-1)
            action_calibration[name] = dict(max_abs=float(delta_abs.max()), p95_abs=float(np.quantile(delta_abs, .95)),
                                            nonzero_steps=int(np.count_nonzero(step_delta > 1e-7)),
                                            nonzero_elements=int(np.count_nonzero(delta_abs > 1e-7)))
        packet=dict(**packets, seed=a.seed, query_tick=query, candidate_roles=['baseline','zero_repeat','positive','negative'],
            candidate_plan_semantics='baseline/zero/positive/negative; env0 actor control is broadcast to all envs, then positive/negative fork at tick48 for24 steps',
            group_mode='synchronous_same_process_noise_calibration',
            replay_identity=identity, canonical_state_keys=state_keys, actor_inference_batch=count * 64,
            engineering_only=True, group_envs=count, group_prefix_exact=(not mismatches),
            control_prefix_exact=bool(np.array_equal(packets['actions'][:,0], packets['actions'][:,1])
                                      and np.array_equal(packets['done'][:,0], packets['done'][:,1])),
            exact_fields=exact_keys,
            rng_semantics='one process-global RNG trace shared by all envs; zero-pair calibration is state/observation based',
            calibration_window=dict(zero_prequery_ticks=[0, query], candidate_postquery_ticks=[query + 1, effect_stop - 1]),
            zero_noise_calibration=zero_noise, candidate_effect_calibration=candidate_effect,
            action_calibration=action_calibration,
            baseline_outcome=outcome, baseline_max_lift_m=float(np.max(packets['object_pose'][:,0,2,3] - packets['object_pose'][0,0,2,3])),
            zero_repeat_max_lift_m=float(np.max(packets['object_pose'][:,1,2,3] - packets['object_pose'][0,1,2,3])),
            initial_semantic_gap=initial_semantic_gap,
            initial_semantic_exact=initial_semantic_exact,
            env_origins=origin_np.tolist(),
            source_backend=actual_backend)
        save(a.worker_output, packet)
        write(a.worker_output.with_suffix('.json'), dict(status='COMPLETED', steps=stop, group_envs=count,
            baseline_zero_exact=not mismatches, mismatches=mismatches, engineering_only=True,
            elapsed_s=time.monotonic() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(), outcome=outcome,
            baseline_max_lift_m=packet['baseline_max_lift_m'], zero_noise_calibration=zero_noise,
            candidate_effect_calibration=candidate_effect, action_calibration=action_calibration,
            initial_semantic_gap=initial_semantic_gap, initial_semantic_exact=initial_semantic_exact))

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
    sys.argv = [sys.argv[0], '--task', 'Dexplore_Inspire', '--cfg_env', config['cfg_env'],
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
    p.add_argument('--engineering-group-envs', type=int, default=0,
                   help='native worker only: synchronous same-process group (baseline/zero/+/-), never GT scoring')
    p.add_argument('--diagnostic-force-cache', action='store_true', help='engineering only: inspect derived force differences; all other prefix fields remain strict')
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
    if not 0 <= a.gpu <= 7 or not 1 <= a.episodes <= 4 or not 30 <= a.seconds <= 900:
        p.error('bounded one-GPU/<=4episode/<=900s Probe required')
    if a.engineering_steps and (not a.worker or not a.finish or not 24 <= a.engineering_steps <= 542):
        p.error('engineering step cap requires native baseline worker and24..542steps')
    if a.engineering_group_envs and (not a.worker or not a.finish or not 4 <= a.engineering_group_envs <= 96):
        p.error('engineering group requires native baseline worker and4..96 environments')
    if a.engineering_group_envs and not a.engineering_steps:
        p.error('engineering group requires an explicit bounded --engineering-steps value')
    if a.engineering_group_envs and a.engineering_steps < 72:
        p.error('engineering group requires at least query tick48 plus24 candidate steps')
    if a.diagnostic_force_cache and not a.worker:
        p.error('force-cache diagnostic is only available to a native engineering worker')
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
