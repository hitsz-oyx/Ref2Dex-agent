"""Isaac-free state, trace and repeatability contracts for frozen actors."""
from __future__ import annotations

import hashlib
import math
import random

import numpy as np
import torch

SCHEMA = 'ref2dex.paired_evaluation.v1'


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    return value


def fingerprint(value):
    """Content hash independent of torch.save storage IDs and aliases."""
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode())
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str((t.dtype, tuple(t.shape))).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.dtype, v.shape)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def restore_rng(saved):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch_cpu'].cpu())
    if saved['torch_cuda']:
        if len(saved['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(saved['torch_cuda'])


def device_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.clone().to(device)
    if isinstance(value, tuple):
        return tuple(device_tree(v, device) for v in value)
    if isinstance(value, list):
        return [device_tree(v, device) for v in value]
    return value


def capture_initial(task, player, observation, physics_properties):
    """Capture all direct task tensors, controller buffers, RNN and RNG.

    PhysX exposes no serialization of solver caches. Require a never-stepped
    fresh simulator, where solver history is empty, for every arm. Refuse
    warm-state restores rather than treating root/DOF tensors as full PhysX.
    """
    if task.gym.get_frame_count(task.sim) != 0:
        raise ValueError('initial state requires a fresh unstepped simulator')
    if task.dr_randomizations or task.projtype != 'None' or task._motion_sampler is not None:
        raise ValueError('unsupported randomized/projectile/motion-sampler state')
    tensors = {k: cpu_copy(v) for k, v in vars(task).items() if isinstance(v, torch.Tensor)}
    required = {'_root_states', '_dof_state', '_rigid_body_state', '_contact_forces',
                '_tar_contact_forces', 'obs_buf', 'progress_buf', 'data_id', 'start_times',
                '_curr_obs', '_hist_obs', 'contact_reset', '_terminate_buf'}
    if not required.issubset(tensors):
        raise ValueError('missing native state tensors: ' + str(required - set(tensors)))
    scalars = {k: v for k, v in vars(task).items() if isinstance(v, (str, bool, int, float))}
    return dict(schema=SCHEMA, tensors=tensors, scalars=scalars,
                reset_default_ids=cpu_copy(task._reset_default_env_ids),
                reset_ref_ids=cpu_copy(task._reset_ref_env_ids),
                rnn=cpu_copy(player.states), is_rnn=bool(player.is_rnn),
                observation=cpu_copy(observation), rng=capture_rng(),
                physics_properties=physics_properties, frame_count=0,
                solver_contract='fresh process/simulator; no preceding simulate; solver cache empty')


def restore_initial(task, player, saved, unwrap, physics_properties):
    if saved['schema'] != SCHEMA or task.gym.get_frame_count(task.sim) != 0:
        raise ValueError('cannot restore a warm simulator')
    if fingerprint(physics_properties) != fingerprint(saved['physics_properties']):
        raise ValueError('physical actor property mismatch')
    tensors = {k: v for k, v in vars(task).items() if isinstance(v, torch.Tensor)}
    if set(tensors) != set(saved['tensors']):
        raise ValueError('native task tensor inventory changed')
    for name, source in saved['tensors'].items():
        current = tensors[name]
        if current.shape != source.shape or current.dtype != source.dtype:
            raise ValueError('state tensor shape/dtype mismatch: ' + name)
        current.copy_(source.to(current.device))
    for name, value in saved['scalars'].items():
        if name in ('device', 'device_id', 'device_type', 'graphics_device_id'):
            if getattr(task, name) != value:
                raise ValueError('device mapping changed')
        else:
            setattr(task, name, value)
    # Restore every actor including tables, and all DOF position/velocities.
    task.gym.set_actor_root_state_tensor(task.sim, unwrap(task._root_states))
    task.gym.set_dof_state_tensor(task.sim, unwrap(task._dof_state))
    task._refresh_sim_tensors()
    # Derived native buffers are reused at cold time0; validate root/DOF from
    # the engine before restoring observation/history/contact caches.
    for name in ('_root_states', '_dof_state'):
        if fingerprint(getattr(task, name)) != fingerprint(saved['tensors'][name]):
            raise ValueError('engine state restore mismatch: ' + name)
    for name, source in saved['tensors'].items():
        getattr(task, name).copy_(source.to(getattr(task, name).device))
    task._reset_default_env_ids = device_tree(saved['reset_default_ids'], task.device)
    task._reset_ref_env_ids = device_tree(saved['reset_ref_ids'], task.device)
    if bool(player.is_rnn) != saved['is_rnn']:
        raise ValueError('RNN architecture mismatch')
    player.states = device_tree(saved['rnn'], player.device)
    restore_rng(saved['rng'])
    observation = device_tree(saved['observation'], player.device)
    actual = capture_initial(task, player, observation, physics_properties)
    if fingerprint(actual) != fingerprint(saved):
        raise ValueError('full cold state verification failed')
    return observation


def wilson_upper(count, total, z=1.959963984540054):
    if total <= 0:
        raise ValueError('empty noise panel')
    p = count / total
    return (p + z*z/(2*total) + z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def repeatability_gate(first, second, *, trace_contract_valid, closed_loop_equivalent):
    """Conservative per-episode noise bound, independent of net cancellation."""
    a = sorted(first, key=lambda r: r['pair_id'])
    b = sorted(second, key=lambda r: r['pair_id'])
    if not a or len(a) != len(b) or len({r['pair_id'] for r in a}) != len(a):
        raise ValueError('incomplete or duplicate paired episodes')
    identity = ('pair_id', 'motion_id', 'start_frame', 'steps', 'initial_object_height', 'control_dt')
    if any(tuple(x[k] for k in identity) != tuple(y[k] for k in identity) for x, y in zip(a, b)):
        raise ValueError('episode pairing mismatch')
    n = len(a)
    disagreements = sum(x['stable_success'] != y['stable_success'] for x,y in zip(a,b))
    drop_changes = sum(x['drop_after_success'] != y['drop_after_success'] for x,y in zip(a,b))
    count_a = sum(x['stable_success'] for x in a)
    count_b = sum(x['stable_success'] for x in b)
    delta = abs(count_a-count_b) / n
    upper = wilson_upper(disagreements, n)
    drop_upper = wilson_upper(drop_changes, n)
    passed = bool(trace_contract_valid and closed_loop_equivalent and delta < .025 and upper < .05 and drop_upper < .05)
    return dict(passed=passed, episodes=n, success_counts=[count_a,count_b],
                absolute_rate_difference=delta, success_label_disagreements=disagreements,
                success_discordance_upper95=upper, drop_label_disagreements=drop_changes,
                drop_discordance_upper95=drop_upper, trace_contract_valid=trace_contract_valid,
                closed_loop_equivalent=closed_loop_equivalent,
                boundary='repeatability noise screen, not statistical power or Cm validation')
