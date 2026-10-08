"""Contracts for exact current-state twin branches.

This module is deliberately Isaac-free.  A native collector may use it after
creating a *fresh* simulator and replaying the same prefix.  Copying only an
object pose or hand keypoints is rejected because controller history, DOF
velocities, contact buffers, and RNG state can change the next 24 steps.
"""
from __future__ import annotations

import copy
import hashlib
import random
import struct
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .contracts import K

SCHEMA = 'ref2dex.consequence-evaluator.twin.v1'
BRANCHES = ('a', 'b')
MIN_PLAN_L2 = .01
MIN_REALIZED_ACTION_L2 = .01
SOLVER_CONTRACT = 'fresh_simulator_prefix_replay'
RESIDUAL_PLAN_SEMANTICS = 'decision_known_requested_residual_plan'
EXECUTED_ACTION_SEMANTICS = 'native_post_noise_pre_physics_control'
NATIVE_CONTROL_DT = 1 / 30
NATIVE_PHYSICS_DT = 1 / 60
NATIVE_SIM_STEPS_PER_CONTROL = 2

# This is the minimum inventory captured by the existing native cold-start
# contract in ``CmResidual/paired_evaluation.py``.  A caller may add task
# specific tensors, but may not replace this inventory with pose-only state.
REQUIRED_NATIVE_STATE_KEYS = frozenset({
    '_root_states', '_dof_state', '_rigid_body_state', '_contact_forces',
    '_tar_contact_forces', 'obs_buf', 'progress_buf', 'data_id', 'start_times',
    '_curr_obs', '_hist_obs', 'contact_reset', '_terminate_buf',
})
REQUIRED_STATE_META_KEYS = frozenset({'controller_state', 'rnn_state', 'observation',
                                      'scalars', 'reset_ids'})
REQUIRED_RNG_KEYS = frozenset({'python', 'numpy', 'torch_cpu', 'torch_cuda'})


def _valid_rng_state(value):
    """Validate the concrete RNG state formats emitted by native capture."""
    if not isinstance(value, dict) or not REQUIRED_RNG_KEYS.issubset(value):
        return False
    try:
        python_rng = random.Random()
        python_rng.setstate(value['python'])
    except (TypeError, ValueError, RuntimeError, OverflowError, IndexError):
        return False
    try:
        numpy_rng = np.random.RandomState()
        numpy_rng.set_state(value['numpy'])
    except (TypeError, ValueError, RuntimeError, OverflowError, IndexError):
        return False

    torch_cpu = value['torch_cpu']
    if (not isinstance(torch_cpu, np.ndarray) or torch_cpu.ndim != 1
            or torch_cpu.size == 0 or torch_cpu.dtype != np.dtype('uint8')):
        return False
    torch_cuda = value['torch_cuda']
    if not isinstance(torch_cuda, (tuple, list)):
        return False
    if any(not isinstance(state, np.ndarray) or state.ndim != 1 or state.size == 0
           or state.dtype != np.dtype('uint8') for state in torch_cuda):
        return False
    if 'torch_cuda_device_count' in value:
        count = value['torch_cuda_device_count']
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer))
                or count != len(torch_cuda)):
            return False
    if 'torch_cuda_device_order' in value:
        order = value['torch_cuda_device_order']
        if (not isinstance(order, (tuple, list))
                or list(order) != list(range(len(torch_cuda)))):
            return False
    return True


def _numeric_trace(value):
    """Finite numeric tree used for prefix traces (metadata is not a trace)."""
    if isinstance(value, np.ndarray):
        return bool(value.dtype.kind in 'biufc' and np.isfinite(value).all())
    if isinstance(value, np.generic):
        return bool(value.dtype.kind in 'biufc' and np.isfinite(value))
    if isinstance(value, (bool, int, float, complex)):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return bool(value) and all(isinstance(k, str) and _numeric_trace(v)
                                   for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_numeric_trace(v) for v in value)
    return False


def _present_contract(value):
    """Reject absent/empty provenance before it is reduced to a hash."""
    if value is None:
        return False
    if isinstance(value, (str, bytes, dict, list, tuple, set, frozenset)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return bool(value.size)
    return True


def capture_native_rng(torch_module=None):
    """Capture the process RNG streams used by a native player/simulator.

    The function intentionally accepts Torch as an argument.  This keeps the
    contract module importable before Isaac Gym/Torch initialization while
    still making the native CUDA stream part of the branch identity once the
    collector is running.
    """
    result = dict(python=random.getstate(), numpy=np.random.get_state(),
                  torch_cpu=None, torch_cuda=[], torch_cuda_device_count=0,
                  torch_cuda_device_order=[])
    if torch_module is None:
        raise ValueError('native twin capture requires the initialized torch module')
    result['torch_cpu'] = _cpu(torch_module.get_rng_state())
    if result['torch_cpu'] is None:
        raise ValueError('native torch CPU RNG state is unavailable')
    if bool(torch_module.cuda.is_available()):
        result['torch_cuda'] = [_cpu(value) for value in torch_module.cuda.get_rng_state_all()]
        result['torch_cuda_device_count'] = len(result['torch_cuda'])
        result['torch_cuda_device_order'] = list(range(result['torch_cuda_device_count']))
    return result


def restore_native_rng(rng, torch_module=None):
    """Restore the captured process RNG streams before launching a branch.

    Isaac/PhysX and any per-generator streams remain the native runner's
    responsibility.  This helper only restores the streams that
    ``capture_native_rng`` records, and refuses a device-count mismatch rather
    than silently running a branch with a different CUDA RNG topology.
    """
    saved = _cpu(rng)
    if not _valid_rng_state(saved):
        raise ValueError('native twin RNG state has an invalid format')
    if torch_module is None:
        raise ValueError('native twin restore requires the initialized torch module')
    cuda = torch_module.cuda
    available = bool(cuda.is_available())
    device_count = int(cuda.device_count()) if available else 0
    saved_count = int(saved.get('torch_cuda_device_count', len(saved['torch_cuda'])))
    if device_count != saved_count:
        raise ValueError('native twin CUDA RNG device count mismatch')
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch_module.set_rng_state(
        torch_module.tensor(saved['torch_cpu'], dtype=torch_module.uint8, device='cpu'))
    if saved_count:
        cuda.set_rng_state_all([
            torch_module.tensor(state, dtype=torch_module.uint8, device='cpu')
            for state in saved['torch_cuda']
        ])


def replay_provenance(prefix_states, prefix_actions, *, replay_max_abs_error,
                      physics_properties, history_contract, controller_identity,
                      physics_dt, control_dt=1 / 30, sim_steps_per_control=1):
    """Build the immutable provenance record for a fresh prefix replay.

    ``prefix_states`` and ``prefix_actions`` are the complete common-prefix
    traces, not just the final pose.  The native caller must create a fresh
    simulator for each arm and report the frame error observed while replaying
    that trace; this helper only records and hashes those facts.
    """
    for name, value in (('physics properties', physics_properties),
                        ('history contract', history_contract),
                        ('controller identity', controller_identity)):
        if not _present_contract(value):
            raise ValueError(name + ' provenance must be nonempty')
    states = _cpu(prefix_states)
    actions = _cpu(prefix_actions)
    try:
        steps = len(actions)
        state_frames = len(states)
    except TypeError as error:
        raise ValueError('prefix actions must be a finite sequence') from error
    if not _numeric_trace(states) or (steps > 0 and not _numeric_trace(actions)):
        raise ValueError('prefix replay traces must be finite')
    actions = np.asarray(actions)
    # Accept the natural ``[]`` spelling for a zero-step trace, but do not
    # silently reinterpret an explicitly shaped empty array with the wrong
    # native action width.
    if steps == 0 and actions.ndim == 1 and actions.size == 0:
        actions = np.zeros((0, 18), dtype='float32')
    if actions.ndim != 2 or actions.shape[1] != 18:
        raise ValueError('prefix actions must have shape [steps,18]')
    if (isinstance(steps, bool) or not isinstance(steps, (int, np.integer))
            or steps < 0):
        raise ValueError('prefix replay length must be a nonnegative integer')
    if state_frames != steps + 1:
        raise ValueError('prefix state trace must contain initial plus one frame per action')
    error = replay_max_abs_error
    if (isinstance(error, (bool, np.bool_)) or not isinstance(error, (int, float, np.number))
            or not np.isfinite(error) or error < 0 or error > 1e-3):
        raise ValueError('fresh prefix replay error exceeds the 1e-3 contract')
    if (isinstance(sim_steps_per_control, (bool, np.bool_))
            or not isinstance(sim_steps_per_control, (int, np.integer))
            or sim_steps_per_control < 1):
        raise ValueError('native twin simulation steps per control must be a positive integer')
    if (isinstance(control_dt, (bool, np.bool_))
            or not isinstance(control_dt, (int, float, np.number))
            or not np.isfinite(control_dt)
            or not np.isclose(control_dt, NATIVE_CONTROL_DT, atol=1e-8, rtol=0)):
        raise ValueError('native twin control dt must be exactly 1/30')
    dt = physics_dt
    if (isinstance(dt, (bool, np.bool_)) or not isinstance(dt, (int, float, np.number))
            or not np.isfinite(dt) or dt <= 0
            or not np.isclose(dt * sim_steps_per_control, control_dt, atol=1e-8, rtol=0)):
        raise ValueError('native twin physics dt and decimation must match control dt')
    return dict(
        prefix_hash=fingerprint(states),
        prefix_state_count=int(state_frames),
        prefix_steps=int(steps),
        prefix_action_hash=fingerprint(actions),
        prefix_action_count=int(steps),
        fresh_simulator=True,
        initial_frame_count=0,
        final_frame_count=int(steps * sim_steps_per_control),
        replay_max_abs_error=float(error),
        physics_properties_hash=fingerprint(physics_properties),
        history_contract_hash=fingerprint(history_contract),
        controller_identity_hash=fingerprint(controller_identity),
        physics_dt=float(dt),
        control_dt=float(control_dt),
        sim_steps_per_control=int(sim_steps_per_control),
    )


def _cpu(value):
    """Copy tensors/arrays without importing Torch before native setup."""
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value.copy())
    if isinstance(value, dict):
        return {str(k): _cpu(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(v) for v in value)
    # Torch tensors are supported lazily by their public detach/cpu contract.
    if hasattr(value, 'detach') and hasattr(value, 'cpu'):
        return np.ascontiguousarray(value.detach().cpu().numpy().copy())
    return copy.deepcopy(value)


def _native_scalar(value):
    """Make task scalar-like control metadata stable and serializable."""
    if isinstance(value, Enum):
        return dict(enum_type=value.__class__.__module__ + '.' + value.__class__.__qualname__,
                    name=value.name, value=_cpu(value.value))
    return _cpu(value)


def fingerprint(value):
    """Content identity independent of pickle memoization and object aliases.

    Pickle encodes shared strings and NumPy dtype objects as backreferences.
    Those identities can change on a save/load round trip without changing
    physical state. Frame each value separately and hash array bytes directly.
    """
    digest = hashlib.sha256()

    def frame(tag, payload=b''):
        digest.update(tag)
        digest.update(struct.pack('>Q', len(payload)))
        digest.update(payload)

    def visit(node):
        if isinstance(node, np.ndarray):
            if node.dtype.hasobject:
                raise ValueError('object arrays cannot define native content identity')
            frame(b'A', repr(node.dtype.descr).encode('utf8'))
            visit(tuple(node.shape))
            frame(b'D', np.ascontiguousarray(node).tobytes())
        elif isinstance(node, np.generic):
            visit(np.asarray(node))
        elif isinstance(node, dict):
            frame(b'M', str(len(node)).encode('ascii'))
            for key in sorted(node):
                visit(key)
                visit(node[key])
        elif isinstance(node, (tuple, list)):
            frame(b'T' if isinstance(node, tuple) else b'L', str(len(node)).encode('ascii'))
            for item in node:
                visit(item)
        elif node is None:
            frame(b'N')
        elif isinstance(node, bool):
            frame(b'B', b'1' if node else b'0')
        elif isinstance(node, int):
            frame(b'I', str(node).encode('ascii'))
        elif isinstance(node, float):
            frame(b'F', struct.pack('>d', node))
        elif isinstance(node, complex):
            frame(b'C', struct.pack('>dd', node.real, node.imag))
        elif isinstance(node, str):
            frame(b'S', node.encode('utf8'))
        elif isinstance(node, bytes):
            frame(b'Y', node)
        else:
            raise ValueError('unsupported native content identity: ' + str(type(node)))

    visit(_cpu(value))
    return digest.hexdigest()


def finite_tree(value):
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, np.generic):
        return bool(np.isfinite(value)) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    return True


def _numeric_finite(value):
    return (isinstance(value, np.ndarray) and value.dtype.kind in 'biufc'
            and bool(np.isfinite(value).all()))


def _is_rigid_pose(value, atol=2e-3):
    """Check a homogeneous object-to-world transform without SciPy."""
    if not isinstance(value, np.ndarray) or value.shape != (4, 4):
        return False
    if not _numeric_finite(value):
        return False
    rotation = value[:3, :3]
    return bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0)
                and np.isclose(np.linalg.det(rotation), 1., atol=atol, rtol=0)
                and np.allclose(value[3], [0., 0., 0., 1.], atol=atol, rtol=0))


def _valid_replay_provenance(value):
    if not isinstance(value, dict):
        return False
    required = {'prefix_hash', 'prefix_steps', 'prefix_state_count', 'prefix_action_hash',
                'prefix_action_count', 'fresh_simulator', 'initial_frame_count',
                'final_frame_count', 'replay_max_abs_error', 'physics_properties_hash',
                'history_contract_hash', 'controller_identity_hash', 'physics_dt',
                'control_dt', 'sim_steps_per_control'}
    if not required.issubset(value):
        return False
    hashes = ('prefix_hash', 'prefix_action_hash', 'physics_properties_hash',
              'history_contract_hash', 'controller_identity_hash')
    if any(not isinstance(value[name], str) or len(value[name]) != 64
           or any(char not in '0123456789abcdef' for char in value[name]) for name in hashes):
        return False
    count_names = ('prefix_steps', 'prefix_state_count', 'prefix_action_count',
                   'initial_frame_count', 'final_frame_count', 'sim_steps_per_control')
    if any(isinstance(value[name], (bool, np.bool_)) or not isinstance(value[name], (int, np.integer))
           for name in count_names):
        return False
    if (value['prefix_steps'] < 0 or value['prefix_state_count'] != value['prefix_steps'] + 1
            or value['prefix_action_count'] != value['prefix_steps']
            or value['sim_steps_per_control'] < 1
            or value['initial_frame_count'] != 0
            or value['final_frame_count'] != value['prefix_steps'] * value['sim_steps_per_control']
            or value['fresh_simulator'] is not True):
        return False
    error = value['replay_max_abs_error']
    return (not isinstance(error, (bool, np.bool_))
            and isinstance(error, (int, float, np.number))
            and np.isfinite(error) and error >= 0 and error <= 1e-3
            and not isinstance(value['physics_dt'], (bool, np.bool_))
            and isinstance(value['physics_dt'], (int, float, np.number))
            and np.isfinite(value['physics_dt'])
            and value['physics_dt'] > 0
            and not isinstance(value['control_dt'], (bool, np.bool_))
            and isinstance(value['control_dt'], (int, float, np.number))
            and np.isfinite(value['control_dt'])
            and np.isclose(value['control_dt'], NATIVE_CONTROL_DT, atol=1e-8, rtol=0)
            and np.isclose(value['physics_dt'] * value['sim_steps_per_control'],
                           value['control_dt'], atol=1e-8, rtol=0))


@dataclass(frozen=True)
class TwinSnapshot:
    """Full pre-branch state captured at a common prefix boundary."""

    pair_id: str
    tick: int
    state: dict
    rng: dict
    history: np.ndarray
    object_pose: np.ndarray
    hand_keypoints: np.ndarray
    replay_provenance: dict
    solver_contract: str = SOLVER_CONTRACT

    def __post_init__(self):
        if not self.pair_id or not isinstance(self.pair_id, str):
            raise ValueError('twin pair_id is required')
        if isinstance(self.tick, (bool, np.bool_)) or not isinstance(self.tick, (int, np.integer)) or self.tick < 0:
            raise ValueError('invalid twin prefix tick')
        if self.solver_contract != SOLVER_CONTRACT:
            raise ValueError('warm PhysX state restore is not a twin contract')
        if not isinstance(self.state, dict) or not self.state:
            raise ValueError('full native task/controller state is required')
        if not REQUIRED_NATIVE_STATE_KEYS.issubset(self.state):
            missing = sorted(REQUIRED_NATIVE_STATE_KEYS - set(self.state))
            raise ValueError('twin snapshot missing native/controller state: ' + ','.join(missing))
        if not REQUIRED_STATE_META_KEYS.issubset(self.state):
            missing = sorted(REQUIRED_STATE_META_KEYS - set(self.state))
            raise ValueError('twin snapshot missing controller provenance: ' + ','.join(missing))
        for name in REQUIRED_NATIVE_STATE_KEYS:
            value = self.state[name]
            if (not isinstance(value, np.ndarray) or value.size == 0
                    or not np.issubdtype(value.dtype, np.number)):
                raise ValueError('native twin state must be a nonempty numeric array: ' + name)
        if (not isinstance(self.state['controller_state'], dict)
                or not self.state['controller_state']
                or not isinstance(self.state['scalars'], dict)
                or not _numeric_finite(self.state['observation'])
                or not self.state['observation'].size):
            raise ValueError('controller/RNN/observation provenance is incomplete')
        if self.state['rnn_state'] is None or self.state['reset_ids'] is None:
            raise ValueError('RNN and reset-id state must be explicit, including non-RNN players')
        rnn_state = self.state['rnn_state']
        if not isinstance(rnn_state, dict) or 'is_rnn' not in rnn_state:
            raise ValueError('RNN state must use an explicit is_rnn sentinel')
        if type(rnn_state['is_rnn']) is not bool:
            raise ValueError('RNN sentinel is_rnn flag must be a bool')
        if rnn_state['is_rnn'] is False:
            if 'state' not in rnn_state or rnn_state['state'] is not None:
                raise ValueError('non-RNN sentinel must explicitly carry state=None')
        elif 'state' not in rnn_state or not _numeric_trace(rnn_state['state']):
            raise ValueError('RNN sentinel must carry a finite numeric state')
        if not _valid_rng_state(self.rng):
            missing = sorted(REQUIRED_RNG_KEYS - set(self.rng)) if isinstance(self.rng, dict) else sorted(REQUIRED_RNG_KEYS)
            if missing:
                raise ValueError('twin snapshot missing RNG state: ' + ','.join(missing))
            raise ValueError('twin snapshot RNG state has an invalid format')
        if not finite_tree(self.state) or not finite_tree(self.rng):
            raise ValueError('nonfinite twin state or RNG')
        if not _valid_replay_provenance(self.replay_provenance):
            raise ValueError('fresh simulator prefix replay provenance is required')
        if self.replay_provenance['prefix_steps'] != self.tick:
            raise ValueError('prefix replay length must equal twin tick')
        for name, value in (('history', self.history), ('object_pose', self.object_pose),
                            ('hand_keypoints', self.hand_keypoints)):
            if not _numeric_finite(value):
                raise ValueError('invalid twin physical/history anchor: ' + name)
        if self.history.ndim < 1 or not self.history.size:
            raise ValueError('nonempty policy history H is required')
        if not _is_rigid_pose(self.object_pose) or self.hand_keypoints.shape != (11, 3):
            raise ValueError('twin physical anchor shape mismatch')

    @property
    def state_hash(self):
        return fingerprint(self.state)

    @property
    def rng_hash(self):
        return fingerprint(self.rng)

    @property
    def common_prefix_hash(self):
        return fingerprint(dict(state=self.state, rng=self.rng, history=self.history,
                                object_pose=self.object_pose, hand_keypoints=self.hand_keypoints,
                                tick=self.tick, replay=self.replay_provenance))


def capture_snapshot(pair_id, tick, state, rng, history, object_pose, hand_keypoints,
                     *, required_state_keys, replay_provenance):
    """Capture a complete state; missing native buffers fail before collection."""
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError('twin pair_id must be a nonempty string')
    if isinstance(tick, (bool, np.bool_)) or not isinstance(tick, (int, np.integer)):
        raise ValueError('twin prefix tick must be an integer')
    state = _cpu(state)
    requested = set(required_state_keys)
    missing_required = (REQUIRED_NATIVE_STATE_KEYS | REQUIRED_STATE_META_KEYS) - requested
    if missing_required:
        raise ValueError('required_state_keys omits canonical native state: ' + ','.join(sorted(missing_required)))
    missing = requested - set(state)
    if missing:
        raise ValueError('twin snapshot missing native/controller state: ' + ','.join(sorted(missing)))
    return TwinSnapshot(pair_id=pair_id, tick=int(tick), state=state, rng=_cpu(rng),
                        history=np.asarray(_cpu(history)), object_pose=np.asarray(_cpu(object_pose)),
                        hand_keypoints=np.asarray(_cpu(hand_keypoints)),
                        replay_provenance=_cpu(replay_provenance))


def capture_native_snapshot(pair_id, tick, task, *, controller_state, rnn_state,
                            observation, scalars, reset_ids, history, object_pose,
                            hand_keypoints, torch_module, replay, is_rnn=False):
    """Adapt one initialized native task/player boundary to ``TwinSnapshot``.

    The adapter is intentionally explicit about the metadata that does not
    live on the Isaac task object (policy/RNN buffers and controller identity).
    It copies every required native tensor by its canonical attribute name and
    refuses a partial or pose-only capture before either branch is run.
    """
    if not hasattr(task, 'gym') or not hasattr(task, 'sim'):
        raise ValueError('native task must expose gym/sim for fresh replay verification')
    if not hasattr(task.gym, 'get_frame_count'):
        raise ValueError('native gym frame counter is required for twin capture')
    if not _valid_replay_provenance(replay):
        raise ValueError('fresh prefix replay provenance is incomplete')
    decimation = getattr(task, 'control_freq_inv', None)
    if (isinstance(decimation, (bool, np.bool_))
            or not isinstance(decimation, (int, np.integer))
            or decimation != replay['sim_steps_per_control']
            or decimation != NATIVE_SIM_STEPS_PER_CONTROL):
        raise ValueError('native simulation decimation does not match replay')
    if (not hasattr(task, 'dt') or not hasattr(task, 'sim_params')
            or not np.isclose(task.dt, replay['control_dt'], atol=1e-8, rtol=0)
            or not np.isclose(task.sim_params.dt, replay['physics_dt'], atol=1e-8, rtol=0)
            or not np.isclose(task.dt, NATIVE_CONTROL_DT, atol=1e-8, rtol=0)
            or not np.isclose(task.sim_params.dt, NATIVE_PHYSICS_DT, atol=1e-8, rtol=0)):
        raise ValueError('native physics/control clock does not match replay')
    frame_count = task.gym.get_frame_count(task.sim)
    if (isinstance(frame_count, (bool, np.bool_))
            or not isinstance(frame_count, (int, np.integer))
            or frame_count != tick * decimation):
        raise ValueError('native frame count does not match the replay prefix tick')
    if getattr(task, 'dr_randomizations', None):
        raise ValueError('native randomization state must be disabled for twin replay')
    if getattr(task, 'projtype', 'None') not in (None, '', 'None'):
        raise ValueError('native projectile/randomization state is unsupported for twin replay')
    if getattr(task, '_motion_sampler', None) is not None:
        raise ValueError('native motion sampler state is unsupported for twin replay')
    if (not isinstance(replay, dict) or replay.get('fresh_simulator') is not True
            or replay.get('prefix_steps') != tick
            or replay.get('final_frame_count') != frame_count):
        raise ValueError('fresh prefix replay provenance does not match native task state')
    if not isinstance(is_rnn, (bool, np.bool_)):
        raise ValueError('native controller is_rnn flag must be boolean')
    if bool(is_rnn) and rnn_state is None:
        raise ValueError('RNN native controller state is missing')
    if not bool(is_rnn) and rnn_state is not None:
        raise ValueError('non-RNN native controller must use an explicit None state')
    rnn_state = ({'is_rnn': False, 'state': None} if not bool(is_rnn)
                 else {'is_rnn': True, 'state': rnn_state})
    state = {}
    for name in REQUIRED_NATIVE_STATE_KEYS:
        if not hasattr(task, name):
            raise ValueError('native task missing twin state: ' + name)
        state[name] = getattr(task, name)
    tensor_names = []
    extra_tensors = {}
    for name, value in vars(task).items():
        if isinstance(value, np.ndarray) or (hasattr(value, 'detach') and hasattr(value, 'cpu')):
            tensor_names.append(str(name))
            if name not in REQUIRED_NATIVE_STATE_KEYS:
                extra_tensors[str(name)] = value
    if not REQUIRED_NATIVE_STATE_KEYS.issubset(tensor_names):
        raise ValueError('native task tensor inventory is missing canonical state')
    state['native_tensor_inventory'] = tuple(sorted(tensor_names))
    state['native_extra_tensors'] = extra_tensors
    state['native_scalar_inventory'] = {
        str(name): _native_scalar(value) for name, value in vars(task).items()
        if isinstance(value, (str, bool, int, float, np.generic, Enum))
    }
    state.update(controller_state=controller_state, rnn_state=rnn_state,
                 observation=observation, scalars=scalars, reset_ids=reset_ids)
    required = REQUIRED_NATIVE_STATE_KEYS | REQUIRED_STATE_META_KEYS
    return capture_snapshot(
        pair_id, tick, state, capture_native_rng(torch_module), history,
        object_pose, hand_keypoints, required_state_keys=required,
        replay_provenance=replay)


@dataclass(frozen=True)
class TwinBranch:
    """One 24-step branch launched from a common snapshot."""

    snapshot_hash: str
    pair_id: str
    branch_id: str
    residual_plan: np.ndarray
    actions: np.ndarray
    object_poses: np.ndarray
    hand_keypoints: np.ndarray
    done: np.ndarray
    common_prefix_hash: str

    def __post_init__(self):
        if self.branch_id not in BRANCHES:
            raise ValueError('twin branch_id must be a or b')
        if not self.pair_id or not self.snapshot_hash or not self.common_prefix_hash:
            raise ValueError('twin branch identity is incomplete')
        for name, value in (('residual_plan', self.residual_plan), ('actions', self.actions),
                            ('object_poses', self.object_poses), ('hand_keypoints', self.hand_keypoints),
                            ('done', self.done)):
            if not isinstance(value, np.ndarray):
                raise ValueError('twin branch arrays must be numpy arrays: ' + name)
        if self.residual_plan.shape != (K, 18) or not _numeric_finite(self.residual_plan):
            raise ValueError('twin residual plan must be a finite 24x18 array')
        if np.abs(self.residual_plan).max() > .2 + 1e-6:
            raise ValueError('twin residual plan exceeds allowed amplitude')
        if (self.actions.shape != (K, 18) or not _numeric_finite(self.actions)
                or np.abs(self.actions).max() > 1. + 1e-6):
            raise ValueError('twin branch must contain exactly 24 executed controls')
        if (self.object_poses.shape != (K + 1, 4, 4)
                or not all(_is_rigid_pose(pose) for pose in self.object_poses)):
            raise ValueError('twin branch must contain current plus 24 object poses')
        if self.hand_keypoints.shape != (K + 1, 11, 3) or not _numeric_finite(self.hand_keypoints):
            raise ValueError('twin branch must contain current plus 24 hand states')
        if self.done.shape != (K,) or self.done.dtype != np.bool_:
            raise ValueError('twin branch done mask must contain 24 booleans')
        if self.done[:-1].any():
            raise ValueError('twin branch terminated before all 24 steps')


def validate_pair(snapshot_a, snapshot_b, branch_a, branch_b, *, atol=1e-6):
    """Validate common state/RNG and distinct known plans before supervision."""
    if snapshot_a.pair_id != snapshot_b.pair_id or branch_a.pair_id != snapshot_a.pair_id:
        raise ValueError('twin pair identity mismatch')
    if branch_b.pair_id != snapshot_a.pair_id:
        raise ValueError('twin pair identity mismatch')
    if branch_a.branch_id != 'a' or branch_b.branch_id != 'b':
        raise ValueError('twin branches must be emitted as a/b')
    if snapshot_a.common_prefix_hash != snapshot_b.common_prefix_hash:
        raise ValueError('twin branches do not share an exact common prefix')
    if snapshot_a.replay_provenance != snapshot_b.replay_provenance:
        raise ValueError('twin replay provenance differs at the branch point')
    if branch_a.common_prefix_hash != snapshot_a.common_prefix_hash or branch_b.common_prefix_hash != snapshot_a.common_prefix_hash:
        raise ValueError('branch common-prefix provenance mismatch')
    if branch_a.snapshot_hash != snapshot_a.state_hash or branch_b.snapshot_hash != snapshot_b.state_hash:
        raise ValueError('branch full-state snapshot identity mismatch')
    if not np.array_equal(snapshot_a.history, snapshot_b.history):
        raise ValueError('twin policy histories differ at the branch point')
    if not np.allclose(snapshot_a.object_pose, snapshot_b.object_pose, atol=atol, rtol=0):
        raise ValueError('twin object states differ at the branch point')
    if not np.allclose(snapshot_a.hand_keypoints, snapshot_b.hand_keypoints, atol=atol, rtol=0):
        raise ValueError('twin hand states differ at the branch point')
    if not np.allclose(branch_a.object_poses[0], snapshot_a.object_pose, atol=atol, rtol=0):
        raise ValueError('twin branch a starts from the wrong object state')
    if not np.allclose(branch_b.object_poses[0], snapshot_b.object_pose, atol=atol, rtol=0):
        raise ValueError('twin branch b starts from the wrong object state')
    if not np.allclose(branch_a.hand_keypoints[0], snapshot_a.hand_keypoints, atol=atol, rtol=0):
        raise ValueError('twin branch a starts from the wrong hand state')
    if not np.allclose(branch_b.hand_keypoints[0], snapshot_b.hand_keypoints, atol=atol, rtol=0):
        raise ValueError('twin branch b starts from the wrong hand state')
    distance = float(np.linalg.norm(branch_a.residual_plan - branch_b.residual_plan))
    if distance < MIN_PLAN_L2:
        raise ValueError('twin residual plans are indistinguishable')
    realized_distance = float(np.linalg.norm(branch_a.actions - branch_b.actions))
    if realized_distance < MIN_REALIZED_ACTION_L2:
        raise ValueError('twin residual plans produce indistinguishable executed controls')
    return dict(schema=SCHEMA, pair_id=snapshot_a.pair_id, branch_ids=list(BRANCHES),
                tick=snapshot_a.tick, common_prefix_hash=snapshot_a.common_prefix_hash,
                state_hash=snapshot_a.state_hash, rng_hash=snapshot_a.rng_hash,
                residual_plan_l2=distance, realized_action_l2=realized_distance, actions=K,
                solver_contract=snapshot_a.solver_contract,
                residual_plan_semantics=RESIDUAL_PLAN_SEMANTICS,
                executed_action_semantics=EXECUTED_ACTION_SEMANTICS,
                prefix_hash=snapshot_a.replay_provenance['prefix_hash'],
                plan_a_hash=fingerprint(branch_a.residual_plan),
                plan_b_hash=fingerprint(branch_b.residual_plan))
