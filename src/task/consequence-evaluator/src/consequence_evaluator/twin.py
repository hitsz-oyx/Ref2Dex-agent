"""Contracts for exact current-state twin branches.

This module is deliberately Isaac-free.  A native collector may use it after
creating a *fresh* simulator and replaying the same prefix.  Copying only an
object pose or hand keypoints is rejected because controller history, DOF
velocities, contact buffers, and RNG state can change the next 24 steps.
"""
from __future__ import annotations

import copy
import hashlib
import pickle
from dataclasses import dataclass

import numpy as np

from .contracts import K

SCHEMA = 'ref2dex.consequence-evaluator.twin.v1'
BRANCHES = ('a', 'b')
MIN_PLAN_L2 = .01
MIN_REALIZED_ACTION_L2 = .01
SOLVER_CONTRACT = 'fresh_simulator_prefix_replay'

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


def fingerprint(value):
    """Stable content identity for a captured state or RNG tree."""
    return hashlib.sha256(pickle.dumps(_cpu(value), protocol=4)).hexdigest()


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
    required = {'prefix_hash', 'prefix_steps', 'prefix_action_hash',
                'prefix_action_count', 'fresh_simulator', 'initial_frame_count',
                'final_frame_count', 'replay_max_abs_error', 'physics_properties_hash',
                'history_contract_hash', 'controller_identity_hash', 'physics_dt'}
    if not required.issubset(value):
        return False
    hashes = ('prefix_hash', 'prefix_action_hash', 'physics_properties_hash',
              'history_contract_hash', 'controller_identity_hash')
    if any(not isinstance(value[name], str) or len(value[name]) != 64
           or any(char not in '0123456789abcdef' for char in value[name]) for name in hashes):
        return False
    if (isinstance(value['prefix_steps'], bool) or not isinstance(value['prefix_steps'], int)
            or value['prefix_steps'] < 0 or value['prefix_action_count'] != value['prefix_steps']
            or value['initial_frame_count'] != 0 or value['final_frame_count'] != value['prefix_steps']
            or value['fresh_simulator'] is not True):
        return False
    error = value['replay_max_abs_error']
    return (not isinstance(error, (bool, np.bool_))
            and isinstance(error, (int, float, np.number))
            and np.isfinite(error) and error >= 0 and error <= 1e-3
            and not isinstance(value['physics_dt'], (bool, np.bool_))
            and isinstance(value['physics_dt'], (int, float, np.number))
            and np.isfinite(value['physics_dt']) and value['physics_dt'] > 0)


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
        if not isinstance(self.rng, dict) or not REQUIRED_RNG_KEYS.issubset(self.rng):
            missing = sorted(REQUIRED_RNG_KEYS - set(self.rng)) if isinstance(self.rng, dict) else sorted(REQUIRED_RNG_KEYS)
            raise ValueError('twin snapshot missing RNG state: ' + ','.join(missing))
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
                        history=_cpu(np.asarray(history)), object_pose=_cpu(np.asarray(object_pose)),
                        hand_keypoints=_cpu(np.asarray(hand_keypoints)),
                        replay_provenance=_cpu(replay_provenance))


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
                prefix_hash=snapshot_a.replay_provenance['prefix_hash'],
                plan_a_hash=fingerprint(branch_a.residual_plan),
                plan_b_hash=fingerprint(branch_b.residual_plan))
