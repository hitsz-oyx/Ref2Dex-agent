"""Reset-bound provenance and one shared producer for spatial waypoint goals.

The receipts enforce declared lineage/order, not the truth of forged metadata.
Only collector-owned timestamps in one clock domain are evidence of availability.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import shutil

import numpy as np

from .goal_scheduler import OrderedGoalScheduler, SchedulerConfig, _pose, update_schedulers


SCHEMA = 'ref2dex.reset-bound-waypoint.v1'
TIME_SEMANTICS = 'ordered_geometric_stage_no_arrival_deadline'
ROLES = {'task_instruction', 'reset_observation', 'reset_geometry'}
KINDS = {'declared_task', 'reset_measurement', 'nominal_constructor', 'independent_recording'}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_reset(reset):
    for key in ('execution_id', 'reset_id', 'clock_id'):
        if not isinstance(reset.get(key), str) or not reset[key]:
            raise ValueError('explicit reset binding required: ' + key)
    if type(reset.get('at_ns')) is not int or reset['at_ns'] < 0:
        raise ValueError('integer reset timestamp required')


def validate_source(source, reset, *, role=None):
    """Accept current/reset observations; reject future execution labels."""
    validate_reset(reset)
    if source.get('source_role') not in ROLES or (role and source['source_role'] != role):
        raise ValueError('illegal source_role; realized future poses are labels')
    if source.get('source_kind') not in KINDS:
        raise ValueError('explicit causal source_kind required')
    if source.get('available_before_reset') is not True:
        raise ValueError('availability before reset is unproven')
    if source.get('reset') != reset:
        raise ValueError('source belongs to a different execution/reset/clock')
    if not isinstance(source.get('source_id'), str) or not source['source_id']:
        raise ValueError('source_id required')
    for field in ('observed_at_ns', 'available_at_ns'):
        timestamp = source.get(field)
        if type(timestamp) is not int or not 0 <= timestamp <= reset['at_ns']:
            raise ValueError('future or missing source timestamp: ' + field)
    if source['observed_at_ns'] > source['available_at_ns']:
        raise ValueError('observation cannot become available before acquisition')
    if source['source_kind'] == 'independent_recording':
        execution = source.get('source_execution_id')
        if not isinstance(execution, str) or not execution or execution == reset['execution_id']:
            raise ValueError('target execution cannot supply its own future instruction')
    elif source.get('source_execution_id') is not None:
        raise ValueError('recorded execution must declare independent_recording')
    if source['source_role'] == 'task_instruction' and source['source_kind'] == 'reset_measurement':
        raise ValueError('observation alone does not declare task intent')
    if source.get('source_sha256') != digest(source.get('payload')):
        raise ValueError('source content differs from frozen source hash')
    return source


def bind_source(payload, *, source_role, source_kind, source_id, reset,
                observed_at_ns, available_at_ns, source_execution_id=None):
    source = dict(source_role=source_role, source_kind=source_kind, source_id=source_id,
                  reset=plain(reset), observed_at_ns=observed_at_ns,
                  available_at_ns=available_at_ns, source_execution_id=source_execution_id,
                  available_before_reset=True, payload=plain(payload), source_sha256=digest(payload))
    validate_source(source, reset)
    return source


def instruction_from_displacement(anchor, rule, query, surface, *, reset,
                                  object_slot, hand_slot, goal_type):
    """Declared fixed WORLD translation from an explicitly identified anchor.

    Supports a collector's nominal constructor anchor as well as measured reset
    poses. It never substitutes a measured pose for a nominal constructor pose.
    A zero translation is legal only as an explicit task rule, not as unknown.
    """
    validate_source(anchor, reset)
    if anchor['source_role'] not in ('reset_observation', 'task_instruction'):
        raise ValueError('pose anchor source required')
    validate_source(rule, reset, role='task_instruction')
    pose = _pose(anchor['payload'])
    if rule['payload'].get('rule') != 'fixed_world_translation':
        raise ValueError('explicit fixed_world_translation rule required')
    shift = np.asarray(rule['payload']['translation_m'], dtype=np.float64)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError('finite world translation required')
    goal = pose.copy()
    goal[:3, 3] += shift
    artifact = dict(schema=SCHEMA, reset=plain(reset), source_role='scheduler_instruction',
                    available_before_reset=True, goal_semantics='spatial_path_waypoint',
                    time_semantics=TIME_SEMANTICS, reference_endpoint_time_s=None,
                    object_slot=object_slot, hand_slot=hand_slot, goal_type=goal_type,
                    scheduler='wam_rl_policy_v3.goal_scheduler.OrderedGoalScheduler',
                    scheduler_config=asdict(SchedulerConfig()),
                    producer='fixed_world_translation.v1',
                    sources=dict(anchor=anchor, rule=rule, query=query, surface=surface),
                    reference_poses=[pose.tolist(), goal.tolist()], reference_times_s=[0., .8])
    artifact['instruction_sha256'] = digest(artifact)
    validate_instruction(artifact)
    return artifact


def validate_instruction(artifact):
    if artifact.get('schema') != SCHEMA or artifact.get('time_semantics') != TIME_SEMANTICS:
        raise ValueError('unknown waypoint/time contract')
    if (artifact.get('goal_semantics') != 'spatial_path_waypoint'
            or artifact.get('reference_endpoint_time_s') is not None
            or artifact.get('source_role') != 'scheduler_instruction'
            or artifact.get('available_before_reset') is not True
            or artifact.get('scheduler') != 'wam_rl_policy_v3.goal_scheduler.OrderedGoalScheduler'
            or artifact.get('producer') != 'fixed_world_translation.v1'):
        raise ValueError('illegal instruction semantics/producer')
    for key in ('object_slot', 'hand_slot'):
        if type(artifact.get(key)) is not int or artifact[key] < 0:
            raise ValueError('explicit nonnegative slot binding required: ' + key)
    reset = artifact['reset']
    sources = artifact['sources']
    for name, role in (('anchor', None), ('rule', 'task_instruction'),
                       ('query', 'reset_geometry'), ('surface', 'reset_geometry')):
        validate_source(sources[name], reset, role=role)
    query = np.asarray(sources['query']['payload'], dtype=np.float64)
    surface = np.asarray(sources['surface']['payload'], dtype=np.float64)
    if query.shape != (32, 3) or not np.isfinite(query).all():
        raise ValueError('32 finite same-ID query points required')
    if surface.ndim != 2 or surface.shape[1:] != (3,) or not len(surface) or not np.isfinite(surface).all():
        raise ValueError('finite interaction surface required')
    # Re-derive the path from bound sources: a valid hash alone must not allow
    # a caller to replace the path with the target execution's future poses.
    anchor = sources['anchor']
    if anchor['source_role'] not in ('reset_observation', 'task_instruction'):
        raise ValueError('pose anchor source required')
    pose = _pose(anchor['payload'])
    rule = sources['rule']['payload']
    if rule.get('rule') != 'fixed_world_translation':
        raise ValueError('explicit fixed_world_translation rule required')
    shift = np.asarray(rule['translation_m'], dtype=np.float64)
    if shift.shape != (3,) or not np.isfinite(shift).all():
        raise ValueError('finite world translation required')
    end = pose.copy(); end[:3, 3] += shift
    if (not np.array_equal(artifact['reference_poses'], np.stack([pose, end]))
            or artifact['reference_times_s'] != [0., .8]):
        raise ValueError('path not derived from bound reset sources')
    unsigned = {k: v for k, v in artifact.items() if k != 'instruction_sha256'}
    if artifact.get('instruction_sha256') != digest(unsigned):
        raise ValueError('instruction binding changed')
    return artifact


class ResetWaypointChannel:
    """Use this producer identically for a recorded replay and a fresh execution.

    Live samples are supplied one at a time. No trajectory or future pose array
    is accepted at this boundary. The scheduler starts at the bound reset and
    must be replayed without skipping observations before a selected window.
    """
    def __init__(self, instruction, *, expected_reset, artifact_dir=None):
        validate_instruction(instruction)
        if instruction['reset'] != expected_reset:
            raise ValueError('instruction reset does not match target execution')
        self.instruction = json.loads(json.dumps(instruction))
        self.reset = dict(expected_reset)
        self.last_tick = -1
        self.last_observed_at_ns = None
        self.scheduler = OrderedGoalScheduler(
            instruction['reference_poses'], instruction['reference_times_s'],
            instruction['sources']['query']['payload'],
            interaction_points=instruction['sources']['surface']['payload'],
            config=SchedulerConfig(**instruction['scheduler_config']), goal_type=instruction['goal_type'])
        self.artifact_dir = None
        self.written_bytes = 0
        if artifact_dir is not None:
            folder = Path(artifact_dir)
            if shutil.disk_usage(folder.parent).free < 20 * 2**30:
                raise RuntimeError('less than 20GiB free; refusing new goal artifacts')
            folder.mkdir(exist_ok=False)
            content = json.dumps(self.instruction, sort_keys=True, allow_nan=False) + '\n'
            if len(content.encode()) > 2**20:
                raise ValueError('reset instruction exceeds 1MiB limit')
            with (folder / 'instruction.json').open('x') as stream:
                stream.write(content)
            (folder / 'waypoints.jsonl').touch(exist_ok=False)
            self.artifact_dir = folder
            self.written_bytes = len(content.encode())

    def update(self, pose, hand11, *, tick, observed_at_ns, available_at_ns, decision_at_ns):
        if type(tick) is not int or tick != self.last_tick + 1:
            raise ValueError('continuous replay from reset required; no new per-window reset')
        times = (observed_at_ns, available_at_ns, decision_at_ns)
        if any(type(x) is not int for x in times) or not (
                self.reset['at_ns'] <= observed_at_ns <= available_at_ns <= decision_at_ns):
            raise ValueError('future/unavailable live observation')
        if self.last_observed_at_ns is not None and observed_at_ns <= self.last_observed_at_ns:
            raise ValueError('live observation acquisition must advance')
        # tick is the resampled WAM 30Hz frame clock, never native controls or
        # wall time. Acquisition timestamps retain real sensor/collector jitter.
        decision = update_schedulers([self.scheduler], np.asarray(pose)[None],
            np.asarray([tick / 30.]), np.asarray(hand11)[None], fast_geometry=True)[0]
        self.last_tick = tick
        self.last_observed_at_ns = observed_at_ns
        # Match online_runtime._goal_from_pose's arithmetic/rounding boundary,
        # including nonidentity world poses; this only encodes the scheduler's
        # pose and does not choose another waypoint.
        relative = np.linalg.inv(np.asarray(pose, dtype=np.float64)) @ decision['goal_pose']
        query = self.scheduler.points
        delta = (query @ relative[:3, :3].T + relative[:3, 3] - query).astype(np.float32)
        tensors = dict(goal_object_delta=delta, goal_object_points=query.astype(np.float32) + delta,
                       object_goal_valid=np.ones(len(query), dtype=bool))
        receipt = dict(
            schema=SCHEMA, source_role='scheduler_waypoint',
            available_before_reset=True, availability_scope='instruction_and_reset_sources_only',
            live_observation_available_before_reset=(tick == 0),
            reset=self.reset, instruction_sha256=self.instruction['instruction_sha256'],
            tick=tick, observed_at_ns=observed_at_ns, available_at_ns=available_at_ns,
            decision_at_ns=decision_at_ns, observation_sha256=digest(dict(pose=pose, hand11=hand11)),
            goal_semantics='spatial_path_waypoint', time_semantics=TIME_SEMANTICS,
            reference_endpoint_time_s=None, planning_horizon_s=decision['horizon_s'],
            mode=decision['mode'], confirmed_index=decision['confirmed_index'],
            target_reference_index=decision['target_reference_index'],
            goal_pose=plain(decision['goal_pose']), goal_delta=plain(delta),
            goal_type=decision['goal_type'])
        if self.artifact_dir is not None:
            content = json.dumps(receipt, sort_keys=True, allow_nan=False) + '\n'
            if self.written_bytes + len(content.encode()) > 16 * 2**20:
                raise RuntimeError('goal artifact exceeds 16MiB per-execution bound')
            if shutil.disk_usage(self.artifact_dir).free < 20 * 2**30:
                raise RuntimeError('less than 20GiB free; stop goal recording')
            with (self.artifact_dir / 'waypoints.jsonl').open('a') as stream:
                stream.write(content)
            self.written_bytes += len(content.encode())
        return tensors, receipt
