"""Bind learned conditioning and execution semantics across save/resume."""
import hashlib
import json
import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=lambda x:
        x.tolist() if hasattr(x, 'tolist') else str(x), separators=(',', ':')).encode()).hexdigest()


def geometry_bindings(objects, geometry):
    result = {}
    for obj, fields in zip(objects, geometry):
        key = '%s@%g' % (obj['object_code'], obj['scale'])
        digest = hashlib.sha256()
        for field in fields:
            array = np.ascontiguousarray(field)
            digest.update(str((array.shape, str(array.dtype))).encode())
            digest.update(array.tobytes())
        if key in result and result[key] != digest.hexdigest():
            raise ValueError('Conflicting geometry for ' + key)
        result[key] = digest.hexdigest()
    return result


def execution_identity(runtime, *, protocol, goal_type, termination_policy):
    from dataclasses import asdict
    from .goal_scheduler import SchedulerConfig
    return dict(schema='ref2dex.scheduled-execution.v2',
        wam=runtime.conditioning_identity,
        geometry=runtime.geometry_bindings,
        reset_protocol=protocol, goal_type=goal_type,
        scheduler=asdict(SchedulerConfig()), termination_policy=termination_policy,
        reset_sensor_semantics='native_zero_action_physics_initialization')


def validate_execution(saved, current, *, evaluating=False, migration_reason=None):
    """Evaluation may select a geometry subset; known geometries must match."""
    differences = []
    if saved is None:
        differences.append('legacy checkpoint has no bound execution identity')
    else:
        for key in sorted(set(saved) | set(current)):
            if key == 'geometry' and evaluating:
                missing = {k: v for k, v in current[key].items() if saved[key].get(k) != v}
                if missing: differences.append('geometry')
            elif saved.get(key) != current.get(key):
                differences.append(key)
    if differences and not (migration_reason and migration_reason.strip()):
        raise ValueError('Checkpoint execution identity mismatch: ' + ', '.join(differences)
                         + '; explicit --checkpoint-migration-reason required')
    return dict(changed=bool(differences), differences=differences,
                migration_reason=migration_reason, source_identity=saved,
                target_identity=current)
