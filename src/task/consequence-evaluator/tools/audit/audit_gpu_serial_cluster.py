"""Audit a same-process frozen-control serial cluster.

This is a descriptive execution diagnostic.  The reactive teacher supplies an
executed action stream; frozen zero and candidate arms replay that stream after
an in-place reset.  The reset does not restore hidden GPU PhysX contact state,
so this output cannot be used by the strict Gate1 scorer or as a utility claim.
"""
import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
import sys
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.gate1 import candidate_plan
from consequence_evaluator.twin import fingerprint

FIELD_FLOORS = {
    'object_pose': 1e-3,
    'hand_keypoints': 2e-3,
    'surface_gap': 1e-3,
    'support_gap': 1e-3,
    'table_footprint': 1e-3,
    'dof_position': 2e-3,
    'dof_velocity': 5e-2,
    'object_velocity': 5e-2,
    'history': 5e-2,
    'native_contact_forces': 5.,
    'native_object_contact_forces': 5.,
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return dict(p50=0., p90=0., p95=0., max=0.)
    return dict(zip(('p50', 'p90', 'p95', 'max'),
                    [float(value) for value in np.quantile(values, [.5, .9, .95, 1.])]))


def p95_abs(value):
    return quantiles(np.abs(np.asarray(value, dtype=np.float64)))


def load_packet(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        packet = pickle.load(stream)
    if packet.get('schema') != 'ref2dex.consequence-gate1.gpu-serial-cluster.v1':
        raise ValueError('serial cluster packet schema required')
    if packet.get('engineering_only') is not True or packet.get('serial_cluster') is not True:
        raise ValueError('engineering-only serial cluster packet required')
    if packet.get('group_mode') != 'same_process_same_env_reset_frozen_control_cluster':
        raise ValueError('frozen-control cluster mode required')
    roles = list(packet.get('roles') or [])
    expected_schedule = ['reactive_teacher', 'frozen_zero_1', 'positive_1',
                         'frozen_zero_2', 'negative_1', 'frozen_zero_3',
                         'negative_2', 'frozen_zero_4', 'positive_2', 'frozen_zero_5']
    zero_roles = list(packet.get('zero_roles') or [])
    candidate_roles = list(packet.get('candidate_roles') or [])
    teacher = packet.get('teacher_role')
    expected_zero_roles = {f'frozen_zero_{index}' for index in range(1, 6)}
    expected_candidate_roles = {f'positive_{index}' for index in range(1, 3)} | {
        f'negative_{index}' for index in range(1, 3)}
    if (roles != expected_schedule or packet.get('execution_order') != expected_schedule
            or teacher != 'reactive_teacher' or teacher in zero_roles or teacher in candidate_roles
            or len(zero_roles) != 5 or len(candidate_roles) != 4
            or set(zero_roles) != expected_zero_roles or set(candidate_roles) != expected_candidate_roles
            or len(set(roles)) != len(roles)
            or set(zero_roles) & set(candidate_roles)
            or set(zero_roles) | set(candidate_roles) != set(roles[1:])
            or set(packet.get('execution_order') or []) != set(roles)
            or len(packet.get('execution_order') or []) != len(roles)):
        raise ValueError('teacher/zero/candidate role contract required')
    if packet.get('control_prefix_exact') is not True:
        raise ValueError('shared teacher prefix contract required')
    query = int(packet.get('query_tick', -1)); horizon = int(packet.get('horizon', -1))
    steps = int(packet.get('steps', -1))
    if query != 48 or horizon != 24 or steps != 72:
        raise ValueError('cluster is fixed to the query tick48 plus24 step window')
    actions = np.asarray(packet.get('actions'))
    done = np.asarray(packet.get('done'))
    arms = len(roles)
    if actions.shape != (arms, steps, 18) or done.shape != (arms, steps):
        raise ValueError('cluster action/done shape mismatch')
    if not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
        raise ValueError('cluster actions are not finite bounded native controls')
    fields = ('object_pose', 'hand_keypoints', 'surface_gap', 'support_gap',
              'table_footprint', 'dof_position', 'dof_velocity', 'object_velocity',
              'history', 'native_contact_forces', 'native_object_contact_forces')
    for key in fields:
        value = np.asarray(packet.get(key))
        if value.shape[0] != arms or value.shape[1] != steps + 1 or not np.isfinite(value).all():
            raise ValueError('cluster field shape/nonfinite mismatch: %s' % key)
    timestamps = np.asarray(packet.get('timestamps'))
    if (timestamps.shape != (steps + 1,) or not np.isfinite(timestamps).all()
            or not np.allclose(np.diff(timestamps), 1. / 30., atol=1e-8, rtol=0)):
        raise ValueError('cluster timestamp shape mismatch')
    residual = np.asarray(packet.get('requested_residual'))
    if residual.shape != actions.shape or not np.isfinite(residual).all():
        raise ValueError('cluster residual plan shape mismatch')
    return path, packet, roles, zero_roles, candidate_roles, query, horizon, steps, fields


def action_contract(packet, roles, zero_roles, candidate_roles, query, horizon):
    actions = np.asarray(packet['actions'], dtype=np.float64)
    done = np.asarray(packet['done'])
    residual = np.asarray(packet['requested_residual'], dtype=np.float64)
    teacher = actions[0]
    prefix = all(np.array_equal(actions[index, :query], teacher[:query])
                 for index in range(1, len(roles)))
    suffix = all(np.array_equal(actions[index, query + horizon:], teacher[query + horizon:])
                 for index in range(1, len(roles)))
    done_exact = all(np.array_equal(done[index], done[0]) for index in range(1, len(roles)))
    done_no_early = not bool(done[:, :-1].any())
    expected_errors = {}
    residual_errors = {}
    for index, role in enumerate(roles):
        if role in zero_roles:
            expected = teacher
            residual_errors[role] = float(np.max(np.abs(np.asarray(packet['requested_residual'])[index])))
        else:
            expected = teacher.copy()
            expected[query:query + horizon] = np.clip(
                expected[query:query + horizon] + residual[index, query:query + horizon], -1., 1.)
            candidate = 1 if role.startswith('positive') else 2
            residual_errors[role] = float(np.max(np.abs(
                residual[index] - np.pad(candidate_plan(candidate),
                                         ((query, actions.shape[1] - query - horizon), (0, 0))))))
        expected_errors[role] = float(np.max(np.abs(actions[index] - expected)))
    zero_exact = all(expected_errors[role] == 0. for role in zero_roles)
    candidate_exact = all(expected_errors[role] <= 1e-6 for role in candidate_roles)
    return dict(control_prefix_exact=bool(prefix), control_suffix_exact=bool(suffix),
                done_exact=bool(done_exact), done_no_early=bool(done_no_early),
                zero_replay_exact=bool(zero_exact), candidate_action_contract_exact=bool(candidate_exact),
                requested_residual_contract_exact=bool(all(value == 0. for role, value in residual_errors.items()
                                                          if role in zero_roles)
                                                       and all(value <= 1e-7 for role, value in residual_errors.items()
                                                               if role in candidate_roles)),
                max_abs_error_by_role=expected_errors,
                residual_max_abs_error_by_role=residual_errors,
                teacher_action_sha256=fingerprint(np.asarray(teacher)),
                packet_teacher_action_sha256=packet.get('teacher_action_sha256'),
                teacher_action_hash_exact=bool(fingerprint(np.asarray(teacher)) == packet.get('teacher_action_sha256')),
                schedule_hash_exact=bool(fingerprint(packet['execution_order']) == packet.get('schedule_sha256')))


def noise_table(packet, roles, zero_roles, candidate_roles, query, steps, fields):
    data = {key: np.asarray(packet[key], dtype=np.float64) for key in fields}
    role_index = {role: index for index, role in enumerate(roles)}
    zero_indices = [role_index[role] for role in zero_roles]
    candidate_indices = [role_index[role] for role in candidate_roles]
    pairs = list(itertools.combinations(zero_indices, 2))
    pairwise = {}
    for key, value in data.items():
        records = []
        for left, right in pairs:
            delta = ((value[right, query + 1:steps + 1] - value[right, query])
                     - (value[left, query + 1:steps + 1] - value[left, query]))
            records.append(dict(left=roles[left], right=roles[right],
                                post_query_displacement_p95=p95_abs(delta)))
        p95s = [record['post_query_displacement_p95']['p95'] for record in records]
        pairwise[key] = dict(pair_count=len(records), pairs=records,
                             p95_across_zero_pairs=quantiles(p95s))
    candidate_effect = {}
    for candidate in candidate_indices:
        role = roles[candidate]; candidate_effect[role] = {}
        for key, value in data.items():
            against = []
            for zero in zero_indices:
                delta = ((value[candidate, query + 1:steps + 1] - value[candidate, query])
                         - (value[zero, query + 1:steps + 1] - value[zero, query]))
                against.append(dict(zero=roles[zero], effect_p95=p95_abs(delta)))
            zero_floor = float(np.median([
                item['post_query_displacement_p95']['p95']
                for item in pairwise[key]['pairs']]))
            effects = [item['effect_p95']['p95'] for item in against]
            declared_floor = float(FIELD_FLOORS.get(key, 1e-3))
            denominator = max(zero_floor, declared_floor)
            candidate_effect[role][key] = dict(
                against_zero=against, zero_pair_median_p95=zero_floor,
                declared_field_floor=declared_floor, ratio_denominator=denominator,
                effect_to_zero_median=quantiles(np.asarray(effects) / denominator))
    return dict(zero_pair_noise=pairwise, candidate_effect_vs_zero=candidate_effect)


def tcc_values(packet, reference, encoder, device):
    import sys
    sys.path[:0] = [str(ROOT), str(TASK / 'src')]
    import torch
    from consequence_evaluator.provenance import sha as provenance_sha
    from consequence_evaluator.reference_bank import BANK_SCHEMA, load_reference_features
    from consequence_evaluator.reference_progress import trajectory_features
    from consequence_evaluator.temporal_phase import ENCODER_SCHEMA, LearnedReferenceProgress
    bank_manifest = json.loads((Path(reference) / 'manifest.json').read_text())
    encoder_manifest = json.loads((Path(encoder) / 'manifest.json').read_text())
    checkpoint = Path(encoder_manifest['checkpoint'])
    if (bank_manifest.get('schema') != BANK_SCHEMA
            or encoder_manifest.get('schema') != ENCODER_SCHEMA
            or encoder_manifest.get('status') != 'COMPLETED'
            or encoder_manifest.get('reference_sha256') != bank_manifest.get('reference_sha256')
            or provenance_sha(checkpoint) != encoder_manifest.get('checkpoint_sha256')):
        raise ValueError('frozen physical reference-bank/encoder contract mismatch')
    references, _ = load_reference_features(reference, bank_manifest)
    matcher = LearnedReferenceProgress(
        references, torch.load(checkpoint, map_location='cpu', weights_only=False), device)
    values = []; progress_start = []
    for pose, hand in zip(np.asarray(packet['object_pose']), np.asarray(packet['hand_keypoints'])):
        trace = matcher.align(trajectory_features(pose, hand, np.asarray(packet['timestamps'])))
        query = int(packet['query_tick']); end = query + int(packet['horizon'])
        values.append(float(trace['progress'][end] - trace['progress'][query]))
        progress_start.append(float(trace['progress'][query]))
    roles = list(packet['roles']); zeros = list(packet['zero_roles'])
    zero_values = [values[roles.index(role)] for role in zeros]
    return dict(values=values, progress_start=progress_start,
                progress_start_range=float(max(progress_start) - min(progress_start)),
                zero_value_mean=float(np.mean(zero_values)),
                zero_value_mad=float(np.median(np.abs(np.asarray(zero_values) - np.median(zero_values)))),
                candidate_contrast_vs_zero={role: float(values[roles.index(role)] - np.mean(zero_values))
                                            for role in packet['candidate_roles']},
                reference_sha256=bank_manifest['reference_sha256'],
                encoder_checkpoint_sha256=encoder_manifest['checkpoint_sha256'])


def reset_contract(diagnostics, roles):
    physical_keys = ('_contact_forces', '_dof_state', '_hist_obs', '_rigid_body_state',
                     '_root_states', '_tar_contact_forces', 'contact_reset', 'data_id',
                     'progress_buf', 'ref_index', 'start_times')
    failures = []
    if not isinstance(diagnostics, list) or len(diagnostics) != len(roles):
        return dict(physical_keys=list(physical_keys), passed=False,
                    failures=[dict(check='diagnostic_role_coverage')],
                    derived_buffers_allowed=['_curr_obs', 'rew_buf', '_reset_ig'])
    if {item.get('role') for item in diagnostics} != set(roles):
        failures.append(dict(check='diagnostic_role_set'))
    for item in diagnostics or []:
        role = item.get('role')
        if item.get('reset_frame_unchanged') is not True:
            failures.append(dict(role=role, check='frame_unchanged'))
        if item.get('reset_rng_equal') is not True:
            failures.append(dict(role=role, check='rng_anchor_equal'))
        state = item.get('state_max_abs') or {}
        for key in physical_keys:
            if float(state.get(key, float('inf'))) != 0.:
                failures.append(dict(role=role, check='reset_state_exact', field=key,
                                     max_abs=float(state.get(key, float('inf')))))
    return dict(physical_keys=list(physical_keys), passed=not failures, failures=failures,
                derived_buffers_allowed=['_curr_obs', 'rew_buf', '_reset_ig'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--encoder', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    packet_path, packet, roles, zero_roles, candidate_roles, query, horizon, steps, fields = load_packet(args.packet)
    if args.output_dir.exists() or not str(args.output_dir.resolve()).startswith(str((ROOT / 'outputs/consequence-evaluator').resolve())):
        raise ValueError('fresh task-owned audit directory required')
    args.output_dir.mkdir(parents=True)
    contract = action_contract(packet, roles, zero_roles, candidate_roles, query, horizon)
    contract['passed'] = bool(all(contract[key] for key in (
        'control_prefix_exact', 'control_suffix_exact', 'zero_replay_exact',
        'done_exact', 'done_no_early', 'candidate_action_contract_exact',
        'requested_residual_contract_exact',
        'teacher_action_hash_exact', 'schedule_hash_exact')))
    reset = reset_contract(packet.get('reset_diagnostics'), roles)
    audit_contract_passed = bool(contract['passed'] and reset['passed'])
    result = dict(schema='ref2dex.consequence-gate1.gpu-serial-cluster-audit.v1',
                  engineering_only=True, training_allowed=False,
                  packet=str(packet_path), packet_sha256=sha(packet_path),
                  roles=roles, teacher_role=packet['teacher_role'], zero_roles=zero_roles,
                  candidate_roles=candidate_roles, execution_order=packet['execution_order'],
                  schedule_sha256=packet.get('schedule_sha256'), query_tick=query,
                  horizon=horizon, steps=steps, outcome_complete=bool(packet.get('outcome_complete')),
                  source_backend=packet.get('source_backend'), replay_identity=packet.get('replay_identity'),
                  contract=contract,
                  reset_contract=reset,
                  audit_contract_passed=audit_contract_passed,
                  reset_diagnostics=packet.get('reset_diagnostics'),
                  teacher_max_lift_m=float(np.max(np.asarray(packet['object_pose'])[0, :, 2, 3]
                                                 - np.asarray(packet['object_pose'])[0, 0, 2, 3])),
                  role_max_lift_m={role: float(np.max(np.asarray(packet['object_pose'])[index, :, 2, 3]
                                                      - np.asarray(packet['object_pose'])[index, 0, 2, 3]))
                                  for index, role in enumerate(roles)},
                  noise=noise_table(packet, roles, zero_roles, candidate_roles, query, steps, fields),
                  note=('descriptive fixed-control residual diagnostic; zero arms are paired noise '
                        'samples within one process, not independent twins or a strict Gate1'))
    if (args.reference is None) != (args.encoder is None):
        raise ValueError('--reference and --encoder must be supplied together')
    if args.reference is not None:
        result['tcc'] = tcc_values(packet, args.reference.resolve(), args.encoder.resolve(), args.device)
    (args.output_dir / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('contract', 'teacher_max_lift_m', 'role_max_lift_m')}, indent=2))


if __name__ == '__main__':
    main()
