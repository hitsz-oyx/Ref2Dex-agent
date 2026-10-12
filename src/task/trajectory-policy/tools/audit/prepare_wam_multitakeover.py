"""Aggregate actual student-state expert takeover futures; pure rows have PD only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit')]
from audit_wam_ref2_shadow import sha, write
from trajectory_policy.retargeter_data import features


def own_recovery_rows(data, takeover, controls):
    """Own expert future begins at the measured student state at takeover."""
    ticks = np.arange(takeover, controls-23)
    if any(not data['expert_applied'][t:t+24, 1].all()
           or data['expert_clipped'][t:t+24, 1].any() for t in ticks):
        raise ValueError('every causal future control must actually be expert/unclipped')
    if (not np.array_equal(data['applied'][takeover:, 3], data['expert_action'][takeover:, 1])
            or not np.array_equal(data['pd_targets'][takeover:, 3], data['expert_PD'][takeover:, 1])):
        raise ValueError('own future expert controls and PD labels differ')
    return ticks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, nargs=3, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    owner = ROOT/'outputs/trajectory-policy'
    out = a.output.resolve()
    if out.exists() or out.parent != owner or not out.name.startswith('wam-ref2-multitakeover-'):
        raise ValueError('fresh task-owned aggregation output required')
    natives = [v.resolve() for v in a.native]
    reference_path = owner/'competent-reference-20261011-r1/reference.npz'
    with np.load(reference_path, allow_pickle=False) as f:
        reference = f['obj']
    evidence = {str(reference_path): sha(reference_path), str(Path(__file__)): sha(__file__),
                str(TASK/'src/trajectory_policy/retargeter_data.py'): sha(TASK/'src/trajectory_policy/retargeter_data.py')}
    arrays = []
    admissions = []
    model_identities = []
    for source_id, native in enumerate(natives):
        if owner not in native.parents:
            raise ValueError('task-owned audited sources required')
        m, r, audit = [json.loads((native/k).read_text()) for k in ('manifest.json', 'result.json', 'root_recovery_audit.json')]
        group = json.loads((native.parent/'group_status.json').read_text())
        for path in [native/k for k in ('manifest.json', 'result.json', 'root_recovery_audit.json', 'trajectory.npz')]+[native.parent/'group_status.json']:
            evidence[str(path)] = sha(path)
        digest = sha(native/'trajectory.npz')
        controls, takeover = m['controls'], m['takeover_tick']
        if (takeover not in (8, 16, 32) or controls != 128 or m.get('status') != 'COMPLETED'
                or r.get('status') != 'COMPLETED' or group.get('status') != 'COMPLETED'
                or group.get('controls') != controls or group.get('takeover_tick') != takeover
                or m.get('training_allowed') is not True or m.get('expert_PD_training_allowed') is not True
                or m.get('mixed_policy_future_hand_training_allowed') is not False
                or audit.get('pass') is not True or audit.get('checks_max') != 0
                or audit.get('native_data_sha256') != digest or m.get('trajectory_sha256') != digest
                or m.get('result_sha256') != sha(native/'result.json')
                or audit.get('controls') != controls or audit.get('takeover_tick') != takeover):
            raise ValueError('completed fresh128/takeover/root-audit chain required')
        identity = m['retargeter_identity']
        if identity.get('jointly_trained_pair') is not True:
            raise ValueError('same screened exact learned WAM/R pair required')
        model_identities.append((identity['head_sha256'], identity['execution_R_sha256']))
        with np.load(native/'trajectory.npz', allow_pickle=False) as f:
            d = {k: f[k] for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'pd_targets', 'expert_PD',
                                  'applied', 'expert_action', 'expert_applied', 'expert_clipped',
                                  'learner_hand_history', 'learner_object_history', 'learner_goal')}
        expert_ticks = own_recovery_rows(d, takeover, controls)
        pure_ticks = np.flatnonzero(~d['expert_clipped'][:, 0])
        records = []
        for role, ticks, valid in ((3, expert_ticks, True), (2, pure_ticks, False)):
            for tick in ticks:
                query = role-2
                past = np.maximum(tick+np.arange(-3, 1), 0)
                if (not np.array_equal(d['learner_hand_history'][tick, query], d['hand'][past, role])
                        or not np.array_equal(d['learner_object_history'][tick, query], d['obj'][past, role])
                        or not np.array_equal(d['learner_goal'][tick, query], reference[min(tick+24, 542)])):
                    raise ValueError('own observed history / known goal differs')
                future = d['hand'][tick+np.arange(1, 25), role] if valid else np.broadcast_to(d['hand'][tick, role], (24, 11, 3))
                state, tau = features(*(d[k][tick, role] for k in ('q', 'dq', 'hand', 'obj', 'velocity')), future)
                records.append(dict(source_id=np.asarray(source_id), tick=np.asarray(tick), role=np.asarray(role),
                                    state87=state, tau24x33=tau, trajectory_valid=np.asarray(valid),
                                    current_q=d['q'][tick, role], expert_PD=d['expert_PD'][tick, query],
                                    hand_history=d['hand'][past, role], object_history=d['obj'][past, role],
                                    planned_object_goal=reference[min(tick+24, 542)],
                                    actual_applied=d['applied'][tick, role], actual_PD=d['pd_targets'][tick, role],
                                    measured_next_hand=d['hand'][tick+1, role], measured_next_object=d['obj'][tick+1, role]))
        arrays += records
        admissions.append(dict(native=str(native), source_id=source_id, source_trajectory_sha256=digest,
                               takeover=takeover, expert_future_role=3, expert_ticks=expert_ticks.tolist(),
                               pure_current_PD_role=2, pure_ticks=pure_ticks.tolist(),
                               pure_future_training_allowed=False, mixed_policy_future_training_allowed=False,
                               actual_expert_future_training_allowed=True))
    if sorted(v['takeover'] for v in admissions) != [8, 16, 32] or len(set(model_identities)) != 1:
        raise ValueError('three distinct takeover times and identical policy pair required')
    packed = {k: np.stack([v[k] for v in arrays]) for k in arrays[0]}
    if (any(not np.isfinite(v).all() for v in packed.values())
            or packed['tau24x33'][~packed['trajectory_valid']].any()
            or any(sha(p) != h for p, h in evidence.items())):
        raise ValueError('nonfinite, pure-future leakage or input drift')
    out.mkdir()
    np.savez_compressed(out/'aggregation.npz', **packed)
    manifest = dict(schema='ref2dex.wam-multitakeover-aggregation.v1', status='COMPLETED',
                    experiment_id='P-20261012-goal-point-wam-ref22-multitakeover', run_id=out.name,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    input_sha256=evidence, admissions=admissions, model_identities=model_identities,
                    samples=len(arrays), true_expert_future_samples=int(packed['trajectory_valid'].sum()),
                    pure_current_PD_samples=int((~packed['trajectory_valid']).sum()),
                    aggregation_sha256=sha(out/'aggregation.npz'), training_allowed=True,
                    scope='development TRAIN aggregation, not held-out evaluation; GT futures from actual expert takeover row only')
    write(out/'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('input_sha256', 'admissions')}))


if __name__ == '__main__':
    main()
