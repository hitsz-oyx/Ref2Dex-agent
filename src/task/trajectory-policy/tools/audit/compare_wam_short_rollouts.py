"""Compare measured closed-loop drift; teacher queries never become actor inputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'tools/audit'))
from audit_wam_ref2_shadow import sha, write


def summarize(native):
    m = json.loads((native/'manifest.json').read_text())
    r = json.loads((native/'result.json').read_text())
    audit = json.loads((native/'root_recovery_audit.json').read_text())
    if (m.get('status') != 'COMPLETED' or r.get('status') != 'COMPLETED'
            or audit.get('pass') is not True or audit.get('native_data_sha256') != sha(native/'trajectory.npz')
            or m.get('trajectory_sha256') != sha(native/'trajectory.npz')
            or m.get('result_sha256') != sha(native/'result.json')):
        raise ValueError('completed audited source required')
    with np.load(native/'trajectory.npz', allow_pickle=False) as f:
        d = {k: f[k] for k in ('hand', 'obj', 'learner_wam_tau', 'pd_targets', 'requested_pd',
                              'expert_PD', 'expert_clipped', 'clipped')}
    rows = []
    for tick in (0, 4, 8, 12, 16, 24, 32):
        if tick >= m['controls']:
            continue
        pd = d['pd_targets'][tick, 2]-d['expert_PD'][tick, 0]
        requested = d['requested_pd'][tick, 2]-d['expert_PD'][tick, 0]
        # Forecast's wrist displacement in CURRENT object frame vs actual own next state.
        displacement = (d['hand'][tick+1, 2, 0]-d['hand'][tick, 2, 0]) @ d['obj'][tick, 2, :3, :3]
        predicted = d['learner_wam_tau'][tick, 0, 0, :3]
        rows.append(dict(tick=tick, PD_xyz_error_mm=float(np.linalg.norm(pd[:3])*1000),
                         requested_PD_xyz_error_mm=float(np.linalg.norm(requested[:3])*1000),
                         own_wrist_h1_prediction_vs_measured_mm=float(np.linalg.norm(predicted-displacement)*1000),
                         wrist_deviation_from_same_trace_teacher_mm=float(np.linalg.norm(d['hand'][tick, 2, 0]-d['hand'][tick, 0, 0])*1000),
                         clipped=bool(d['clipped'][tick, 2]), expert_query_clipped=bool(d['expert_clipped'][tick, 0])))
    return dict(native=str(native), trajectory_sha256=sha(native/'trajectory.npz'),
                controls=m['controls'], takeover=m['takeover_tick'], checkpoints=m.get('retargeter_identity'),
                tick_diagnostics=rows, outcomes=r['outcomes'], local_grasp_calibration=r.get('local_grasp_calibration'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--previous', type=Path, default=ROOT/'outputs/trajectory-policy/wam-ref2-execution-startup-anchor-20261012-r1/native')
    a = p.parse_args()
    new, previous = a.new.resolve(), a.previous.resolve()
    owner = ROOT/'outputs/trajectory-policy'
    if owner not in new.parents or owner not in previous.parents:
        raise ValueError('task-owned sources required')
    report = dict(new=summarize(new), previous=summarize(previous),
                  scope='same-motion/reset descriptive development comparison; independent live PhysX launches, no causal validation')
    write(new/'short_rollout_comparison.json', report)
    for phase, value in (('previous', report['previous']), ('new', report['new'])):
        print(phase, json.dumps(value['tick_diagnostics']))
        print(phase, 'outcomes', json.dumps(value['outcomes']))


if __name__ == '__main__':
    main()
