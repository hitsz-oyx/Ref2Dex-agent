"""Locate approach failure in already captured native trajectories; array statistics only."""
import argparse
import json
from pathlib import Path

import numpy as np


def first_tick(condition):
    ticks = np.flatnonzero(condition)
    return int(ticks[0]) if len(ticks) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    rows = []
    for index, name in enumerate(('control-airplane', 'control-sphere', 'control-mug')):
        source = a.source / name / 'native'
        manifest = json.loads((source / 'manifest.json').read_text())
        audit = json.loads((source / 'independent_audit.json').read_text())
        assert manifest['status'] == 'COMPLETED' and audit['pass']
        assert manifest['execution_steps'] == 1 and manifest['no_teacher_intervention']
        with np.load(source / 'trajectory.npz') as z:
            q, pd, gap, obj = (z[k].copy() for k in ('q', 'pd_targets', 'surface_gap', 'obj'))
            hand = z['hand'].copy()
        initial_error = {k: float(np.abs(value[0, 2] - value[0, 0]).max())
                         for k, value in [('q', q), ('hand', hand), ('object', obj)]}
        if max(initial_error.values()) > 1e-6:
            raise ValueError('Initial teacher/student states differ')
        error = np.linalg.norm(q[:, 2, :3] - q[:, 0, :3], axis=-1)
        near = first_tick(gap[:, 0] <= .03)
        lift = obj[:, 0, 2, 3] - obj[0, 0, 2, 3]
        ticks = np.arange(len(q))
        row = dict(sequence=manifest['native_input']['sequence'], source=str(source),
            initial_state_max_error=initial_error,
            first_action_PD_XYZ_error_mm=float(np.linalg.norm(pd[0, 2, :3] - pd[0, 0, :3])*1000),
            initial_teacher_surface_gap_m=float(gap[0, 0]),
            teacher_first_near3cm_tick=near,
            teacher_first_lift3cm_tick=first_tick(lift >= .03),
            student_first_near3cm_tick=first_tick(gap[:, 2] <= .03),
            student_min_surface_gap_m=float(gap[:, 2].min()),
            first_wrist_deviation5cm_tick=first_tick(error >= .05),
            first_wrist_deviation10cm_tick=first_tick(error >= .10),
            wrist_XYZ_deviation_at_teacher_near_m=float(error[near]) if near is not None else None,
            wrist_XYZ_deviation_m={str(t): float(error[t]) for t in (1, 4, 8, 16, 32)})
        rows.append(row)
        axes[index, 0].plot(ticks/30, gap[:, 0], label='Teacher')
        axes[index, 0].plot(ticks/30, gap[:, 2], label='Student shared7k')
        axes[index, 0].axhline(.03, color='gray', ls=':', label='Near 3 cm')
        axes[index, 0].set_ylabel(name[len('control-'):] + '\nsurface gap (m)')
        axes[index, 1].plot(ticks/30, error, color='tab:red', label='Student vs teacher root XYZ')
        axes[index, 1].set_ylabel('Root position deviation (m)')
        for ax in axes[index]:
            if near is not None:
                ax.axvline(near/30, color='gray', ls='--', label='Teacher first near')
            ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8);axes[0, 1].legend(fontsize=8)
    fig.savefig(a.output / 'approach.png', dpi=140)
    report = dict(scope='Descriptive timing in three existing K1 val episodes; later teacher/student states diverge; no action-error or cross-hand causal claim',
                  cpu_only=True, neural_inference=False, rows=rows)
    (a.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
