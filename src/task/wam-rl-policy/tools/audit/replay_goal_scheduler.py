"""Bounded CPU geometry replay of ordered goals; never runs a robot policy."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from wam_rl_policy.goal_scheduler import OrderedGoalScheduler, SchedulerConfig, transform_points


def poses_from_record(values):
    result = np.repeat(np.eye(4)[None], len(values), axis=0)
    result[:, :3, :3] = Rotation.from_quat(values[:, 3:7]).as_matrix()
    result[:, :3, 3] = values[:, :3]
    return result


def interpolate_index(arc, index):
    return float(np.interp(index, np.arange(len(arc)), arc))


def replay(name, reference, ref_time, points, observed, live_time, *,
           hands=None, ready=None, lost=None, metadata=None):
    scheduler = OrderedGoalScheduler(reference, ref_time, points)
    rows = []
    ref_time = np.asarray(ref_time)-ref_time[0]
    for i, (pose, now) in enumerate(zip(observed, live_time)):
        row = scheduler.update(pose, now, execution_ready=None if ready is None else bool(ready[i]),
                               lost_control=False if lost is None else bool(lost[i]))
        clock_index = min(int(np.searchsorted(ref_time, now+scheduler.config.horizon_s)), len(reference)-1)
        clock_position, clock_rotation, clock_point = scheduler._errors(pose, reference[clock_index])
        row.update(time_s=float(now), object_pose=pose.tolist(), clock_goal_pose=reference[clock_index].tolist(),
            clock_goal_point_m=clock_point, clock_goal_position_m=clock_position,
            clock_goal_rotation_rad=clock_rotation,
            clock_lead_m=float(scheduler.arc[clock_index]-scheduler.arc[row['projected_index']]),
            adaptive_lead_m=interpolate_index(scheduler.arc, row['target_reference_index'])-float(scheduler.arc[row['projected_index']]),
            hand=None if hands is None else hands[i].tolist())
        row['goal_pose'] = row['goal_pose'].tolist()
        row['goal_delta'] = row['goal_delta'].tolist()
        rows.append(row)
    config = scheduler.config
    # Entire-output geometry checks, including timeout/recovery outputs.
    assert all(r['goal_position_m'] <= config.max_goal_translation_m+1e-7 for r in rows)
    assert all(r['goal_rotation_rad'] <= config.max_goal_rotation_rad+1e-7 for r in rows)
    assert all(r['goal_point_m'] <= config.max_goal_point_m+1e-7 for r in rows)
    assert np.all(np.diff([r['confirmed_index'] for r in rows]) >= 0)
    return dict(name=name, metadata=metadata or {}, config=asdict(config), points=points.tolist(),
        reference=reference.tolist(), milestones=scheduler.milestones.tolist(), rows=rows,
        summary=dict(frames=len(rows), complete=rows[-1]['complete'], timed_out=rows[-1]['timed_out'],
            confirmed_progress=rows[-1]['confirmed_progress'], modes=dict(Counter(r['mode'] for r in rows)),
            clock_goal_point_p95_mm=float(np.percentile([r['clock_goal_point_m'] for r in rows], 95)*1000),
            adaptive_goal_point_p95_mm=float(np.percentile([r['goal_point_m'] for r in rows], 95)*1000),
            clock_goal_cap_violation_frames=sum(r['clock_goal_point_m'] > config.max_goal_point_m for r in rows),
            max_clock_lead_mm=max(r['clock_lead_m'] for r in rows)*1000,
            max_adaptive_lead_mm=max(r['adaptive_lead_m'] for r in rows)*1000))


def synthetic_cases():
    points = np.array([[.05, 0, 0], [-.05, 0, 0], [0, .05, 0], [0, -.05, 0], [0, 0, .05], [0, 0, -.05]])
    result = []
    for kind in ('prepare_lift', 'out_and_back', 'rotation', 'recovery', 'stalled'):
        n = 181
        ref = np.repeat(np.eye(4)[None], n, axis=0)
        if kind == 'rotation':
            ref[:, :3, :3] = Rotation.from_euler('z', np.linspace(0, np.pi, n)).as_matrix()
        elif kind == 'out_and_back':
            ref[:, 0, 3] = np.r_[np.linspace(0, .15, 91), np.linspace(.15, 0, 91)[1:]]
        else:
            ref[40:, 2, 3] = np.linspace(0, .15, n-40)
        indices = np.r_[np.zeros(30, int), np.arange(n), np.full(20, n-1)]
        if kind == 'stalled':
            indices = np.zeros(541, int)
        actual = ref[indices].copy()
        ready = np.ones(len(indices), bool)
        lost = np.zeros(len(indices), bool)
        if kind == 'prepare_lift':
            ready[:65] = False
        if kind == 'stalled':
            ready[:] = False
        if kind == 'recovery':
            actual[110:125, 1, 3] += .12
            lost[110:125] = True
        result.append(replay('synthetic / '+kind, ref, np.arange(n)/60, points, actual,
            np.arange(len(indices))/60, ready=ready, lost=lost,
            metadata=dict(kind='synthetic_geometry', intervention=kind)))
    return result


def native_cases(source, cache, count):
    identities = json.loads((source/'environment-objects.json').read_text())
    lookup = {r['object_code']: r['geometry'] for r in json.loads((cache/'manifest.json').read_text())['records']}
    with np.load(source/'last-episode.npz') as data:
        obj, hand, clock, reset = [data[k].copy() for k in ('object', 'hand', 'time', 'reset')]
    if reset[1:-1].any() or not np.isfinite(obj).all() or not np.isfinite(hand).all():
        raise ValueError('finite contiguous recorded episodes required')
    # Deterministic geometry coverage, not a success-selected or representative panel.
    displacement = np.linalg.norm(obj[-1, :, :3]-obj[0, :, :3], axis=-1)
    candidates = np.argsort(-displacement, kind='stable')
    selected, seen = [], set()
    for env in candidates:
        code = identities[env]['object_code']
        if code in seen:
            continue
        seen.add(code)
        selected.append(int(env))
        if len(selected) >= count:
            break
    result = []
    for env in selected:
        identity = identities[env]
        with np.load(lookup[identity['object_code']]) as geometry:
            if (str(geometry['object_code'].item()) != identity['object_code']
                    or abs(float(geometry['scale'].item())-identity['scale']) > 1e-7):
                raise ValueError('recorded object/CAD identity mismatch')
            surface = geometry['surface'][np.linalg.norm(geometry['normals'], axis=-1) > 1e-6]
        points = surface[np.linspace(0, len(surface)-1, 32, dtype=int)].copy()
        ref = poses_from_record(obj[:, env])
        dt = float(np.median(np.diff(clock)))
        for variant in ('nominal', 'delayed_half_speed'):
            speed, delay = (1., 0.) if variant == 'nominal' else (.5, .6)
            live_time = np.arange(0, delay+(len(ref)-1)*dt/speed+.4, dt)
            indices = np.clip(np.floor(np.maximum(live_time-delay, 0)*speed/dt+1e-6).astype(int), 0, len(ref)-1)
            result.append(replay(identity['object_code'].split('/')[-1].split('-')[0]+' / '+variant,
                ref, clock, points, ref[indices], live_time, hands=hand[indices, env],
                metadata=dict(kind='recorded_geometry_replay', source=str(source), environment=env,
                    identity=identity, variant=variant, ready_signal='unknown',
                    frame_indices=indices.tolist(), scope='known recorded path; time-warped observations, no new physics')))
    return result


def render_overview(cases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    selected = cases[:4]+[c for c in cases[4:] if 'half_speed' in c['name']][:2]
    for ax, case in zip(axes.flat, selected):
        rows = case['rows']
        t = [r['time_s'] for r in rows]
        ax.plot(t, [r['clock_goal_point_m']*1000 for r in rows], label='Clock target gap')
        ax.plot(t, [r['goal_point_m']*1000 for r in rows], label='State-based target gap')
        ax.axhline(50, color='grey', ls='--', lw=.8)
        ax.set_title(case['name'], fontsize=10)
        ax.set_xlabel('Replay time (s)')
        ax.set_ylabel('Mean paired-point gap (mm)')
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Goal selection geometry only: recorded path replay, no policy or physical execution')
    fig.tight_layout()
    fig.savefig(output/'overview.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objects', type=int, default=4)
    parser.add_argument('--seconds', type=float, default=180)
    args = parser.parse_args()
    if not 1 <= args.objects <= 8 or not 0 < args.seconds <= 600:
        raise ValueError('bounded object/time budget required')
    output = args.output.resolve()
    if output.exists() or ROOT/'outputs/wam-rl-policy' not in output.parents:
        raise ValueError('fresh task-local output required')
    import shutil
    if shutil.disk_usage(ROOT).free < 20*2**30:
        raise RuntimeError('less than 20GiB free')
    output.mkdir(parents=True)
    scratch = ROOT/'tmp'/output.name
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(scratch/'matplotlib')
    import matplotlib
    started = time.monotonic()
    manifest = dict(status='RUNNING', experiment_id='P-20261018-goal-scheduler', run_id=output.name,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        python=sys.executable, python_version=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, matplotlib=matplotlib.__version__, hostname=platform.node(),
        conda_prefix=os.environ.get('CONDA_PREFIX'), pid=os.getpid(),
        source=str(args.source.resolve()), cache=str(args.cache.resolve()), device='cpu',
        device_reason='pure NumPy/SciPy pose scheduling; no neural inference, training or simulation',
        seconds_budget=args.seconds, output_budget_mib=100, objects=args.objects,
        scope='geometry engineering Probe only; no WAM/control/task-success claim')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    cases = synthetic_cases()
    print(json.dumps(dict(stage='synthetic', seconds=time.monotonic()-started)), flush=True)
    cases += native_cases(args.source, args.cache, args.objects)
    if time.monotonic()-started > args.seconds:
        raise TimeoutError('geometry replay budget')
    (output/'replay.json').write_text(json.dumps(cases, separators=(',', ':'), allow_nan=False))
    template = (TASK/'tools/audit/goal_scheduler_viewer.html').read_text()
    payload = json.dumps(cases, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (output/'replay.html').write_text(template.replace('__REPLAY_DATA__', payload))
    render_overview(cases, output)
    manifest.update(status='COMPLETED', seconds=time.monotonic()-started,
        cases=len(cases), frames=sum(len(c['rows']) for c in cases),
        geometry_audit='PASS', results=[dict(name=c['name'], **c['summary']) for c in cases])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if sum(p.stat().st_size for p in output.iterdir()) > 100*2**20:
        raise RuntimeError('output budget exceeded; artifacts retained')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
