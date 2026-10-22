"""Fixed-budget v6 stage evaluation on synthetic injections or saved native traces.

No model/optimizer/simulator is constructed. All outputs must be Task-local.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/wam-rl-policy-v3/src')]
from ptv3_shared_wam_v6.staged_progress import StagedProgress, StagedDenseReward, STAGES, STAGE_CONTRACT, REWARD_CONTRACT


def distribution(values):
    return dict(n=len(values), **dict(zip(('min', 'p25', 'median', 'p75', 'max'),
        np.percentile(values, [0, 25, 50, 75, 100]).tolist()))) if values else dict(n=0, min=None, p25=None, median=None, p75=None, max=None)


def summarize(records):
    result = {}
    for index, stage in enumerate(STAGES):
        times = [r['stages']['first_time_s'][stage] for r in records if r['stages']['achieved'][stage]]
        durations = []
        for row in records:
            stamps = row['stages']['first_time_s']
            if stamps[stage] is not None:
                origin = 0. if index == 0 else stamps[STAGES[index-1]]
                durations.append(stamps[stage]-origin)
        result[stage] = dict(successes=len(times), total=len(records), not_observed=len(records)-len(times),
                             first_passage_s=distribution(times), from_previous_stage_s=distribution(durations))
    return result


def demo():
    records = []
    for case in ('positive', 'lift_without_contact', 'contact_without_lift',
                 'intermittent_hold', 'false_release_attached', 'invalid_force'):
        stages = StagedProgress(0.)
        reward = None
        for tick in range(151):  # fixed 150 controls / 5s, six injections
            touching = case != 'lift_without_contact'
            valid = case != 'invalid_force'
            z = .05 if tick >= 2 and case != 'contact_without_lift' else 0.
            if case == 'intermittent_hold' and tick % 15 == 0:
                valid = False
            released = tick >= 40 and case in ('positive', 'false_release_attached')
            attached = touching and (not released or case == 'false_release_attached')
            feedback = SimpleNamespace(force_valid=valid, workspace_ok=True, contact_observed=attached,
                grasp_established=attached, grasp_lost=False, supported=False, released=released)
            features = SimpleNamespace(palm_gap_m=.02, mean_tip_gap_m=.01,
                hand_clearance_m=.04 if released else .01)
            stages.update(z, tick/30, features, feedback, np.ones((5, 3)) if attached else np.zeros((5, 3)),
                          release_authorized=released)
            if tick == 0:
                reward = StagedDenseReward(stages)
            else:
                decision = SimpleNamespace(task_complete=False, terminated=False, failure_reason=None)
                reward.step(stages, decision, np.zeros(12), terminal=tick == 150)
        records.append(dict(sequence=case, observed_controls=150, stages=stages.snapshot(),
                            reward_parts_sum=reward.totals))
    return records


def capture(panel, path):
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_scheduler import TaskSchedulerConfig
    from wam_rl_policy_v3.task_spec import TaskSpec, Phase
    from wam_rl_policy_v3.measured_reward_bridge import reward_features
    from wam_rl_policy_v3.robot_adapter import InspireAdapter
    from wam_rl_policy_v3.teacher_quality import table_support
    if panel['hz'] != 30 or panel['controls_per_episode'] != 600 or len(panel['sequence_order']) != 6:
        raise ValueError('registered six-episode / 600-control / 30Hz panel required')
    data_path = path/'trajectory.npz'
    meta = json.loads((path/'manifest.json').read_text())
    if meta['sequence_order'] != panel['sequence_order']:
        raise ValueError('capture ordering differs from fixed panel')
    with np.load(data_path) as data, np.load(ROOT/panel['initial_state_capture']/'trajectory.npz') as initial:
        for key in ('q', 'dq', 'physical', 'objects', 'hand', 'table_pose', 'contact_forces', 'contact_force_valid'):
            if data[key].shape[:2] != (601, 6):
                raise ValueError('complete fixed-budget capture required: '+key)
            if not np.allclose(data[key][0], initial[key][0], rtol=0, atol=1e-7):
                raise ValueError('fixed measured initial state differs: '+key)
        if not np.array_equal(data['tick'], np.repeat(np.arange(601)[:, None], 6, axis=1)):
            raise ValueError('missing, duplicate or non-30Hz capture ticks')
        teacher = ROOT/panel['geometry_source']
        geometry = json.loads((teacher/'manifest.json').read_text())
        records = []
        for index, name in enumerate(panel['sequence_order']):
            row = next(r for r in geometry['records'] if r['sequence'] == name)
            with np.load(teacher/row['path']/'canonical.npz') as z:
                points = z['points'].copy()
            side = ROOT/panel['declarations']/'sequences'/name
            with np.load(side/'admission.npz') as z:
                bounds = z['table_bounds'].copy()
            spec = TaskSpec.from_packet(json.loads((side/'raw-task.json').read_text()))
            controller = TaskController(spec, points, scheduler_config=TaskSchedulerConfig(episode_timeout_s=20.))
            stages = StagedProgress(data['objects'][0, index, 2])
            for tick in range(601):
                obj = data['objects'][tick, index]
                pose = np.eye(4);pose[:3, :3] = Rotation.from_quat(obj[3:7]).as_matrix();pose[:3, 3] = obj[:3]
                hand = InspireAdapter.canonical_hand(data['hand'][tick, index])
                support = table_support(pose[None], points, data['table_pose'][tick, index], bounds,
                                        data['object_contact_force'][tick, index][None])[0][0]
                forces = data['contact_forces'][tick, index]
                observed = controller.observe(pose, hand, obj[7:13], tick/30, contact_forces=forces,
                    force_valid=bool(data['contact_force_valid'][tick, index]), supported=bool(support))
                decision = observed['decision']
                features = reward_features(pose, hand, points, decision, obj[7:13], forces)
                stages.update(obj[2], tick/30, features, observed['feedback'], forces,
                    release_authorized=decision.phase == Phase.RELEASE or 'release_authorized' in controller.scheduler.paid_events)
                if decision.terminated:
                    break
            records.append(dict(sequence=name, observed_controls=tick, stages=stages.snapshot(),
                full_task_success=bool(decision.task_complete), terminal_reason=decision.failure_reason,
                right_censored=not decision.terminated, post_terminal_archive_states_ignored=600-tick,
                reward_evaluated=False))
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo', action='store_true')
    p.add_argument('--panel', type=Path, default=TASK/'configs/staged-eval-panel-v1.json')
    p.add_argument('--capture', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if TASK not in output.parents or output.exists():
        raise ValueError('fresh Task-local output required')
    panel = json.loads(args.panel.read_text())
    records = demo() if args.demo else capture(panel, args.capture or ROOT/panel['initial_state_capture'])
    report = dict(status='PASS', mode='synthetic_injections' if args.demo else 'archived_native_trace',
        stage_contract=STAGE_CONTRACT, reward_contract=REWARD_CONTRACT,
        python=sys.executable, python_version=sys.version.split()[0], numpy=np.__version__,
        conda_prefix=os.environ.get('CONDA_PREFIX'), gpu_allocated=False, optimizer_updates=0,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        implementation_state='S2c working tree at execution; implementation diff committed with this receipt',
        panel=panel if not args.demo else dict(episodes=6, controls_per_episode=150, hz=30),
        maximum_controls=900 if args.demo else 3600, new_physics_controls=0,
        summary=summarize(records), records=records,
        limitation='Engineering evidence only. Release stage is detachment after hold, not complete ordered path/placement success. Archive initial states predate the production reset fix; hidden native state and new policy success are not certified.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], mode=report['mode'], successes={k:v['successes'] for k,v in report['summary'].items()}, episodes=6)))


if __name__ == '__main__':
    main()
