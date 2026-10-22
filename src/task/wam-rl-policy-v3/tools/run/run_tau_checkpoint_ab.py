"""Paired checkpoint intervention on the frozen v11 executor, without refitting.

Keep the original worker reviewable: emit its exact instrumentation diff before
executing it. Only the v6 inference package is refreshed, identically in both
arms, to support the checkpoint's explicit unknown condition policies.
"""
import argparse
import difflib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
PANEL = ROOT / 'outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1'
OLD = ROOT / 'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt'
NEW = ROOT / 'tmp/t38-r2/training/step001223.pt'


def refresh_inference(frozen):
    package = Path('src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6')
    # The stage scorer belongs to the frozen executor, not the intervention.
    stage = (frozen / package / 'staged_progress.py').read_bytes()
    shutil.copytree(ROOT / package, frozen / package, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    (frozen / package / 'staged_progress.py').write_bytes(stage)


def make_runtime(cls, checkpoint, stats, geometry):
    runtime = cls(checkpoint, stats, geometry, device='cuda:0', forecast_only=True)
    runtime.boundary = dict(calls=0, goal_valid=0, event_valid=0)
    encode = runtime.model.encode

    def checked_encode(batch):
        for key in ('object_goal_valid', 'goal_object_delta', 'goal_object_points',
                    'event_type', 'event_delta_steps', 'event_valid'):
            if batch[key].count_nonzero().item():
                raise ValueError('A/B unknown condition violated: ' + key)
        runtime.boundary['calls'] += 1
        return encode(batch)

    runtime.model.encode = checked_encode
    return runtime


def transformed_worker():
    original = (PANEL / 'frozen-worker.py').read_text()
    changes = [
        ('    copy_geometry_runtime(frozen)',
         '    copy_geometry_runtime(frozen)\n    refresh_inference(frozen)'),
        ("wam=OnlineV6Runtime(ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry',device='cuda:0',forecast_only=True)",
         "runtimes={arm:make_runtime(OnlineV6Runtime,path,ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry') for arm,path in [('OLD',OLD),('NEW',NEW)]};wam=runtimes['OLD']"),
        ("            for arm in job['order']:",
         "            for arm in job['order']:\n                wam=runtimes[arm]"),
        ("if arm in ('A1','A2'):", "if arm in ('OLD','NEW'):"),
        ("decision.local_goal_pose[None],[decision.next_event.wam_condition()]",
         "None,[dict(event_type=0,event_delta_steps=0.,event_valid=False,event_hz=30)]"),
        ("row.update(control_counts(control_info,actions));", 
         "row.update(tau_checkpoint=str(OLD if arm=='OLD' else NEW),tau_step=wam.step,condition_boundary=dict(wam.boundary));row.update(control_counts(control_info,actions));"),
        ("wam_seconds=wam.seconds", "wam_seconds={key:value.seconds for key,value in runtimes.items()}"),
    ]
    changed = original
    for before, after in changes:
        if changed.count(before) != 1:
            raise ValueError('frozen worker seam changed: ' + before)
        changed = changed.replace(before, after)
    return original, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--gpu', type=int, choices=[4, 5, 6], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    if len(registration['jobs']) != 20 or registration['maximum_controls'] != 24000:
        raise ValueError('exact registered 20-pair / 24000-control envelope required')
    if shutil.disk_usage(ROOT).free < 24 * 2**30:
        raise RuntimeError('4GiB output envelope plus 20GiB reserve required')
    original, changed = transformed_worker()
    # Output is made by the unchanged worker; write the reviewable diff beside it.
    diff = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                       fromfile=str(PANEL / 'frozen-worker.py'), tofile='paired-worker.py'))
    (args.output.parent / ('worker-%d.diff' % args.worker)).write_text(diff)
    emitted = args.output.parent / ('paired-worker-%d.py' % args.worker)
    emitted.write_text(changed)
    sys.argv = [str(PANEL / 'frozen-worker.py'), '--registration', str(args.registration),
                '--worker', str(args.worker), '--gpu', str(args.gpu), '--output', str(args.output)]
    namespace = dict(__name__='__main__', __file__=str(emitted),
                     refresh_inference=refresh_inference, make_runtime=make_runtime, OLD=OLD, NEW=NEW)
    exec(compile(changed, str(PANEL / 'frozen-worker.py'), 'exec'), namespace)


if __name__ == '__main__':
    main()
