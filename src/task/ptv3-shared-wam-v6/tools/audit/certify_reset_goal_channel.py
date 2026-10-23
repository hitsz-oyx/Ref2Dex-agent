#!/usr/bin/env python3
"""Exercise reset-bound artifact generation; join existing certification to r7.

Does not relabel, train, or repeat the prior raw-data provenance investigation.
The synthetic collector exercise is never counted as historical coverage.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT / 'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.goal_channel import (
    ResetWaypointChannel, bind_source, instruction_from_displacement,
)


def coverage(rows, admitted):
    """Exact ordered identity join; unknown records remain unknown."""
    expected = [(int(r['source_id']), int(r['sequence_id']), int(r['slot']), int(r['tick'])) for r in admitted]
    actual = [(int(r['source_id']), int(r['sequence_id']), int(r['slot']), int(r['tick'])) for r in rows]
    if expected != actual or len(set(actual)) != len(actual):
        raise ValueError('certification rows must match frozen ordered windows exactly')
    missing = Counter()
    for row in rows:
        missing.update(x for x in row['missing'].split(';') if x)
    # This audit consumes the previous audit's unproven records, not a new
    # certification scheme for arbitrary CSVs declaring themselves legal.
    if any(r['waypoint_status'] != 'UNCERTAIN_MISSING_RECORDS' for r in rows):
        raise ValueError('new evidence discovered; inspect before changing coverage')
    return dict(windows=len(rows), certified_legal=0, uncertain=len(rows),
                proven_self_future=0, source_role_explicit_hits=sum(r['source_role'] != 'UNKNOWN' for r in rows),
                available_before_reset_proven_hits=sum(r['available_before_reset'] == 'True' for r in rows),
                missing_window_counts=dict(missing), ordered_join='PASS',
                by_source=dict(Counter(str(r['source_id']) for r in rows)))


def synthetic_collector(output):
    # A NEW synthetic execution. Nominal constructor anchor is intentionally
    # distinct from the measured pose; this is a mechanism example, not a
    # retrofit of the three historical Shadow episodes.
    reset = dict(execution_id='synthetic-new-execution', reset_id='reset-0', clock_id='synthetic-clock', at_ns=1000000000)
    def source(payload, role, kind, name):
        return bind_source(payload, source_role=role, source_kind=kind, source_id=name,
                           reset=reset, observed_at_ns=900000000, available_at_ns=950000000)
    nominal = np.eye(4); nominal[2, 3] = .7
    measured = nominal.copy(); measured[0, 3] = .002
    query = np.zeros((32, 3)); query[:, 0] = np.linspace(-.02, .02, 32)
    query[:, 1] = np.tile([-.01, .01], 16)
    instruction = instruction_from_displacement(
        source(nominal, 'task_instruction', 'nominal_constructor', 'nominal-object-start'),
        source(dict(rule='fixed_world_translation', translation_m=[0, 0, .2], target_stage='task_completion'),
               'task_instruction', 'declared_task', 'declared-displacement'),
        source(query, 'reset_geometry', 'declared_task', 'query-ids'),
        source(np.zeros((16, 3)), 'reset_geometry', 'declared_task', 'interaction-surface'),
        reset=reset, object_slot=0, hand_slot=0, goal_type='position_only')
    deployed = ResetWaypointChannel(instruction, expected_reset=reset, artifact_dir=output / 'synthetic-execution')
    persisted = json.loads((output / 'synthetic-execution/instruction.json').read_text())
    replay = ResetWaypointChannel(persisted, expected_reset=reset)
    receipts = []
    for tick in range(4):
        stamp = reset['at_ns'] + round(tick * 1e9 / 30)
        args = dict(tick=tick, observed_at_ns=stamp, available_at_ns=stamp, decision_at_ns=stamp)
        live, receipt = deployed.update(measured, np.zeros((11, 3)), **args)
        train, replay_receipt = replay.update(measured, np.zeros((11, 3)), **args)
        assert receipt == replay_receipt
        assert all(live[k].tobytes() == train[k].tobytes() for k in live)
        size = deployed.written_bytes
        retry, retry_receipt = deployed.update(measured, np.zeros((11, 3)), **args)
        assert retry_receipt == receipt and deployed.written_bytes == size
        assert all(live[k].tobytes() == retry[k].tobytes() for k in live)
        assert receipt['goal_pose'] == instruction['task_goal_pose']
        assert receipt['goal_pose'] != receipt['planner_waypoint_diagnostic']['goal_pose']
        receipts.append(receipt)
    persisted_receipts = [json.loads(line) for line in (output / 'synthetic-execution/waypoints.jsonl').read_text().splitlines()]
    assert persisted_receipts == receipts
    return dict(status='PASS', synthetic=True, new_physical_execution=False,
                historical_windows_certified=0, decisions=len(receipts),
                persisted_instruction_and_receipts_replayed_bit_exact=True,
                condition_quantity='task_level_goal', target_stage='task_completion',
                task_goal_distinct_from_planner_waypoint=True, exact_retries_no_duplicate_writes=True,
                nominal_anchor_preserved=True, artifact_bytes=deployed.written_bytes)


def run(output, prior_dir, curated_path):
    if output.exists():
        raise FileExistsError('preserve previous run: ' + str(output))
    free = shutil.disk_usage(output.parent).free
    if free < 20 * 2**30:
        raise RuntimeError('less than 20GiB available')
    start = time.monotonic()
    with (prior_dir / 'window-certification.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    curated = json.loads(curated_path.read_text())
    admitted = [r for r in curated['selection']['train_window_rows'] if r.get('admitted') is True]
    counts = coverage(rows, admitted)
    prior = json.loads((prior_dir / 'receipt.json').read_text())
    if counts['windows'] != prior['counts']['total_windows']:
        raise ValueError('prior counts disagree')
    mothers = json.loads((prior_dir / 'mother-provenance.json').read_text())
    native = [m for m in mothers if 'native_goal_position_evidence' in m]
    output.mkdir()
    fields = ['ordinal', 'source_id', 'sequence_id', 'slot', 'tick', 'source_role',
              'available_before_reset', 'target_reset_binding', 'waypoint_status', 'missing']
    with (output / 'coverage.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    result = dict(schema='ref2dex.reset-goal-channel.audit.v2',
        experiment_id='P-20261023-reset-bound-goal-channel', run_id=output.name,
        status='BLOCKED_ZERO_CERTIFIED_HISTORICAL_COVERAGE',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                         platform=platform.platform(), device='CPU provenance/state-machine only'),
        prior_certification=str(prior_dir), curated_manifest=str(curated_path), counts=counts,
        native_shadow_candidates=dict(mothers=len(native), windows=sum(m['windows'] for m in native),
            certified_waypoints=0, missing='original execution/reset and scheduler binding'),
        synthetic_generation_chain=synthetic_collector(output),
        new_raw_provenance_investigation=False, training_updates=0, new_physical_rollouts=0,
        comparison_executed=False, task_goal_minus_unknown_mm=None,
        disk_free_before_bytes=free, elapsed_s=time.monotonic()-start,
        cost_envelope=json.loads((TASK / 'docs/audit/brief32-cost-envelope-20261023.json').read_text())['new_recording_scale'])
    (output / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prior-dir', type=Path, default=TASK / 'docs/audit/brief35-source-role-20261023-r1')
    parser.add_argument('--curated', type=Path, default=ROOT / 'outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json')
    args = parser.parse_args()
    result = run(args.output, args.prior_dir, args.curated)
    print(json.dumps(dict(status=result['status'], counts=result['counts'],
                         synthetic_generation_chain=result['synthetic_generation_chain'])))


if __name__ == '__main__':
    main()
