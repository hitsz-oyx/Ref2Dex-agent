#!/usr/bin/env python3
"""Diagnose whether the curated recording can causally drive the deployed scheduler.

This is a bounded input-contract audit, not a reference-path reconstruction
heuristic. In particular, recorded object poses never become instructions.
Stop for inspection if a potential task-intent field is discovered.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v6.scheduler_cache_identity import (
    GOAL_SEMANTICS, SPARSE11_IN_21, condition_tensor_hash, scheduler_conditions_for_ticks,
)

INTENT_FIELDS = {
    'scheduler_reference_poses', 'reference_poses', 'reference_times_s',
    'scheduler_instruction', 'task_goal_pose', 'target_pose', 'goal_pose',
    'goal_object_pose', 'target_object_pose', 'planned_object_poses',
    'task_spec', 'object_path', 'instruction_producer',
}


def ambiguity_witness():
    """Two legal tasks, identical observations <=t, different scheduler outputs.

    These are independent synthetic instructions used to prove the missing
    input matters. Neither is admitted as a training condition.
    """
    live = np.broadcast_to(np.eye(4), (8, 1, 4, 4)).copy()
    joints = np.zeros((8, 1, 21, 3))
    query = np.zeros((32, 3))
    query[:, 0] = np.linspace(-.02, .02, 32)
    query[:, 1] = np.tile([-.01, .01], 16)
    geometry = dict(query=query, surface=np.zeros((16, 3)))
    conditions = []
    for sign in (-1, 1):
        instruction = live.copy()
        instruction[4:, 0, 0, 3] = sign * np.arange(1, 5) * .02
        sequence = dict(object_poses=live, joints=joints,
                        scheduler_reference_poses=instruction)
        conditions.append(scheduler_conditions_for_ticks(sequence, geometry, 0, 0, [3])[3])
    assert not np.array_equal(*conditions)
    # Also mutate ONLY observation tails, retaining the independent instruction.
    baseline = conditions[-1]
    for name in ('object_poses', 'joints'):
        changed = dict(sequence)
        changed[name] = changed[name].copy()
        changed[name][4:] = np.nan
        assert np.array_equal(baseline, scheduler_conditions_for_ticks(
            changed, geometry, 0, 0, [3])[3])
        changed[name] = sequence[name][:4]
        assert np.array_equal(baseline, scheduler_conditions_for_ticks(
            changed, geometry, 0, 0, [3])[3])
    return dict(tick=3, same_live_history=True, same_instruction_prefix=True,
                instruction_tail_is_independent_known_task_input=True,
                negative_x_mean_m=float(conditions[0][:, 0].mean()),
                positive_x_mean_m=float(conditions[1][:, 0].mean()),
                max_abs_condition_difference_m=float(np.max(np.abs(conditions[0]-conditions[1]))),
                condition_hashes=[condition_tensor_hash(v) for v in conditions],
                fixed_instruction_observation_tail_nan_and_truncation='PASS',
                training_admission=False)


def audit(curated_path):
    curated = json.loads(curated_path.read_text())
    parent_path = Path(curated['data_root']) / 'recipe/manifest.json'
    parent = json.loads(parent_path.read_text())
    rows = [r for r in curated['selection']['train_window_rows'] if r.get('admitted') is True]
    root_fields = (set(curated) | set(parent)) & INTENT_FIELDS
    if root_fields:
        raise ValueError('inspect discovered root task-intent fields: ' + str(sorted(root_fields)))
    groups = defaultdict(list)
    for row in rows:
        groups[int(row['sequence_id']), int(row['slot'])].append(row)
    inventories, excluded = [], []
    field_counts = Counter()
    source_counts = Counter()
    histories = Counter()
    for (seq_id, slot), group in sorted(groups.items()):
        record = parent['sequences'][seq_id]
        folder = Path(record['path'])
        metadata = json.loads((folder/'meta.json').read_text())
        arrays = {p.stem: p for p in folder.glob('*.npy')}
        found = (set(record) | set(metadata) | set(arrays)) & INTENT_FIELDS
        if found:
            raise ValueError('inspect discovered task-intent fields in ' + record['key'] + ': ' + str(sorted(found)))
        source_id = int(record['source_id'])
        source_counts[str(source_id)] += len(group)
        for field in arrays:
            field_counts[field] += len(group)
        # Read only causal state prefixes. These secondary counts do not
        # override the primary missing-intent blocker or fabricate a hand.
        end = max(int(row['tick']) for row in group) + 1
        obj_valid = np.load(arrays['object_pose_valid'], mmap_mode='r')[:end, slot]
        jvalid = np.load(arrays['joint_valid'], mmap_mode='r')[:end]
        ids = (0, 2, 5, 9, 13, 17, 4, 8, 12, 16, 20) if source_id == 4 else SPARSE11_IN_21
        obj_prefix = np.logical_and.accumulate(obj_valid)
        hand_prefix = np.logical_and.accumulate(jvalid[:, :, list(ids)].all(-1), axis=0)
        group_histories = Counter()
        for row in group:
            tick = int(row['tick'])
            state = dict(object_pose_valid_all_0_to_t=bool(obj_prefix[tick]),
                         hand11_valid_all_0_to_t_by_slot=hand_prefix[tick].tolist())
            if not state['object_pose_valid_all_0_to_t']:
                group_histories['invalid_pose_in_prefix'] += 1
            for hand_index, valid in enumerate(hand_prefix[tick]):
                if not valid:
                    group_histories[f'invalid_hand{hand_index}_in_prefix'] += 1
            if not hand_prefix[tick].any():
                group_histories['neither_hand_continuously_valid_from_reset'] += 1
            excluded.append(dict(source_id=source_id, sample_id=[seq_id, slot, tick],
                window_start_frame=tick, history_start_frame=tick-3,
                missing_runtime_inputs=['OrderedGoalScheduler.reference_poses (known task intent)'],
                reason='no task instruction or causal instruction-producer inputs in recording',
                causal_state=state))
        histories.update(group_histories)
        inventories.append(dict(sequence_id=seq_id, key=record['key'], slot=slot,
            source_id=source_id, window_count=len(group), path=str(folder),
            array_fields=sorted(arrays), record_fields=sorted(record),
            metadata_fields=sorted(metadata), discovered_intent_fields=[],
            secondary_causal_history_counts=dict(group_histories)))
    assert len(excluded) == len(rows)
    assert len({(r['source_id'], tuple(r['sample_id'])) for r in excluded}) == len(rows)
    return dict(
        schema='ref2dex.v6.scheduler-cache-identity-receipt.v2',
        status='BLOCKED_CAUSAL_TASK_INTENT_UNAVAILABLE',
        experiment_id='P-20261023-scheduler-cache-identities',
        task='ptv3-shared-wam-v6', run_id='tau-curated-20261023-r9',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        curated_manifest=str(curated_path.resolve()), parent_manifest=str(parent_path.resolve()),
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                         platform=platform.platform(), device='CPU metadata/NumPy geometry only', gpu_used=False),
        counts=dict(curated_admitted_windows=len(rows), sequence_slot_groups=len(groups),
            scheduler_eligible_windows=0, excluded_unrecomputable_windows=len(excluded),
            scheduler_condition_hash_unique_count=0, scheduler_condition_hash_duplicate_count=0,
            scheduler_condition_hash_repeat_rate=None, composite_identity_collisions=None,
            scheduler_vs_legacy_differing_windows=None, by_source=dict(source_counts)),
        missing_runtime_input_counts={'OrderedGoalScheduler.reference_poses (known task intent)': len(rows)},
        recorded_array_field_window_counts=dict(field_counts),
        secondary_causal_history_counts=dict(histories),
        contract=dict(goal_semantics=GOAL_SEMANTICS, horizon_steps=24, scheduler_hz=30,
            emits='one spatial waypoint pose/delta with 0.8s horizon; not a 24-frame predicted path',
            known_defaults=dict(reference_times_s='30Hz timebase if a matching instruction exists',
                goal_type='se3_pose', scheduler_config='SchedulerConfig()',
                execution_ready=None, lost_control=False, fast_geometry=True),
            additional_binding_needed='which recorded hand slot corresponds to the deployment hand11',
            missing_intent_alternatives=[
                'independent task instruction known before replay (need not be precomputed waypoints)',
                'a causal task-goal producer plus its runtime-only intent inputs'],
            prohibited='promoting observed object_poses to reference_poses, including their future tail'),
        gates=dict(coverage='FAIL', recorded_subset_bitwise_equivalence='NOT_RUN_NO_LEGAL_CONDITIONS',
            recorded_subset_no_lookahead='NOT_RUN_NO_LEGAL_CONDITIONS',
            helper_observation_tail_invariance='PASS_SYNTHETIC_FIXED_INSTRUCTION_ONLY',
            bounded_training_smoke='NOT_RUN_GATE_2_BLOCKED'),
        scheduler_vs_legacy=dict(status='UNDEFINED_NO_LEGAL_SCHEDULER_CONDITIONS',
            max_abs_delta=None, quantiles=None, direction_cosine=None,
            significant_inconsistencies=None, legal_planning_intent_explanation=None,
            legacy_hash_receipt='outputs/ptv3-shared-wam-v6/tau-curated-20261023-r8/scheduler-cache-identity-receipt.json'),
        causal_non_identifiability_witness=ambiguity_witness(),
        identities=[], excluded_windows=excluded, sequence_input_inventory=inventories,
        training_updates=0, checkpoint_bytes=0, config_status_modified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curated-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing receipt: ' + str(args.output))
    result = audit(args.curated_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=result['status'], counts=result['counts'],
                          causal_witness=result['causal_non_identifiability_witness'],
                          secondary_causal_history_counts=result['secondary_causal_history_counts'])))


if __name__ == '__main__':
    main()
