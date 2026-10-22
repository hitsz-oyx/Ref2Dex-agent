"""Prepare a held-out executor panel from immutable measured robot rollouts.

This is CPU metadata/array inspection. It neither replays the teacher nor
certifies that stored actions remain executable in a newly initialized host.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def longest_true(values):
    values = np.asarray(values, dtype=bool)
    edges = np.flatnonzero(np.r_[False, values, False][1:] !=
                           np.r_[False, values, False][:-1])
    return int(max(edges[1::2] - edges[::2], default=0))


def reference_metrics(data, admission, controls, hz, lift_threshold):
    end = controls + 1
    lift = data['obj'][:end, 2, 3] - data['obj'][0, 2, 3]
    coupled = admission['coupled'][:end].astype(bool)
    supported = admission['supported'][:end].astype(bool)
    hold = coupled & ~supported & (lift >= lift_threshold)
    return dict(controls=controls, frames=len(lift),
                max_lift_m=float(lift.max()), final_lift_m=float(lift[-1]),
                longest_coupled_frames=longest_true(coupled),
                longest_lifted_coupled_frames=longest_true(hold),
                longest_lifted_coupled_seconds=longest_true(hold)/hz,
                lifted_coupled_frames=int(hold.sum()),
                unsupported_lift_frames=int(((lift >= lift_threshold) & ~supported).sum()),
                canonical_applied_max_error=float(np.abs(
                    data['action'][:controls]-data['applied'][:controls]).max()) if controls else 0.,
                maximum_object_linear_speed_mps=float(np.linalg.norm(data['velocity'][:end, :3], axis=-1).max()),
                contact_force_valid_fraction=float(data['contact_force_valid'][1:end].mean()) if controls else 0.)


def prepare(manifest_path, max_cases=16, min_controls=80, max_controls=300,
            min_hold_frames=15, lift_threshold=.05):
    manifest_path = manifest_path.resolve()
    meta = json.loads(manifest_path.read_text())
    if meta.get('schema') != 'ref2dex.teacher-measured-rollout-admission.v2' or meta.get('status') != 'COMPLETED':
        raise ValueError('completed measured robot admission required')
    train = [r for r in meta['records'] if r['split'] == 'train']
    heldout = [r for r in meta['records'] if r['split'] in ('val', 'dev')]
    train_names = {r['sequence'] for r in train}
    train_sources = {str(Path(r['source']).resolve()) for r in train}
    if train_names & {r['sequence'] for r in heldout}:
        raise ValueError('sequence membership overlaps TRAIN and held-out')
    if train_sources & {str(Path(r['source']).resolve()) for r in heldout}:
        raise ValueError('canonical rollout source overlaps TRAIN and held-out')
    candidates = []
    for row in sorted(heldout, key=lambda r: r['sequence']):
        source = Path(row['source']).resolve()
        sidecar = (manifest_path.parent/row['sidecar']).resolve()
        prefix = int(row['training_prefix_commands'])
        controls = min(prefix, max_controls)
        record = dict(sequence=row['sequence'], object=row['object'],
                      source=str(source), record=row,
                      admission=str(sidecar/'admission.npz'),
                      spec_path=str(sidecar/'task.json'),
                      reference_controls=controls,
                      reference_replay_status='NOT_RUN',
                      evaluation_only=True, training_allowed=False)
        try:
            with np.load(source/'trajectory.npz', allow_pickle=False) as data, np.load(sidecar/'admission.npz', allow_pickle=False) as admission:
                if len(data['action']) != len(data['obj'])-1:
                    raise ValueError('one action per state transition required')
                if not np.array_equal(data['frame_tick'], np.arange(len(data['obj']))):
                    raise ValueError('consecutive 30 Hz native control ticks required')
                if prefix > len(data['action']) or not admission['training_prefix'][:prefix].all():
                    raise ValueError('declared prefix does not match sidecar')
                if any(not np.isfinite(data[k]).all() for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'action', 'applied')):
                    raise ValueError('nonfinite measured robot trajectory')
                metrics = reference_metrics(data, admission, controls, 30., lift_threshold)
                record['reference_prefix_metrics'] = metrics
                record['reference_full_diagnostic_metrics'] = reference_metrics(
                    data, admission, len(data['action']), 30., lift_threshold)
                record['full_suffix_training_allowed'] = False
                record['spec'] = json.loads((sidecar/'task.json').read_text()) if prefix else None
                batch_path = source.parents[1]/'manifest.json'
                batch = json.loads(batch_path.read_text())
                record['collection_batch_manifest'] = str(batch_path)
                record['collection_seed'] = batch['seed']
                record['geometric_plan_root'] = batch['geometric_plan_root']
                reasons = []
                if row['engineering'] != 'PASS': reasons.append('engineering_rejected')
                if controls < min_controls: reasons.append('short_admitted_prefix')
                if metrics['longest_lifted_coupled_frames'] < min_hold_frames:
                    reasons.append('insufficient_continuous_lifted_coupling')
                if metrics['canonical_applied_max_error'] > 1e-6:
                    reasons.append('canonical_applied_mismatch')
                record['eligible'] = not reasons
                record['exclusion_reasons'] = reasons
        except (OSError, KeyError, ValueError) as error:
            record.update(eligible=False, exclusion_reasons=['unreadable_or_invalid_source'], error=repr(error))
        candidates.append(record)

    # Cover distinct objects before second recordings. Ordering never uses the
    # magnitude of lift, hold duration, prediction error or executor outcome.
    by_object = defaultdict(list)
    for record in candidates:
        if record['eligible']: by_object[record['object']].append(record)
    selected = []
    depth = 0
    while len(selected) < max_cases:
        next_rows = [by_object[obj][depth] for obj in sorted(by_object)
                     if len(by_object[obj]) > depth]
        if not next_rows: break
        selected.extend(next_rows[:max_cases-len(selected)])
        depth += 1
    counts = Counter(reason for row in candidates for reason in row['exclusion_reasons'])
    summary = dict(train_sequences=len(train), heldout_sequences=len(heldout),
                   heldout_nonempty_prefixes=sum(r['training_prefix_commands'] > 0 for r in heldout),
                   heldout_prefix_ge80=sum(r['training_prefix_commands'] >= 80 for r in heldout),
                   heldout_prefix_ge150=sum(r['training_prefix_commands'] >= 150 for r in heldout),
                   eligible_sequences=sum(r['eligible'] for r in candidates),
                   eligible_objects=len(by_object), selected_sequences=len(selected),
                   selected_objects=len({r['object'] for r in selected}),
                   selection_exclusions=dict(counts),
                   train_heldout_sequence_overlap=0, train_heldout_source_overlap=0,
                   selected_reference_prefix_controls=sum(r['reference_controls'] for r in selected))
    if selected:
        durations = [r['reference_prefix_metrics']['longest_lifted_coupled_seconds'] for r in selected]
        summary.update(selected_reference_hold_seconds_min=min(durations),
                       selected_reference_hold_seconds_median=float(np.median(durations)),
                       selected_reference_hold_seconds_max=max(durations))
    return dict(schema='ref2dex.wam-v6-executor-panel.v1', status='PREPARED_NOT_REPLAYED',
                experiment_id='P-20261022-wam-v6-executor-qualification',
                run_id='executor-qualification-20261022-r1', task='ptv3-shared-wam-v6',
                source_manifest=str(manifest_path), python=sys.executable,
                numpy=np.__version__, git_commit=subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], text=True).strip(),
                training_allowed=False, fps=30,
                selection=dict(min_controls=min_controls, max_controls=max_controls,
                               min_hold_frames=min_hold_frames, lift_threshold_m=lift_threshold,
                               max_cases=max_cases,
                               hold_definition='measured_coupled AND NOT measured_table_supported AND object_center_z - initial_z >= threshold',
                               ordering='round_robin_by_sorted_object_then_sorted_sequence',
                               prefix_scope='admitted safe prefix only; full suffix diagnostic metrics never used for selection'),
                unseen_scope='disjoint sequence and canonical source paths versus TRAIN of this robot admission; WAM pretraining lifetime exposure not certified',
                replay_scope='stored same-execution robot behavior only; new-host action replay, GT-tau executor qualification, prediction execution and perturbation recovery NOT_RUN',
                summary=summary, sequences=[r['sequence'] for r in selected],
                records=selected, candidates=candidates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-cases', type=int, default=16)
    p.add_argument('--min-controls', type=int, default=80)
    p.add_argument('--max-controls', type=int, default=300)
    p.add_argument('--min-hold-frames', type=int, default=15)
    p.add_argument('--lift-threshold-m', type=float, default=.05)
    a = p.parse_args()
    if not 1 <= a.max_cases <= 32 or not 1 <= a.min_controls <= a.max_controls <= 1200:
        raise ValueError('bounded panel of 1..32 sequences and up to 1200 controls required')
    if a.min_hold_frames < 1 or a.lift_threshold_m <= 0:
        raise ValueError('positive hold duration and lift threshold required')
    if a.output.exists(): raise FileExistsError('preserve existing panel: '+str(a.output))
    result = prepare(a.manifest, a.max_cases, a.min_controls, a.max_controls,
                     a.min_hold_frames, a.lift_threshold_m)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__': main()
