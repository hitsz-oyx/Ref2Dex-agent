#!/usr/bin/env python3
"""Read-only event provenance census and counterexamples; never train or edit inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v5.event_labels import annotate_events, proximity_counts


def read(path):
    return json.loads(Path(path).read_text())


def packet(arrays, tick, slot=0):
    return {k: np.asarray(arrays[k][tick, slot]).item()
            for k in ('event_type', 'event_delta_steps', 'event_valid')}


def census(parent, events, indices):
    by_key = {r['key']: r for r in events['sequences']}
    counts = Counter()
    samples = []
    for seq_id in np.unique(indices[:, 0]):
        row = parent['sequences'][int(seq_id)]
        selected = indices[indices[:, 0] == seq_id]
        folder = Path(by_key[row['key']]['path'])
        arrays = {k: np.load(folder/(k+'.npy'), mmap_mode='r')
                  for k in ('event_type', 'event_delta_steps', 'event_valid')}
        slots, ticks = selected[:, 1], selected[:, 2]
        kind, delta, valid = (arrays[k][ticks, slots] for k in arrays)
        counts['windows'] += len(selected)
        counts['raw_valid'] += int(valid.sum())
        counts['raw_invalid'] += int((~valid).sum())
        counts['valid_none'] += int((valid & (kind == 0)).sum())
        counts['valid_contact'] += int((valid & (kind == 1)).sum())
        counts['valid_release'] += int((valid & (kind == 2)).sum())
        counts['valid_nonzero_delta'] += int((valid & (kind != 0) & (delta > 0)).sum())
        counts['valid_delta_gt_24'] += int((valid & (delta > 24)).sum())
        for cat in ('none', 'contact', 'release', 'invalid'):
            eligible = (~valid if cat == 'invalid' else valid & (kind == ('none', 'contact', 'release').index(cat)))
            if eligible.any() and not any(v['category'] == cat for v in samples):
                at = int(np.flatnonzero(eligible)[0])
                samples.append(dict(category=cat, source_id=row['source_id'],
                    sample_id=selected[at].tolist(), **packet(arrays, int(ticks[at]), int(slots[at]))))
    return dict(counts=counts, examples=samples)


def counterexamples(parent, events, indices):
    # Real source record: reconstruct the original teacher annotation, then
    # alter ONLY frames >t while keeping record length fixed.
    by_key = {r['key']: r for r in events['sequences']}
    for seq_id, slot, tick in indices:
        row = parent['sequences'][int(seq_id)]
        if row['source_id'] != 1:
            continue
        folder = Path(by_key[row['key']]['path'])
        cached = {k: np.load(folder/(k+'.npy'), mmap_mode='r') for k in
                  ('event_type', 'event_delta_steps', 'event_valid')}
        if not cached['event_valid'][tick, slot] or cached['event_delta_steps'][tick, slot] <= 0:
            continue
        meta = read(folder/'meta.json')
        cf = Path(row['contact_path'])
        confidence = np.load(cf/'contact.npy', mmap_mode='r')
        validity = np.load(cf/'valid.npy', mmap_mode='r')
        enter, leave = proximity_counts(confidence, validity, meta['protocol'])
        original = annotate_events(enter, leave, validity, meta['protocol'], meta['observation_stride'])
        assert all(np.array_equal(original[k], cached[k]) for k in cached)
        altered = [np.asarray(v).copy() for v in (enter, leave, validity)]
        for value in altered:
            value[tick+1:] = value[tick]
        modified = annotate_events(*altered, protocol=meta['protocol'])
        truncated = annotate_events(enter[:tick+1], leave[:tick+1], validity[:tick+1], meta['protocol'])
        real = dict(sequence=row['key'], sample_id=[int(seq_id), int(slot), int(tick)],
            prefix_identical=True, full_recomputation_matches_stored_sidecar=True,
            original=packet(original, tick, slot), future_tail_held_at_current=packet(modified, tick, slot),
            truncated_at_t=packet(truncated, tick, slot))
        assert real['original'] != real['future_tail_held_at_current']
        break
    else:
        raise AssertionError('no real positive-delta source1 example')
    n, tick = 80, 5
    clear = np.zeros((n, 1), np.uint16)
    valid = np.ones((n, 1), bool)
    synth = {}
    for name, initial, tail in (('none', 0, 0), ('contact', 0, 3), ('release', 3, 0)):
        count = clear.copy()+initial
        count[12:] = tail
        synth[name] = packet(annotate_events(count, count, valid), tick)
    count = clear.copy()
    synth['known_none_length32'] = packet(annotate_events(count[:32], count[:32], valid[:32]), tick)
    synth['censored_none_length31'] = packet(annotate_events(count[:31], count[:31], valid[:31]), tick)
    # The sustained transition at 12 is only confirmed at 14, then backdated.
    count[12:] = 3
    full = annotate_events(count, count, valid)
    prefix = annotate_events(count[:13], count[:13], valid[:13])
    synth['backdated_state_at12_full'] = int(full['state'][12, 0])
    synth['state_at12_prefix_only'] = int(prefix['state'][12, 0])
    return dict(real_record=real, synthetic=synth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--curated-manifest', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    recipe = read(args.recipe)
    parent, events = read(recipe['parent_manifest']), read(recipe['event_manifest'])
    curated, anchor = read(args.curated_manifest), read(args.anchor)
    populations = {split: np.load(parent['indices'][split], mmap_mode='r') for split in ('train', 'dev')}
    populations['curated6000'] = np.array([[r['sequence_id'], r['slot'], r['tick']]
        for r in curated['selection']['train_window_rows'] if r.get('admitted') is True])
    anchor_rows = anchor.get('rows', anchor.get('windows'))
    if anchor_rows is None:
        raise ValueError('anchor row field missing')
    populations['anchor512'] = np.array([[r['sequence_id'], r['slot'], r['tick']] for r in anchor_rows])
    result = dict(schema='ref2dex.v6.event-condition-source-audit.v1',
        run_id='tau-contract-20261023-r1', task='ptv3-shared-wam-v6',
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        recipe=str(args.recipe.resolve()), event_manifest=recipe['event_manifest'],
        encoding=recipe['event_encoding'], dropout=recipe['event_dropout'],
        environment=dict(executable=sys.executable, python=sys.version, numpy=np.__version__, platform=platform.platform()),
        populations={name: census(parent, events, index) for name, index in populations.items()},
        counterexamples=counterexamples(parent, events, populations['train']),
        conclusion='UNUNIFIED_FUTURE_TEACHER_EVENTS_VS_CAUSAL_TASK_SCHEDULER', training_updates=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
