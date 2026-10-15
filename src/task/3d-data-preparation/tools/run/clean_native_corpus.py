"""Freeze source lineage, normalize CAD normals and filter native H4/K24 windows."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
REPO = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from pose_mesh_data.cleaning import (CLEANING_SCHEMA, CleaningPolicy, clean_sequence,
    normalize_geometry, reconcile_files, window_ticks)


def write_json(path, value):
    part = path.with_suffix('.partial.json')
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    part.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save_npz(path, values):
    part = path.with_suffix('.partial.npz')
    np.savez_compressed(part, **values)
    part.replace(path)


def footprint(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def build(source, output, config_path, *, workers=4, max_seconds=1800, artifact_budget_bytes=2_000_000_000):
    source, output = source.resolve(), output.resolve()
    if (REPO/'outputs'/TASK.name not in output.parents or output.exists()
            or not 1 <= workers <= 8 or not 60 <= max_seconds <= 7200
            or not 0 < artifact_budget_bytes <= 4_000_000_000):
        raise ValueError('fresh Task output and bounded workers/time/artifacts required')
    if shutil.disk_usage(output.parent).free < 20*2**30:
        raise RuntimeError('less than 20GiB free')
    config = json.loads(config_path.read_text())
    policy = CleaningPolicy(**config['policy'])
    meta = json.loads((source/'manifest.json').read_text())
    if meta.get('schema') != 'ref2dex.pose-mesh.v1' or meta.get('training_allowed') is not False:
        raise ValueError('provisional native input required')
    groups = {s: {r['group_id'] for r in meta['sequences'] if r['split'] == s} for s in ('train', 'dev')}
    if groups['train'] & groups['dev']:
        raise ValueError('input source groups overlap TRAIN/DEV')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('geometry', 'sequences', 'sources'):
        (output/name).mkdir()
    started = time.monotonic()
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    write_json(output/'input-manifest.json', meta)
    write_json(output/'policy.json', config)
    progress = dict(status='RUNNING', phase='lineage', pid=os.getpid(), run_id=output.name,
        git_commit=git_commit, input=str(source), policy=str(config_path.resolve()), workers=workers,
        max_seconds=max_seconds, artifact_budget_bytes=artifact_budget_bytes,
        device='CPU: file/array/geometry processing only; no neural computation', training_allowed=False)

    def checkpoint(phase, processed=0, total=None):
        elapsed = time.monotonic()-started
        progress.update(phase=phase, processed=processed, total=total, elapsed_s=elapsed, checked_at=time.time(),
                        free_bytes=shutil.disk_usage(output).free)
        if processed and total:
            progress['estimated_remaining_seconds'] = elapsed*(total-processed)/processed
        write_json(output/'progress.json', progress)
        if elapsed > max_seconds or progress['free_bytes'] < 20*2**30 or (output/'STOP').exists():
            raise RuntimeError('cleaning deadline/free-space/STOP boundary reached')
        if footprint(output) > artifact_budget_bytes:
            raise RuntimeError('cleaning artifact budget exceeded')

    try:
        checkpoint('lineage')
        origins, source_audits = {}, []
        for i, source_dir in enumerate(meta['source_manifests']):
            root = Path(source_dir).resolve()
            original = json.loads((root/'manifest.json').read_text())
            saved = 'sources/source-%03d.json' % i
            write_json(output/saved, original)
            heldout_path = root/'heldout-test-manifest.json'
            heldout = json.loads(heldout_path.read_text()) if heldout_path.exists() else None
            if heldout is not None:
                write_json(output/('sources/source-%03d-test.json' % i), heldout)
            if (root/'sequences').exists():
                files = reconcile_files(root, original, heldout)
                if files['missing'] or files['unclaimed']:
                    raise ValueError('unreconciled native file catalog: '+str(root))
                source_audits.append(dict(source_manifest=str(root/'manifest.json'), **files))
            for local_id, row in enumerate(original['sequences']):
                path = str((root/row['path']).resolve())
                if path in origins:
                    raise ValueError('duplicate source payload lineage')
                origins[path] = dict(row=row, source_manifest=str(root/'manifest.json'),
                    source_manifest_snapshot=saved, source_row=local_id,
                    parent_source_manifest=original.get('source_manifest'), source_clock=original.get('source_clock'))
        accepted_geometry, geometry_ledger, bad_geometry = [], [], set()
        for i, row in enumerate(meta['objects']):
            path = (source/row['path']).resolve()
            try:
                cleaned, report = normalize_geometry(load_npz(path), policy)
            except ValueError as error:
                bad_geometry.add(row['key'])
                geometry_ledger.append(dict(key=row['key'], source_path=str(path), status='QUARANTINED', reason=str(error)))
                continue
            relative = 'geometry/object-%05d.npz' % i
            save_npz(output/relative, cleaned)
            accepted_geometry.append({**row, 'path':relative, 'source_path':str(path), 'normal_repair':'unit-norm-preserve-direction.v1'})
            geometry_ledger.append(dict(key=row['key'], status='CLEANED', source_path=str(path), path=relative, **report))
            if i % 128 == 0:
                checkpoint('geometry', i, len(meta['objects']))
        write_json(output/'geometry-ledger.json', geometry_ledger)

        def process(item):
            i, row = item
            sid = str(row['source_id'])
            if sid not in config['sources']:
                raise ValueError('source has no explicit cleaning profile')
            path = (source/row['path']).resolve()
            origin = origins.get(str(path))
            if origin is None:
                raise ValueError('live row has no source-local lineage')
            for field in ('source_id', 'profile', 'group_id', 'objects', 'target_slot', 'split', 'format'):
                if row[field] != origin['row'][field]:
                    raise ValueError('live/source metadata mismatch: '+field)
            base = dict(input_row=i, source_id=row['source_id'], split=row['split'], source_path=str(path))
            if set(row['objects']) & bad_geometry:
                return dict(**base, status='QUARANTINED', reason='ambiguous_geometry',
                    bad_geometry=sorted(set(row['objects']) & bad_geometry)), None, []
            try:
                before = path.stat()
                values = load_npz(path)
                sequence, quality, report = clean_sequence(values, row, config['sources'][sid], policy)
                ticks, windows = window_ticks(sequence, quality, row['target_slot'], policy)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('source changed during read')
                if not len(ticks):
                    return dict(**base, status='QUARANTINED', reason='no_clean_windows', **report, windows=windows), None, []
                payload = path
                new_format = row['format']
                if report['changed_arrays']:
                    payload = output/('sequences/sequence-%05d.npz' % i)
                    save_npz(payload, sequence)
                    new_format = 'canonical'
                stamp = payload.stat()
                cleaned_row = {**row, 'path':str(payload), 'format':new_format,
                    'source_path':str(path), 'source_manifest':origin['source_manifest'],
                    'source_manifest_snapshot':origin['source_manifest_snapshot'], 'source_row':origin['source_row'],
                    'parent_source_manifest':origin['parent_source_manifest'],
                    'source_sequence':row.get('source_sequence', row.get('source_episode_id', path.name)),
                    'source_clock_profile':config['sources'][sid], 'clock_verified':report['clock_verified'],
                    'training_allowed':False, 'payload_size_bytes':stamp.st_size, 'payload_mtime_ns':stamp.st_mtime_ns,
                    'clean_windows':len(ticks)}
                return dict(**base, status='CLEANED', **report, windows=windows), cleaned_row, ticks
            except ValueError as error:
                return dict(**base, status='QUARANTINED', reason=str(error)), None, []

        rows, counts = [], {}
        indices = {k: [] for k in ('all_train', 'all_dev', 'verified_train', 'verified_dev')}
        quarantine = Counter()
        ledger = output/'sequence-ledger.jsonl'
        total = len(meta['sequences'])
        with ledger.open('x') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, total, 256):
                batch = list(enumerate(meta['sequences'][offset:offset+256], start=offset))
                for record, row, ticks in pool.map(process, batch):
                    handle.write(json.dumps(record, ensure_ascii=False)+'\n')
                    sid = str(record['source_id'])
                    c = counts.setdefault(sid, dict(input_sequences=0, retained_sequences=0, quarantined_sequences=0,
                        frames_retained=0, windows_train=0, windows_dev=0, verified_windows_train=0, verified_windows_dev=0,
                        rejected_windows=Counter(), repaired_sequence_payloads=0))
                    c['input_sequences'] += 1
                    if row is None:
                        c['quarantined_sequences'] += 1
                        quarantine[sid+':'+record['reason']] += 1
                        continue
                    idx = len(rows)
                    rows.append(row)
                    split = row['split']
                    if split not in ('train', 'dev'):
                        raise ValueError('TEST may not enter the live TRAIN/DEV corpus')
                    indices['all_'+split].extend((idx, int(t)) for t in ticks)
                    if row['clock_verified']:
                        indices['verified_'+split].extend((idx, int(t)) for t in ticks)
                        c['verified_windows_'+split] += len(ticks)
                    c['retained_sequences'] += 1
                    c['frames_retained'] += record['frames']
                    c['windows_'+split] += len(ticks)
                    c['repaired_sequence_payloads'] += int(record['changed_arrays'])
                    c['rejected_windows'].update(record['windows']['rejected'])
                handle.flush()
                checkpoint('sequences', min(offset+256, total), total)
                print(json.dumps(progress), flush=True)
        save_npz(output/'windows.npz', {k:np.asarray(v, np.int64).reshape(-1, 2) for k, v in indices.items()})
        used = {k for row in rows for k in row['objects']}
        result = dict(schema='ref2dex.pose-mesh.v1', cleaning_schema=CLEANING_SCHEMA, status='COMPLETED',
            units='m', fps=30, hand_order=['right', 'left'], training_allowed=False, run_id=output.name,
            git_commit=git_commit, input_snapshot='input-manifest.json', policy='policy.json',
            source_profiles=config['sources'], window_policy=policy.to_dict(), window_index='windows.npz',
            window_index_size_bytes=(output/'windows.npz').stat().st_size,
            window_index_mtime_ns=(output/'windows.npz').stat().st_mtime_ns,
            objects=[r for r in accepted_geometry if r['key'] in used], sequences=rows,
            limitations=['Engineering cleaned snapshot, not scientific training admission.',
                'Native source profiles and target meanings remain explicit; no robot action labels manufactured.',
                'Clock-pending windows are excluded by the default CleanPoseMeshWindows reader.',
                'Source group splits preserved; object-disjoint claims require a separate evaluation panel.'])
        write_json(output/'manifest.pending.json', result)
        summary = dict(status='COMPLETED', training_allowed=False, git_commit=git_commit, run_id=output.name,
            source_counts=counts, geometry_files=len(accepted_geometry), quarantined_geometry=sorted(bad_geometry),
            retained_sequences=len(rows), input_sequences=total, quarantine_counts=dict(quarantine),
            windows={k:len(v) for k,v in indices.items()}, source_file_reconciliation=source_audits,
            elapsed_s=time.monotonic()-started, device=progress['device'])
        write_json(output/'cleaning-summary.json', summary)
        checkpoint('complete', total, total)
        progress.update(status='COMPLETED', artifact_bytes=footprint(output))
        write_json(output/'progress.json', progress)
        # Publish last: a failed final budget/deadline check cannot leave a
        # completed manifest that the cleaned reader would accept.
        (output/'manifest.pending.json').replace(output/'manifest.json')
        return summary
    except BaseException as error:
        progress.update(status='FAILED', error_type=type(error).__name__, reason=str(error), elapsed_s=time.monotonic()-started)
        write_json(output/'progress.json', progress)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=TASK/'configs/clean_native_v1.json')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seconds', type=int, default=1800)
    p.add_argument('--artifact-budget-bytes', type=int, default=2_000_000_000)
    a = p.parse_args()
    print(json.dumps(build(a.input, a.output, a.config, workers=a.workers, max_seconds=a.seconds,
        artifact_budget_bytes=a.artifact_budget_bytes), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
