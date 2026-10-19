"""Bounded immutable candidate sidecars from the already frozen contact caches."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v5.event_labels import (
    EVENT_LABEL_SCHEMA, DEFAULT_PROTOCOL, OBSERVATION_STRIDE_BY_SOURCE,
    proximity_counts, annotate_events, summarize_events)
from ptv3_shared_wam_v5.cache_contract import identity, metadata_identity
from ptv3_shared_wam_v5.storage import read_json, load_array


def write(path, value):
    pending = path.with_suffix('.json.part')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path); path.chmod(0o664)


def prepare_one(job):
    row, output, protocol, deadline = job
    root = Path(output)
    if time.time() > deadline or (root/'STOP').exists():
        raise InterruptedError('deadline/STOP')
    if shutil.disk_usage(root).free < 20*2**30:
        raise InterruptedError('free disk below20GiB')
    contact = Path(row['contact_path'])
    if metadata_identity(contact/'meta.json') != row['contact_meta_identity']:
        raise ValueError('contact metadata changed: '+row['key'])
    c = load_array(contact/'contact.npy', mmap_mode='r')
    v = load_array(contact/'valid.npy', mmap_mode='r')
    if c.shape != v.shape+(2048,) or v.shape[1] != len(row['objects']):
        raise ValueError('expected2048 queries and identical object slots')
    enter, leave = proximity_counts(c, v, protocol)
    stride = OBSERVATION_STRIDE_BY_SOURCE.get(row['source_id'], 1)
    arrays = annotate_events(enter, leave, v, protocol, stride)
    counts = summarize_events(arrays, v)
    # Fixed sensitivity comparison on each source-diverse panel sequence.
    sensitivity = {}
    for dwell in (2, 4):
        alt = annotate_events(enter, leave, v, dict(protocol, persistence=dwell), stride)
        sensitivity[str(dwell)] = summarize_events(alt, v)
    target = root/'sequences'/row['key']
    pending = target.with_name(target.name+'.partial')
    pending.mkdir(exist_ok=False)
    for name, array in dict(arrays, enter_count=enter, leave_count=leave, raw_valid=v).items():
        path = pending/(name+'.npy'); np.save(path, array); path.chmod(0o664)
    summary = dict(schema=EVENT_LABEL_SCHEMA, status='COMPLETED', key=row['key'],
                   source_id=row['source_id'], contact_meta_identity=row['contact_meta_identity'],
                   objects=row['objects'], shape=list(v.shape), protocol=protocol, counts=counts,
                   observation_stride=stride,
                   synthetic_frame_events='time-query between adjacent known real observations; C/state remain masked',
                   persistence_sensitivity=sensitivity)
    write(pending/'meta.json', summary)
    if target.exists():
        raise ValueError('never overwrite published sequence')
    pending.rename(target)
    return dict(key=row['key'], source_id=row['source_id'], path=str(target.resolve()),
                counts=counts, persistence_sensitivity=sensitivity,
                contact_meta_identity=row['contact_meta_identity'],
                event_meta_identity=metadata_identity(target/'meta.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--git-commit', required=True)
    p.add_argument('--per-source', type=int, default=16, help='0 means complete census')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seconds', type=int, default=7200)
    p.add_argument('--seed', type=int, default=177)
    a = p.parse_args()
    if not 1 <= a.workers <= 8 or a.seconds <= 0 or a.per_source < 0:
        p.error('bounded1..8 workers, positive seconds and nonnegative panel size required')
    a.output.mkdir(parents=True, exist_ok=False); (a.output/'sequences').mkdir()
    m = read_json(a.manifest); rows = m['sequences']
    if m.get('status') != 'FROZEN' or not m.get('training_allowed') or m.get('pending_sequences'):
        raise ValueError('complete frozen admitted parent required')
    if a.per_source:
        rng = np.random.default_rng(a.seed); selected = []
        for source in m['source_ids']:
            pool = [r for r in rows if r['source_id'] == source]
            selected.extend(pool[int(i)] for i in rng.choice(len(pool), min(len(pool), a.per_source), replace=False))
    else: selected = rows
    started = time.time(); protocol = dict(DEFAULT_PROTOCOL)
    run = dict(schema=EVENT_LABEL_SCHEMA, run_id=a.output.name, git_commit=a.git_commit,
               work_version='v5.proximity-event-candidate.v2', pid=os.getpid(), protocol=protocol,
               manifest=str(a.manifest.resolve()), parent_manifest_identity=metadata_identity(a.manifest),
               workers=a.workers, seconds=a.seconds, seed=a.seed, python=sys.version, numpy=np.__version__,
               device='CPU streaming cached confidences; no neural forward or KNN',
               expected_sequences=len(selected), full_parent_sequences=len(rows),
               observation_stride_by_source=OBSERVATION_STRIDE_BY_SOURCE,
               min_free_gib=20, max_output_gib=4,
               scope='candidate sidecars only; training_allowed=false pending annotation review')
    write(a.output/'run-manifest.json', run)
    done = []; sources = {}; last_report = 0

    def report(status, reason=None):
        write(a.output/'progress.json', dict(status=status, reason=reason, sequences=len(done),
            expected_sequences=len(selected), elapsed_s=time.time()-started,
            sources=sources, training_allowed=False))
        print(json.dumps(dict(status=status, sequences=len(done), total=len(selected),
                              elapsed_s=time.time()-started)), flush=True)

    report('RUNNING')
    try:
        # Pure CPU workers: no CUDA/NCCL initialized in this process.
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn')) as pool:
            jobs = ((r, str(a.output), protocol, started+a.seconds) for r in selected)
            # executor.map preserves input order and keeps manifest deterministic.
            with (a.output/'sequence-results.jsonl').open('x') as ledger:
                for row in pool.map(prepare_one, jobs, chunksize=4):
                    done.append(row); ledger.write(json.dumps(row)+'\n'); ledger.flush()
                    sid = str(row['source_id'])
                    count = sources.setdefault(sid, Counter())
                    count.update(row['counts'])
                    if count['frames']*20 > 4*2**30:
                        raise InterruptedError('output budget exceeded')
                    if time.time()-last_report >= 30:
                        report('RUNNING'); last_report=time.time()
        manifest = dict(schema=EVENT_LABEL_SCHEMA, status='COMPLETED', training_allowed=False,
            parent_manifest=str(a.manifest.resolve()), parent_manifest_identity=run['parent_manifest_identity'],
            protocol=protocol, sequences=done, sources=sources, git_commit=a.git_commit,
            observation_stride_by_source=OBSERVATION_STRIDE_BY_SOURCE,
            synthetic_frame_events='time-query between adjacent known real observations; C/state remain masked',
            scope='complete candidate census' if not a.per_source else 'source-diverse diagnostic panel',
            full_parent_sequences=len(rows), limitation='surface proximity, any-hand union; not force contact')
        manifest['identity'] = identity(manifest)
        write(a.output/'manifest.json', manifest); report('COMPLETED')
    except BaseException as error:
        report('FAILED', repr(error)); raise


if __name__ == '__main__': main()
