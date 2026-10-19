"""Stream full available v5 contact caches, reporting source and split exposure."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'src'))
from ptv3_shared_wam_v5.contact_audit import (
    AUDIT_VERSION, SOURCE_NAMES, empty_counts, add_counts, contact_counts, describe_counts, future_exposure)
from ptv3_shared_wam_v5.contracts import LABEL_SEMANTICS
from ptv3_shared_wam_v5.storage import read_json, load_array


def write(path, value):
    pending = path.with_suffix('.json.part')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path); path.chmod(0o664)


def sequence_counts(job, plan_root, contacts, check):
    folder = contacts/job['key']; meta = read_json(folder/'meta.json')
    if (meta['label_semantics'] != LABEL_SEMANTICS or meta['key'] != job['key']
            or meta['source_id'] != job['source_id']):
        raise ValueError('contact label identity changed')
    c = load_array(folder/'contact.npy', mmap_mode='r')
    valid = load_array(folder/'valid.npy', mmap_mode='r')
    parts = load_array(folder/'part.npy', mmap_mode='r')
    if c.shape != valid.shape+(2048,) or valid.dtype != np.bool_ or parts.shape != c.shape:
        raise ValueError('unexpected contact cache shape/dtype')
    exposures = {split: future_exposure(load_array(plan_root/path), len(c), c.shape[1])
                 for split, path in job['windows'].items()}
    totals = {name: empty_counts() for name in ('cached_frames', 'train_uniform_windows', 'dev_uniform_windows')}
    for start in range(0, len(c), 16):
        check()
        sl = slice(start, start+16)
        add_counts(totals['cached_frames'], contact_counts(c[sl], valid[sl], parts[sl]))
        for split, exposure in exposures.items():
            add_counts(totals[split+'_uniform_windows'], contact_counts(c[sl], valid[sl], parts[sl], exposure[sl]))
    if totals['cached_frames']['object_frames'] != meta['valid_object_frames']:
        raise ValueError('contact validity disagrees with metadata')
    return dict(key=job['key'], source_id=job['source_id'], views=totals,
                masked_object_frames=int(valid.size-valid.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'contacts', 'output'): p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--git-commit', required=True); p.add_argument('--watch', action='store_true')
    p.add_argument('--seconds', type=int, default=10800)
    a = p.parse_args()
    if a.seconds <= 0: p.error('positive runtime required')
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); plan = read_json(a.plan)
    expected = Counter(j['source_id'] for j in plan['jobs'])
    manifest = dict(work_version=AUDIT_VERSION, git_commit=a.git_commit, run_id=a.output.name,
                    pid=os.getpid(), plan=str(a.plan.resolve()), contacts=str(a.contacts.resolve()),
                    seconds=a.seconds, min_free_gib=20, max_output_mib=128,
                    python=sys.version, numpy=np.__version__, device='CPU cache reductions; no model inference',
                    label_semantics=LABEL_SEMANTICS)
    write(a.output/'run-manifest.json', manifest)
    done = set(); sources = {}; last_write = 0

    def check():
        if time.monotonic()-started > a.seconds: raise InterruptedError('deadline')
        if (a.output/'STOP').exists(): raise InterruptedError('STOP')
        if shutil.disk_usage(a.output).free < 20*2**30: raise InterruptedError('free space')
        if ledger.exists() and ledger.stat().st_size > 128*2**20: raise InterruptedError('output budget')

    def report(status, reason=None):
        rows = {}
        for sid, source in sources.items():
            rows[str(sid)] = dict(source, name=SOURCE_NAMES[sid], expected_sequences=expected[sid],
                views={k: describe_counts(v) for k, v in source['views'].items()})
        value = dict(status=status, reason=reason, work_version=AUDIT_VERSION, sequences=len(done),
            expected_sequences=len(plan['jobs']), elapsed_s=time.monotonic()-started, checked_at=time.time(),
            sources=rows, scope='Full cached valid frame/object/query counts on admitted TRAIN/DEV sequences; no TEST.',
            window_exposure='TRAIN/DEV K24 labels counted with repeated appearances, uniform windows within source. Source probabilities and motion-stratum weights are NOT applied.',
            limitation='Source rates and coverage must be read together. Partial caches can be biased. No automatic BCE/teacher change.')
        write(a.output/'summary.json', value)
        print(json.dumps({k: value[k] for k in ('status', 'sequences', 'expected_sequences', 'elapsed_s')}), flush=True)

    ledger = a.output/'sequence-counts.jsonl'
    report('RUNNING')
    try:
        with ledger.open('w') as log:
            ledger.chmod(0o664)
            while len(done) < len(plan['jobs']):
                changed = False
                # listdir avoids tens of thousands of NAS stat calls per waiting pass.
                available = set(os.listdir(a.contacts))
                for job in plan['jobs']:
                    key = job['key']
                    if key in done or key not in available: continue
                    check()
                    if not (a.contacts/key/'meta.json').exists(): continue
                    row = sequence_counts(job, a.plan.parent, a.contacts, check)
                    source = sources.setdefault(job['source_id'], dict(sequences=0, masked_object_frames=0,
                        views={k: empty_counts() for k in row['views']}))
                    source['sequences'] += 1; source['masked_object_frames'] += row['masked_object_frames']
                    for view, counts in row['views'].items(): add_counts(source['views'][view], counts)
                    log.write(json.dumps(row, separators=(',', ':'))+'\n'); log.flush()
                    done.add(key); changed = True
                    if time.monotonic()-last_write > 20:
                        report('RUNNING'); last_write = time.monotonic()
                if len(done) == len(plan['jobs']): break
                check()
                progress = read_json(a.contacts/'progress.json')
                if not a.watch or (not changed and progress.get('status') in ('STOPPED', 'COMPLETED_FOR_AVAILABLE_HANDS')):
                    report('PARTIAL', 'remaining contact caches unavailable'); return
                report('WAITING_CONTACTS')
                for _ in range(30): check(); time.sleep(1)
        report('COMPLETED')
    except InterruptedError as exc:
        report('PARTIAL', str(exc))
    except Exception as exc:
        report('FAILED', type(exc).__name__+': '+str(exc)); raise


if __name__ == '__main__': main()
