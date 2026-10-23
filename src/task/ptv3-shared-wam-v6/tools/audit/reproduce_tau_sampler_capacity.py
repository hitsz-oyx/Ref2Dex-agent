#!/usr/bin/env python3
"""Replay the loader capacity fault without model/data loading or GPU work."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v6.curated_sampling import CuratedTauSampler, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before-source', type=Path, required=True)
    parser.add_argument('--horizon', type=int, default=4008)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location('sampler_before', args.before_source)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    manifest = load_manifest(args.manifest)
    old = before.CuratedTauSampler(manifest, seed=170, max_per_sequence=8)
    new = CuratedTauSampler(manifest, seed=170, max_per_sequence=8)
    started = time.monotonic()
    failures = []
    original_totals = {s: 0 for s in new.sources}
    repaired_totals = dict(original_totals)
    for step in range(1, args.horizon + 1):
        for rank in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([170, step, rank, 20261023]))
            chosen = rng.choice(new.sources, size=96, replace=True)
            counts = {s: int((chosen == s).sum()) for s in new.sources}
            excess = {s: n for s, n in counts.items() if n > new.source_capacities[s]}
            rng = np.random.default_rng(np.random.SeedSequence([170, step, rank, 20261023]))
            repaired = new._draw_source_counts(96, rng)
            assert sum(repaired.values()) == 96
            assert all(n <= new.source_capacities[s] for s, n in repaired.items())
            if excess:
                failures.append(dict(step=step, rank=rank, original_overflows=excess,
                                     original=counts, repaired=repaired))
            else:
                assert counts == repaired
            for s in new.sources:
                original_totals[s] += counts[s]
                repaired_totals[s] += repaired[s]
    actual = []
    for fault in failures:
        try:
            old.draw(96, step=fault['step'], rank=fault['rank'])
        except ValueError as exc:
            error = str(exc)
        else:
            raise AssertionError('before-source did not reproduce the allocation fault')
        batch = new.draw(96, step=fault['step'], rank=fault['rank'])
        sequence_counts = {}
        for row in batch:
            key = row['sequence_key']
            sequence_counts[key] = sequence_counts.get(key, 0) + 1
        assert len(batch) == 96 and max(sequence_counts.values()) <= 8
        assert batch == new.draw(96, step=fault['step'], rank=fault['rank'])
        actual.append(dict(step=fault['step'], rank=fault['rank'], old_error=error,
                           repaired_batch_rows=len(batch), max_per_sequence=max(sequence_counts.values())))
    valid = []
    for step, rank in [(1, 0), (128, 1), (1500, 2), (1570, 2)]:
        assert old.draw(96, step=step, rank=rank) == new.draw(96, step=step, rank=rank)
        valid.append(dict(step=step, rank=rank, all_rows_bit_exact=True))
    result = dict(status='PASS',scope='CPU sampler only; not evidence of optimizer updates',
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  before_source_sha256=hashlib.sha256(args.before_source.read_bytes()).hexdigest(),
                  horizon=args.horizon,ranks=3,batch_per_rank=96,seed=170,
                  source_capacities=new.source_capacities, allocation_draws=args.horizon*3,
                  overflow_allocations=failures, actual_sampler_replays=actual,
                  original_totals=original_totals,repaired_totals=repaired_totals,
                  total_absolute_count_change=sum(abs(original_totals[s]-repaired_totals[s]) for s in new.sources),
                  feasible_row_draws_bit_exact=valid,elapsed_s=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
