"""Freeze eight measured successful train230 robot trajectories for ref4_2."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]; ROOT = TASK.parents[2]
sys.path[:0] = [str(ROOT), str(TASK / 'src')]
from consequence_evaluator.contracts import HAND_LINKS, is_within
from consequence_evaluator.data import sha
from consequence_evaluator.historical_sources import verify_collection_sources
from consequence_evaluator.reference_bank import BANK_SCHEMA, BANK_ORIGIN, select_physical_references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); source = args.source.resolve()
    if out.exists() or not is_within(out, ROOT / 'outputs/consequence-evaluator'):
        parser.error('fresh owned bank output required')
    started = time.monotonic(); path = source / 'manifest.json'; raw = json.loads(path.read_text())
    live, historical = verify_collection_sources(raw['sources'], raw['git_commit'], ROOT)
    members, packets, frozen = select_physical_references(source, raw)
    frozen.update(live); frozen[str(path)] = sha(path); frozen[str(Path(__file__).resolve())] = sha(__file__)
    for name in ('reference_bank.py', 'reference_progress.py', 'reference_motion.py', 'contracts.py',
                 'value_outcomes.py', 'historical_sources.py', 'data.py', 'supervision.py'):
        file = TASK / 'src/consequence_evaluator' / name; frozen[str(file)] = sha(file)
    if any(sha(p) != h for p, h in frozen.items()):
        raise ValueError('physical-reference input drift')
    out.mkdir(parents=True)
    np.savez_compressed(out / 'reference.npz', **{k: np.stack([p[k] for p in packets])
                                               for k in ('object_pose', 'hand_keypoints', 'timestamps')})
    meta = dict(schema=BANK_SCHEMA, status='COMPLETED', origin=BANK_ORIGIN,
        task=members[0]['task'], motion=members[0]['motion'], source_seed=230,
        source=str(source), source_manifest_sha256=sha(path), members=members,
        aggregation='fixed_uniform_mean', fps=30, units='m', hand_links=list(HAND_LINKS),
        selection='first eight recorded clean train episodes passing existing full-episode weak task_success',
        success_semantics='existing weak geometry qualification; not human-verified success or new Y',
        measurements='native measured object poses and11hand points; no URDF reconstruction',
        sources=frozen, collection_sources=dict(git_commit=raw['git_commit'], verified_historical_code=historical),
        reference_sha256=sha(out / 'reference.npz'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        elapsed_s=time.monotonic() - started,
        device_reason='CPU: bounded file/hash/geometry qualification; no model inference')
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(dict(status='COMPLETED', members=[m['episode'] for m in members], elapsed_s=meta['elapsed_s'])))


if __name__ == '__main__':
    main()
