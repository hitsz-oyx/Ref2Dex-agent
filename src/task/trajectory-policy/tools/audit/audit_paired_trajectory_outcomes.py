"""Pair frozen policies by saved common noise and independently count physical holds."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.evaluation / 'paired_outcomes.json'
    if output.exists():
        raise ValueError('preserve existing pair report')
    manifest = json.loads((args.evaluation / 'manifest.json').read_text())
    result = json.loads((args.evaluation / 'result.json').read_text())
    if manifest['status'] != 'COMPLETED' or manifest['action_mode'] != 'paired_gaussian':
        raise ValueError('completed paired wave required')
    roles, ids = np.asarray(manifest['roles']), np.asarray(manifest['pair_ids'])
    with np.load(args.evaluation / 'trajectory.npz') as f:
        height = f['obj'][:, :, 2, 3] - f['obj'][0, :, 2, 3]
        held = (f['surface_gap'] <= .01) & ~(f['table_footprint'] & (abs(f['support_gap']) <= .02)) & (height >= .03)
        held[0] = False
        clipped = f['clipped'].copy()
    with np.load(args.evaluation / 'plans.npz') as f:
        eps = f['epsilon'].copy()
        history0 = f['history'][0].copy()
    counts = dict(both=0, final_only=0, warm_only=0, neither=0)
    pairs = []
    for pair_id in range(28):
        pair = dict(pair_id=pair_id)
        indices = []
        for role in ('warm_start', 'ppo'):
            rows = np.flatnonzero((roles == role) & (ids == pair_id))
            if len(rows) != 1:
                raise ValueError('nonunique policy pair')
            env = int(rows[0])
            indices.append(env)
            run = longest = 0
            for flag in held[:, env]:
                run = run + 1 if flag else 0
                longest = max(run, longest)
            saved = result['outcomes'][env]
            if saved['env'] != env or saved['role'] != role or saved['maximum_held_frames'] != longest or saved['terminal_held'] != bool(held[-1, env]):
                raise ValueError('paired physical outcome mismatch')
            pair[role] = dict(env=env, maximum_held_frames=longest,
                acquired=bool(held[:, env].any()), terminal_held=bool(held[-1, env]),
                stable=longest >= 433 and bool(held[-1, env]),
                dropout_after_acquisition=bool(held[:, env].any() and not held[-1, env]),
                held_loss_transitions=int(np.sum(held[:-1, env] & ~held[1:, env])),
                clipping_count=int(clipped[:, env].sum()))
        if not np.array_equal(eps[:, indices[0]], eps[:, indices[1]]) or not np.array_equal(history0[indices[0]], history0[indices[1]]):
            raise ValueError('common noise or observed reset differs')
        warm, final = pair['warm_start']['stable'], pair['ppo']['stable']
        label = 'both' if warm and final else ('warm_only' if warm else ('final_only' if final else 'neither'))
        counts[label] += 1
        pairs.append(pair)
    aggregate = {role: dict(acquired=sum(p[role]['acquired'] for p in pairs),
        dropout_after_acquisition=sum(p[role]['dropout_after_acquisition'] for p in pairs),
        stable=sum(p[role]['stable'] for p in pairs)) for role in ('warm_start', 'ppo')}
    files = [args.evaluation / name for name in ('manifest.json', 'result.json', 'trajectory.npz', 'plans.npz', 'noise.npy')] + [Path(__file__)]
    report = dict(pairs=pairs, paired_stable_counts=counts, aggregate=aggregate,
        elapsed_s=time.monotonic()-started, cpu_reason='Saved physical-state statistics and exact pairing checks only; no model computation',
        input_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        claim='Single-wave paired descriptive counts; parallel environments are not exact hidden-physics forks')
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(paired_stable_counts=counts, aggregate=aggregate), indent=2))


if __name__ == '__main__':
    main()
