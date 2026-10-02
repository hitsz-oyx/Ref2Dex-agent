"""Deterministic saved-trace reproduction of the common-prefix matching failure."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-match', action='store_true')
    a = p.parse_args()
    source, out = a.source.resolve(), a.output.resolve()
    assert ROOT in out.parents and not out.exists()
    i = torch.load(source / 's601/initial.pt', map_location='cpu', weights_only=False)
    t = torch.load(source / 's601/trace.pt', map_location='cpu', weights_only=False)
    r = json.loads((source / 'results.json').read_text())
    cluster, arm = i['option_cluster'].numpy(), i['policy_group'].numpy()
    lookup = np.full((192, 4), -1, dtype=np.int64)
    lookup[cluster, arm] = np.arange(768)
    metrics, first = {}, {}
    for key in ('context', 'normalized_context', 'base_target', 'target', 'action',
                'request', 'native_q', 'native_dq', 'object_root', 'contact',
                'hand_root', 'hand_body_position', 'hand_body_quaternion'):
        data = t[key].numpy()
        greatest, earliest = 0., None
        for ids in lookup:
            decision = int(i['decision_steps'][ids[0]])
            # POST native fields end just BEFORE the decision PRE state.
            stop = decision + 1 if key in ('context', 'normalized_context', 'base_target') else decision
            difference = np.abs(data[:stop, ids] - data[:stop, ids[0], None])
            greatest = max(greatest, float(difference.max()))
            if difference.max() > 1e-7:
                index = np.unravel_index(np.argmax(difference > 1e-7), difference.shape)
                item = dict(trace_tick=int(index[0]), cluster=int(cluster[ids[0]]),
                            compared_arm=int(index[1]), component=list(map(int, index[2:])),
                            difference=float(difference[index]))
                if earliest is None or item['trace_tick'] < earliest['trace_tick']:
                    earliest = item
        metrics[key] = greatest
        first[key] = earliest
    initial = {}
    for key in ('base_q', 'initial_dof_vel', 'object_root', 'table_root', 'initial_contact'):
        data = i[key].numpy()
        initial[key] = float(np.abs(data[lookup] - data[lookup[:, :1]]).max())
    out.mkdir()
    result = dict(run_status='COMPLETED', original_matching_gates=r['matching_gates'],
                  initial_group_maximum=initial, common_prefix_maximum=metrics,
                  first_difference_over1e7=first,
                  source_sha256={str(source / 's601' / f):sha(source / 's601' / f) for f in ('initial.pt','trace.pt')},
                  no_new_physics_or_model_computation=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    if a.require_match:
        assert all(r['matching_gates'].values()), 'saved common-prefix matching gates FAIL'


if __name__ == '__main__':
    main()
