"""Summarize pre/post-query GPU-group noise without rerunning simulation.

The output is an engineering diagnostic. It does not assert that envs are
same-state twins and cannot be passed to the strict Gate1 scorer.
"""
import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np

FIELDS = ('object_pose', 'hand_keypoints', 'dof_position', 'dof_velocity',
          'object_velocity', 'history', 'native_contact_forces',
          'native_object_contact_forces')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _p95_abs(array):
    flat = np.abs(np.asarray(array, dtype=np.float64)).reshape(-1)
    return dict(max_abs=float(flat.max()) if flat.size else 0.,
                p95_abs=float(np.quantile(flat, .95)) if flat.size else 0.)


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return (dict(zip(('p50', 'p90', 'p95', 'max'),
                     [float(value) for value in np.quantile(values, [.5, .9, .95, 1.])]))
            if values.size else {})


def load_packet(path):
    with Path(path).open('rb') as stream:
        packet = pickle.load(stream)
    if packet.get('engineering_only') is not True:
        raise ValueError('engineering-only packet required')
    count = int(packet.get('group_envs', 0)); query = int(packet.get('query_tick', -1))
    if count < 4 or query != 48:
        raise ValueError('native group query-tick48 packet required')
    if packet.get('candidate_roles') != ['baseline', 'zero_repeat', 'positive', 'negative']:
        raise ValueError('candidate/noise role contract required')
    if packet.get('group_mode') not in {
            'synchronous_same_process_act_candidate_noise',
            'synchronous_same_process_noise_calibration'}:
        raise ValueError('candidate/noise group packet required')
    if packet.get('control_prefix_exact') is not True:
        raise ValueError('exact shared control prefix required')
    actions = np.asarray(packet.get('actions'))
    done = np.asarray(packet.get('done'))
    if (actions.ndim != 3 or actions.shape[0] < query + 24 or actions.shape[1] != count
            or done.ndim != 2 or done.shape[:2] != actions.shape[:2]):
        raise ValueError('complete group action window required')
    raw_zero_roles = packet.get('zero_role_indices')
    try:
        zero_roles = [int(index) for index in raw_zero_roles]
    except (TypeError, ValueError):
        raise ValueError('valid candidate/noise zero-role set required')
    if (len(zero_roles) < 2 or len(set(zero_roles)) != len(zero_roles)
            or any(index < 0 or index >= count or index in (2, 3) for index in zero_roles)):
        raise ValueError('valid candidate/noise zero-role set required')
    for index in range(count):
        if (not np.array_equal(actions[:query, index], actions[:query, 0])
                or not np.array_equal(done[:query, index], done[:query, 0])):
            raise ValueError('all group roles must share the recorded prefix controls/done')
    for key in FIELDS:
        value = np.asarray(packet.get(key))
        if (value.ndim < 2 or value.shape[0] != actions.shape[0] + 1
                or value.shape[0] < query + 25 or value.shape[1] != count):
            raise ValueError('packet field shape mismatch for %s' % key)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite packet field %s' % key)
    timestamps = np.asarray(packet.get('timestamps'))
    if (timestamps.ndim != 1 or timestamps.shape[0] != actions.shape[0] + 1
            or not np.isfinite(timestamps).all()
            or not np.allclose(np.diff(timestamps), 1. / 30., atol=1e-8, rtol=0)):
        raise ValueError('state timestamps must match native 30Hz cadence')
    return packet


def audit_packet(packet, packet_path):
    count = int(packet['group_envs']); query = int(packet['query_tick'])
    stop = min(len(packet['object_pose']), query + 25)
    roles = [int(index) for index in packet['zero_role_indices']]
    if len(roles) < 2 or len(set(roles)) != len(roles) or any(i < 0 or i >= count for i in roles):
        raise ValueError('invalid nominal zero-role set')
    pair_list = list(itertools.combinations(roles, 2))
    pairwise = {}
    for key in FIELDS:
        value = np.asarray(packet[key], dtype=np.float64)
        records = []
        for left, right in pair_list:
            pre = value[:query + 1, right] - value[:query + 1, left]
            post = ((value[query + 1:stop, right] - value[query, right])
                    - (value[query + 1:stop, left] - value[query, left]))
            records.append(dict(left=left, right=right,
                                prequery=_p95_abs(pre),
                                postquery_displacement=_p95_abs(post)))
        pair_pre = [item['prequery']['p95_abs'] for item in records]
        pair_post = [item['postquery_displacement']['p95_abs'] for item in records]
        pairwise[key] = dict(zero_roles=roles, pair_count=len(records), pairs=records,
                             prequery_p95_quantiles=_quantiles(pair_pre),
                             postquery_displacement_p95_quantiles=_quantiles(pair_post))

    candidate_effect = {}
    for env_index, name in ((2, 'positive'), (3, 'negative')):
        candidate_effect[name] = {}
        for key in FIELDS:
            value = np.asarray(packet[key], dtype=np.float64)
            against_zero = []
            for zero in roles:
                incremental = ((value[query + 1:stop, env_index] - value[query, env_index])
                               - (value[query + 1:stop, zero] - value[query, zero]))
                measure = _p95_abs(incremental)
                against_zero.append(dict(zero=zero, **measure))
            noise = [item['postquery_displacement']['p95_abs'] for item in pairwise[key]['pairs']]
            median_noise = float(np.median(noise)) if noise else 0.
            ratios = [item['p95_abs'] / median_noise if median_noise > 0 else None
                      for item in against_zero]
            finite_ratios = [ratio for ratio in ratios if ratio is not None]
            candidate_effect[name][key] = dict(
                against_zero=against_zero,
                ratio_to_zero_pair_median_p95=_quantiles(finite_ratios),
                zero_pair_median_postquery_p95=median_noise,
            )
    return dict(
        schema='ref2dex.consequence-gate1.gpu-group-noise-audit.v1',
        engineering_only=True, packet=str(Path(packet_path).resolve()),
        packet_sha256=_sha256(packet_path),
        group_mode=packet.get('group_mode'), group_envs=count, query_tick=query,
        source_backend=packet.get('source_backend'), replay_identity=packet.get('replay_identity'),
        horizon=24, zero_roles=roles, selected_zero_pair=packet.get('zero_env_pair'),
        pairwise_zero_noise=pairwise, candidate_effect_vs_zero=candidate_effect,
        note='pre/post-query engineering noise summary; no strict same-state or Gate1 claim',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [audit_packet(load_packet(path), path) for path in args.packet]
    output = args.output_dir / 'noise-audit.json'
    if output.exists():
        raise FileExistsError(str(output))
    output.write_text(json.dumps(dict(schema='ref2dex.consequence-gate1.gpu-group-noise-audit.v1',
                                      engineering_only=True, results=results), indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
