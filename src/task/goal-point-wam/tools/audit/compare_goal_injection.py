#!/usr/bin/env python3
"""Compare completed matched injection arms; never choose a checkpoint or train."""
import argparse
import json
from pathlib import Path


def compare(root):
    routes = ('memory', 'query', 'none')
    runs = {r: json.loads((root / r / 'result.json').read_text()) for r in routes}
    audits = {r: json.loads((root / r / 'goal_audit.json').read_text()) for r in routes}
    identities = {r: json.loads((root / r / 'input_manifest.json').read_text()) for r in routes}
    if any(x['status'] != 'COMPLETED' or x['step'] != 600 for x in runs.values()):
        raise ValueError('all final600 arms must complete')
    keys = ('initial_parameter_sha256', 'model_parameters', 'data_manifest_sha256', 'stats_sha256',
            'model_sha256', 'script_sha256', 'train_indices_sha256', 'validation_indices_sha256', 'git_commit')
    for key in keys:
        if len({x[key] for x in identities.values()}) != 1:
            raise ValueError('matched identity drift: ' + key)
    if any(x['pretrained'] != identities['memory']['pretrained'] for x in identities.values()):
        raise ValueError('pretrained identity drift')
    configs = {}
    for route in routes:
        config = dict(audits[route]['config'])
        if config.pop('goal_injection') != route: raise ValueError('wrong injection route')
        configs[route] = config
    if any(c != configs['memory'] for c in configs.values()): raise ValueError('non-route config drift')
    if any(a['windows'] != audits['memory']['windows'] for a in audits.values()):
        raise ValueError('actual validation window/mask drift')
    sources = {}
    accuracy_pass, response_pass = [], []
    for name, a in audits['memory']['sources'].items():
        b, c = audits['query']['sources'][name], audits['none']['sources'][name]
        improvement = dict(vs_memory=1-b['trajectory_epe_m']/a['trajectory_epe_m'],
                           vs_no_goal=1-b['trajectory_epe_m']/c['trajectory_epe_m'],
                           vs_zero=b['improvement_vs_zero_hand'], vs_own_static_goal=b['improvement_vs_static_goal'])
        good = all(x >= .05 for x in improvement.values())
        response = (b['translated_goal_endpoint_rms_m'] >= .001 and
                    b['translated_goal_endpoint_rms_m'] >= 5*a['translated_goal_endpoint_rms_m'])
        if good: accuracy_pass.append(name)
        if response: response_pass.append(name)
        if c['real_vs_static_goal_prediction_difference_m'] > 1e-7 or c['translated_goal_endpoint_rms_m'] > 1e-7:
            raise ValueError('no-goal control retains future goal influence')
        sources[name] = dict(arms={r: audits[r]['sources'][name] for r in routes},
                             query_improvement=improvement, accuracy_gate=good, response_gate=response)
    regression = (runs['query']['validation']['macro/world_point_epe_m_h24'] /
                  runs['memory']['validation']['macro/world_point_epe_m_h24'] - 1)
    isolation = max(r['goal_world_isolation_max'] for r in runs.values())
    passed = len(accuracy_pass) >= 2 and regression <= .05 and isolation <= 1e-5
    return dict(status='COMPLETED', conclusion='PROMISING' if passed else 'UNCLEAR',
                scope='single-seed real-data injection Probe; not physical/robot/RL validation',
                matched_identity={k: identities['memory'][k] for k in keys},
                accuracy_pass_sources=accuracy_pass, response_pass_sources=response_pass,
                query_world_regression_vs_memory=regression, goal_world_isolation_max=isolation,
                sources=sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    output = args.root / 'comparison.json'
    if output.exists(): raise FileExistsError(output)
    result = compare(args.root)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
