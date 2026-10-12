#!/usr/bin/env python3
"""Bind final artifacts and report separate observed-hand/goal/world-off gates."""
import argparse
import json
from pathlib import Path

from compare_goal_injection import digest, load_run


def compare(root, baseline):
    routes = ('memory', 'query', 'none')
    names = ('memory-world1', 'query-world1', 'none-world1', 'memory-world0', 'query-world0')
    new = {name: load_run(root / name) for name in names}
    old = {route: load_run(baseline / route) for route in routes}
    full_hashes = {x[2]['initial_parameter_sha256'] for x in new.values()}
    if len(full_hashes) != 1: raise ValueError('new arms have different full initialization')
    core = {x[2]['initial_core_parameter_sha256'] for x in new.values()}
    core.update(x[2]['initial_parameter_sha256'] for x in old.values())
    if len(core) != 1: raise ValueError('old/new core initialization differs')
    all_runs = list(new.values()) + list(old.values())
    for key in ('data_manifest_sha256', 'stats_sha256', 'train_indices_sha256', 'validation_indices_sha256'):
        if len({x[2][key] for x in all_runs}) != 1: raise ValueError('old/new identity drift: '+key)
    if any(x[2]['pretrained'] != all_runs[0][2]['pretrained'] for x in all_runs):
        raise ValueError('pretrained transfer drift')
    reference = new['memory-world1'][1]
    common_configs = []
    for run, audit, identity in all_runs:
        if audit['windows'] != reference['windows']: raise ValueError('actual panel/mask drift')
        config = dict(audit['config'])
        for key in ('goal_injection', 'world_weight', 'observed_hand_query', 'expected_initial_core_parameter_sha256'):
            config.pop(key, None)
        common_configs.append(config)
    isolation = {}
    for name, path, loaded in ([(n, root/n, x) for n, x in new.items()] +
                                [('no-hand-'+r, baseline/r, x) for r, x in old.items()]):
        evidence = json.loads((path/'world_isolation.json').read_text())
        if (evidence['checkpoint_sha256'] != digest(path/'final.pt') or
                evidence['input_identity'] != loaded[2] or
                evidence['checkpoint_step'] != loaded[0]['step']):
            raise ValueError('isolation evidence identity drift: '+name)
        if (evidence['actual_model_sha256'] != loaded[2]['model_sha256'] or
                not evidence['engineering_only']):
            raise ValueError('isolation model/scope mismatch: '+name)
        if (not evidence['cached_isolation_pass'] or
                not evidence['world_to_goal_gradient_is_none'] or
                any(v['all_max'] > 1e-5 for v in evidence['cached_world'].values())):
            raise ValueError('fixed-input world independence failed: '+name)
        isolation[name] = dict(original_raw_max=loaded[0]['goal_world_isolation_max'],
                               original_raw_gate_pass=loaded[0]['goal_world_isolation_max'] <= 1e-5,
                               supplementary=evidence)
    if any(c != common_configs[0] for c in common_configs): raise ValueError('uncontrolled configuration difference')
    for name, (_, audit, _) in new.items():
        config = audit['config']
        route, world = name.split('-world')
        if (config['goal_injection'] != route or config['world_weight'] != int(world)
                or not config['observed_hand_query']): raise ValueError('wrong new arm wiring')
    for route, (_, audit, _) in old.items():
        if (audit['config']['goal_injection'] != route or audit['config']['world_weight'] != 1
                or audit['config'].get('observed_hand_query', False)): raise ValueError('wrong old arm wiring')
    gates = dict(hand_access={r: [] for r in routes}, goal={r: [] for r in ('memory', 'query')},
                 world_off={r: [] for r in ('memory', 'query')})
    sources = {}
    for source in reference['sources']:
        measured = {n: x[1]['sources'][source] for n, x in new.items()}
        original = {r: x[1]['sources'][source] for r, x in old.items()}
        control = measured['none-world1']
        if (control['real_vs_static_goal_prediction_difference_m'] > 1e-7 or
                control['translated_goal_endpoint_rms_m'] > 1e-7): raise ValueError('no-goal response leak')
        improvements = dict(hand_access={}, goal={}, world_off={})
        for route in routes:
            arm = measured[route+'-world1']
            gain = 1-arm['trajectory_epe_m']/original[route]['trajectory_epe_m']
            improvements['hand_access'][route] = gain
            if gain >= .05 and arm['improvement_vs_zero_hand'] >= .05:
                gates['hand_access'][route].append(source)
        for route in ('memory', 'query'):
            arm, off = measured[route+'-world1'], measured[route+'-world0']
            gain = 1-arm['trajectory_epe_m']/control['trajectory_epe_m']
            improvements['goal'][route] = dict(vs_no_goal=gain, vs_own_static_goal=arm['improvement_vs_static_goal'])
            if gain >= .05 and arm['improvement_vs_static_goal'] >= .05:
                gates['goal'][route].append(source)
            gain = 1-off['trajectory_epe_m']/arm['trajectory_epe_m']
            improvements['world_off'][route] = gain
            if gain >= .05: gates['world_off'][route].append(source)
        sources[source] = dict(new_arms=measured, no_hand_arms=original, improvements=improvements)
    labels = {group: {r: 'PROMISING' if len(s) >= 2 else 'UNCLEAR' for r, s in arms.items()}
              for group, arms in gates.items()}
    return dict(status='COMPLETED', scope='single-seed local input/loss Probe, no robot/RL claim',
                component_labels=labels, component_label_scope='accuracy gates only; original raw isolation failures retained',
                world_isolation=isolation,
                original_raw_isolation_gate_pass=all(v['original_raw_gate_pass'] for v in isolation.values()),
                supplementary_cached_isolation_gate_pass=True, gate_pass_sources=gates, sources=sources,
                common_core_parameter_sha256=next(iter(core)), new_full_parameter_sha256=next(iter(full_hashes)),
                matched_train_indices_sha256=all_runs[0][2]['train_indices_sha256'],
                matched_validation_indices_sha256=all_runs[0][2]['validation_indices_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    output = args.root / 'comparison.json'
    if output.exists(): raise FileExistsError(output)
    result = compare(args.root, args.baseline)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
