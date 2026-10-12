#!/usr/bin/env python3
"""Bind complete fixed-final method Probes; comparisons do not isolate added compute."""
import argparse
import json
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'tools/run'))
from train_three_domain import atomic_json, digest


def improvement(reference, value):
    return 1 - value / reference if reference > 0 else None


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    parent = json.loads((args.root / 'action_world_parent.json').read_text())
    results = {}
    for directory in sorted((args.root / 'training').iterdir()):
        if not directory.is_dir():
            continue
        result = json.loads((directory / 'result.json').read_text())
        identity = json.loads((directory / 'input_manifest.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        if result['status'] != 'COMPLETED' or result['step'] != 3000 or result['engineering_only']:
            raise ValueError('fixed3000 scientific result required: ' + str(directory))
        if identity['parent_checkpoint_sha256'] != parent['checkpoint_sha256'] or identity['validation_indices_sha256'] != parent['indices_sha256']:
            raise ValueError('parent/panel mismatch')
        for key in ('stats_sha256', 'data_manifest_sha256', 'model_sha256'):
            if identity[key] != parent['parent_identity'][key]:
                raise ValueError('frozen input mismatch: ' + key)
        if digest(directory / 'validation_balanced.npy') != parent['indices_sha256']:
            raise ValueError('actual panel drift')
        auxiliary = result['auxiliary_metrics']; sources = {}; passing = []
        for name, baseline in parent['sources'].items():
            aux = auxiliary[name]
            row = dict(auxiliary_improvement_vs_static=improvement(aux['static_epe_m'], aux['prediction_epe_m']), auxiliary=aux)
            if directory.name == 'world_roi':
                moving = result['validation']['metrics'][f'source/{name}/moving_world_point_epe_m_h24']
                row['moving_world_epe_m'] = moving
                row['moving_improvement_vs_identity'] = improvement(baseline['moving_identity_epe_m'], moving)
                positive = row['auxiliary_improvement_vs_static'] is not None and row['auxiliary_improvement_vs_static'] >= .05 and row['moving_improvement_vs_identity'] >= .05
            else:
                actor = result['goal_ablation'][name]
                row.update(actor=actor, actor_improvement_vs_parent=improvement(baseline['parent_actor_trajectory_epe_m'], actor['real_epe_m']),
                           actor_improvement_vs_static_goal=improvement(actor['static_epe_m'], actor['real_epe_m']),
                           actor_improvement_vs_zero=improvement(actor['zero_epe_m'], actor['real_epe_m']))
                positive = all(row[key] is not None and row[key] >= .05 for key in
                               ('actor_improvement_vs_parent', 'actor_improvement_vs_static_goal', 'actor_improvement_vs_zero'))
            if positive:
                passing.append(name)
            sources[name] = row
        results[directory.name] = dict(status='PROMISING' if len(passing) >= 2 else 'UNCLEAR', passing_sources=passing,
                                      sources=sources, final_step=result['step'], config=config, input_identity=identity,
                                      checkpoint_sha256=digest(directory / 'final.pt'), result_sha256=digest(directory / 'result.json'))
    if set(results) != {'world_roi', 'hindsight_prefix', 'intermediate_plan', 'change_supervision'}:
        raise ValueError('all four predeclared methods required')
    atomic_json(args.output, dict(status='COMPLETED', methods=results, parent_audit_sha256=digest(args.root / 'action_world_parent.json'),
                                 comparator_sha256=digest(Path(__file__)),
                                 scope='Single-seed shared-panel method continuation; no same-duration control, paper reproduction, causal mechanism superiority, physical execution or RL claim.'))
    print(json.dumps({name: dict(status=row['status'], passing_sources=row['passing_sources']) for name, row in results.items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
