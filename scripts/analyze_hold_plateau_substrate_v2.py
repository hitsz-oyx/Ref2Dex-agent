#!/usr/bin/env python3
"""Correct a duplicate-key reporting bug without recollecting or changing scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def score_phase(height, contact, progress, initial, start, stop):
    height, contact, progress = np.asarray(height), np.asarray(contact), np.asarray(progress)
    if not np.array_equal(progress, np.arange(1, len(progress) + 1)):
        raise ValueError('native first-episode progress is discontinuous')
    phase = (progress >= start) & (progress <= stop)
    if phase.sum() != 90 or not np.array_equal(progress[phase], np.arange(start, stop + 1)):
        raise ValueError('incomplete actual plateau phase')
    rise = height[phase] - np.float32(initial)
    valid = (rise >= np.float32(.03)) & contact[phase].astype(bool).all(-1)
    longest = current = 0
    for flag in valid:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return dict(max_phase_hold_steps=longest, stable45=longest >= 45, retained75=longest >= 75,
        phase_steps=int(phase.sum()), phase_min_height_m=float(rise.min()),
        phase_mean_height_m=float(rise.mean()), phase_held_fraction=float(valid.mean()),
        phase_force_proxy_fraction=float(contact[phase].astype(bool).all(-1).mean()))


def summarize(rows):
    return dict(episodes=len(rows), retained75_count=sum(r['retained75'] for r in rows),
        retained75_rate=float(np.mean([r['retained75'] for r in rows])),
        stable45_count=sum(r['stable45'] for r in rows),
        stable45_rate=float(np.mean([r['stable45'] for r in rows])),
        mean_max_phase_hold_steps=float(np.mean([r['max_phase_hold_steps'] for r in rows])),
        mean_phase_height_mm=1000 * float(np.mean([r['phase_mean_height_m'] for r in rows])),
        mean_phase_min_height_mm=1000 * float(np.mean([r['phase_min_height_m'] for r in rows])),
        mean_phase_force_proxy_fraction=float(np.mean([r['phase_force_proxy_fraction'] for r in rows])),
        initial_near20mm_count=sum(r['initial_gap_m'] <= .02 for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.directory
    manifest = json.loads((root / 'run_manifest.json').read_text())
    if manifest['run_status'] != 'FAILED' or manifest.get('error') != "RuntimeError('analysis exit 1; retained log')":
        raise ValueError('only the preserved reporting failure is eligible')
    if manifest['panels'] != [[286,498],[286,499],[287,498],[287,499]] or any(
            p['run_status'] != 'COMPLETED' for p in manifest['phases'] if p['name'] != 'analysis'):
        raise ValueError('nonterminal/frozen physical collection drift')
    if any(sha(Path(p)) != h for p,h in manifest['input_sha256'].items()):
        raise ValueError('protected collection input drift')
    target = root / 'analysis_correction_r1'
    target.mkdir(exist_ok=False)
    rows, panel_results = [], {}
    for actor, evaluation in manifest['panels']:
        directory = root / f't{actor}_s{evaluation}'
        result = json.loads((directory / 'results.json').read_text())
        if result['complete_episodes'] != 192 or not result['all_episodes_complete']:
            raise ValueError('incomplete panel')
        for name in ('initial', 'episodes'):
            if sha(directory / (name + '.pt')) != result[name + '_sha256']:
                raise ValueError('saved data drift')
        data = torch.load(directory / 'episodes.pt', map_location='cpu', weights_only=False)
        initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=False)
        if not torch.equal(initial['motion'], data['motion']) or not torch.equal(initial['initial_height'], data['initial_height']):
            raise ValueError('initial state identity drift')
        if sorted(r['environment'] for r in data['episodes']) != list(range(192)):
            raise ValueError('episode identity coverage')
        panel = []
        for episode in data['episodes']:
            env, motion = episode['environment'], episode['motion']
            if motion != int(data['motion'][env]):
                raise ValueError('motion identity drift')
            active = data['active'][:, env].numpy()
            steps = int(active.sum())
            if not np.array_equal(active, np.arange(len(active)) < steps) or steps != episode['steps']:
                raise ValueError('first episode active mask drift')
            done = data['done'][:, env].numpy()[active]
            if not done[-1] or done[:-1].any() or episode['terminate']:
                raise ValueError('first episode termination drift')
            expected_steps = data['loader_checks'][motion]['reference_frames'] - 1
            if steps != expected_steps:
                raise ValueError('first episode reference endpoint drift')
            scores = score_phase(data['height'][:, env].numpy()[active],
                data['contact'][:, env].numpy()[active], data['progress'][:, env].numpy()[active],
                data['initial_height'][env], int(data['phase_start'][motion]), int(data['phase_stop'][motion]))
            for key in ('max_phase_hold_steps','stable45','retained75','phase_steps'):
                if episode[key] != scores[key]:
                    raise ValueError('independent online/offline phase scorer disagreement')
            row = dict(episode)
            row.update(scores)
            row.update(training_seed=actor, evaluation_seed=evaluation,
                initial_gap_m=float(initial['gap'][env]), initial_height_m=float(data['initial_height'][env]))
            rows.append(row)
            panel.append(row)
        panel_results[f't{actor}_s{evaluation}'] = summarize(panel)
    motions = {str(m): summarize([r for r in rows if r['motion'] == m]) for m in range(3)}
    pooled = summarize(rows)
    gates = dict(pooled75_at_least10percent=pooled['retained75_rate'] >= .10,
        **{f'motion{m}_75_at_least5percent': motions[str(m)]['retained75_rate'] >= .05 for m in range(3)})
    subgroups = {dimension: {str(value): summarize([r for r in rows if r[dimension] == value])
        for value in sorted({r[dimension] for r in rows})} for dimension in ('training_seed','evaluation_seed')}
    result = dict(run_status='COMPLETED', label='PROMISING' if all(gates.values()) else 'UNPROMISING',
        gates=gates, complete_episodes=len(rows), pooled=pooled, motions=motions, panels=panel_results, subgroups=subgroups,
        synthetic_task=True, no_training=True,
        boundary='actor-only synthetic-task feasibility; no novel method, matched superiority or formal Validation')
    torch.save(dict(rows=rows), target / 'analysis.pt')
    (target / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    (target / 'trajectory_audit.json').write_text(json.dumps(dict(status='PASS',
        independently_scored_episodes=len(rows), actual_native_progress_verified=True,
        plateau_samples_per_episode=90, first_episode_endpoints_verified=True,
        online_offline_physical_hold_scores_agree=True), indent=2) + '\n')
    (target / 'correction_manifest.json').write_text(json.dumps(dict(run_status='COMPLETED',
        source_sha256=sha(Path(__file__)), frozen_source_sha256=manifest['input_sha256'][str(ROOT / 'scripts/analyze_hold_plateau_substrate.py')],
        original_failed_manifest_sha256=sha(root / 'run_manifest.json'),
        change='duplicate reporting-key repair only; identical score_phase and frozen gates',
        no_new_physics=True, protected_inputs_unchanged=True),indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
