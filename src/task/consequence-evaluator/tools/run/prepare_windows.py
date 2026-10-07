"""Slice continuous episodes after splitting; require explicit local preferences."""
import argparse
import json
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))

import numpy as np
from consequence_evaluator.data import K, K_EXEC, SCHEMA, Windows, object_effect, interaction_future, sha
from consequence_evaluator.contracts import (EPISODE_SCHEMA, ACTION_SEMANTICS, HAND_LINKS,
                                             MIN_PREFERENCE_PAIRS, HISTORY_MATCH_MAX_RELATIVE_RMS)
from consequence_evaluator.supervision import CONTACT_SEMANTICS, RULE, pair_coverage_ready, states_match


def prepare(source, preferences, output, max_windows=20000, seconds=600):
    source, preferences, output = map(Path, (source, preferences, output))
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= max_windows <= 20000 or not 1 <= seconds <= 600:
        raise ValueError('bounded preparation: <=20000 windows and <=600 seconds')
    started = time.monotonic()
    source_manifest_path = source / 'manifest.json'
    source_manifest_sha = sha(source_manifest_path)
    preferences_sha = sha(preferences)
    manifest = json.loads(source_manifest_path.read_text())
    if (manifest.get('schema') != EPISODE_SCHEMA or manifest.get('action_semantics')!=ACTION_SEMANTICS
            or manifest.get('status') != 'COMPLETED'
            or manifest.get('rollout_kind') != 'continuous'
            or manifest.get('training_allowed') is not True
            or manifest.get('fps') != 30 or manifest.get('units') != 'm'
            or not manifest.get('history_contract')
            or manifest.get('contact_semantics') != CONTACT_SEMANTICS
            or manifest.get('label_rule') != RULE):
        raise ValueError('continuous, real 30Hz robot episode provenance required')
    pairs = json.loads(preferences.read_text())
    provenance = pairs.get('label_provenance') or {}
    if pairs.get('scope') != 'local_window' or not provenance:
        raise ValueError('explicit local-window annotations required; do not inherit episode outcomes')
    if (provenance.get('rule') != RULE or provenance.get('contact_semantics') != CONTACT_SEMANTICS
            or provenance.get('rule_sha256') != sha(TASK / 'src/consequence_evaluator/supervision.py')
            or provenance.get('contracts_sha256') != sha(TASK / 'src/consequence_evaluator/contracts.py')
            or provenance.get('history_match_relative_rms') != HISTORY_MATCH_MAX_RELATIVE_RMS
            or provenance.get('minimum_pair_coverage') != MIN_PREFERENCE_PAIRS):
        raise ValueError('preference annotation provenance is stale for the current H-matched rule')
    pair_coverage_required = manifest.get('pair_coverage_required', True)
    if pair_coverage_required:
        if (manifest.get('all_experts_operationally_qualified') is not True
                or provenance.get('all_experts_operationally_qualified') is not True):
            raise ValueError('qualified six-expert route provenance is required')
        if (not manifest.get('route_sha256')
                or provenance.get('route_sha256') != manifest['route_sha256']):
            raise ValueError('preference route provenance does not match source route')
        if (not manifest.get('label_report_sha256')
                or provenance.get('report_sha256') != manifest['label_report_sha256']):
            raise ValueError('preference report provenance does not match source label report')
    windows = {key: [] for key in ('history', 'action', 'effect','interaction','current_object','current_hand', 'progress', 'progress_mask',
                                  'episode', 'split', 'task', 'phase', 'quality','expert','motion')}
    lookup, groups, source_hashes = {}, {}, {}
    for record in manifest['episodes']:
        episode = record['episode']
        if episode in source_hashes:
            raise ValueError('duplicate episode identity')
        group = record['split_group']
        if group in groups and groups[group] != record['split']:
            raise ValueError('seed/episode source group crosses splits')
        groups[group] = record['split']
        path = (source / record['path']).resolve()
        if sha(path) != record['sha256']:
            raise ValueError('episode source identity changed')
        source_hashes[episode] = dict(path=str(path), sha256=record['sha256'], split_group=group)
        with np.load(path, allow_pickle=False) as packet:
            expected = {'history', 'action','residual_plan','plan_known','hand_keypoints', 'object_pose', 'timestamps', 'phase', 'progress', 'progress_mask'}
            if set(packet.files) != expected:
                raise ValueError('episode array whitelist mismatch')
            a = {name: packet[name] for name in packet.files}
        steps = len(a['action'])
        if (a['action'].shape != (steps, 18) or len(a['history']) != steps + 1
                or a['object_pose'].shape != (steps + 1, 4, 4)
                or a['timestamps'].shape != (steps + 1,)
                or a['phase'].shape != (steps,)
                or a['progress'].shape != (steps + 1,)
                or a['progress_mask'].shape != (steps + 1,)):
            raise ValueError('pre/post action alignment mismatch')
        if (a['residual_plan'].shape!=(steps,K,18) or a['plan_known'].shape!=(steps,)
                or a['plan_known'].dtype!=np.bool_ or a['hand_keypoints'].shape!=(steps+1,len(HAND_LINKS),3)):
            raise ValueError('requested-plan or measured interaction contract mismatch')
        if not np.allclose(np.diff(a['timestamps']), 1 / 30, atol=1e-6, rtol=0):
            raise ValueError('episode clock is not contiguous 30 Hz')
        ticks = record.get('window_ticks', list(range(steps - K + 1)))
        if (not isinstance(ticks, list) or any(isinstance(t, bool) or not isinstance(t, int)
                                             or t < 0 or t > steps-K for t in ticks)
                or len(set(ticks)) != len(ticks)):
            raise ValueError('invalid explicit full-window selection')
        for tick in sorted(ticks):
            if not a['plan_known'][tick]:
                raise ValueError('unknown state-triggered intervention schedule cannot enter evaluator')
            if len(windows['history']) >= max_windows:
                raise ValueError('window budget exhausted; explicitly subset source episodes')
            if time.monotonic() - started >= seconds:
                raise TimeoutError('bounded episode preparation deadline')
            index = len(windows['history'])
            lookup[(episode, tick)] = index
            values = dict(history=a['history'][tick], action=a['residual_plan'][tick],
                          effect=object_effect(a['object_pose'][tick:tick + K + 1]),
                          interaction=interaction_future(a['object_pose'][tick:tick+K+1],a['hand_keypoints'][tick:tick+K+1]),
                          current_object=a['object_pose'][tick],current_hand=a['hand_keypoints'][tick],
                          progress=a['progress'][tick + 1:tick + K + 1],
                          progress_mask=a['progress_mask'][tick + 1:tick + K + 1],
                          episode=episode, split=record['split'], task=record['task'],
                          phase=str(a['phase'][tick]), quality=record['quality'],expert=record['expert'],motion=record['motion'])
            for key, value in values.items():
                windows[key].append(value)
    indices, annotations = [], []
    pair_directions = {}
    for pair in pairs['pairs']:
        endpoints = {}
        for name in ('chosen', 'rejected'):
            endpoint = pair.get(name, {})
            tick = endpoint.get('tick')
            if (isinstance(tick, bool) or not isinstance(tick, int)
                    or (endpoint.get('episode'), tick) not in lookup):
                raise ValueError('annotation tick must be an integer selected full-window endpoint')
            endpoints[name] = (endpoint['episode'], tick)
        try:
            current_indices = [lookup[endpoints[name]]
                               for name in ('chosen', 'rejected')]
        except KeyError as error:
            raise ValueError('annotation refers to a missing/incomplete 24-step window') from error
        key = tuple(sorted(current_indices))
        direction = tuple(current_indices)
        previous = pair_directions.get(key)
        if previous is not None and previous != direction:
            raise ValueError('contradictory local preference directions')
        pair_directions[key] = direction
        indices.append(current_indices)
        annotations.append(pair['annotation'])
    arrays = {key: np.asarray(values) for key, values in windows.items()}
    arrays.update(pairs=np.asarray(indices, dtype='int64').reshape(-1, 2),
                  pair_annotation=np.asarray(annotations, dtype='U'))
    pair_groups = {split: set() for split in ('train', 'val', 'test')}
    for left, right in arrays['pairs']:
        split = str(arrays['split'][left])
        pair_groups[split].add(tuple(sorted((str(arrays['episode'][left]), str(arrays['episode'][right])))))
    pair_counts = {split: len(groups) for split, groups in pair_groups.items()}
    if pair_coverage_required and not pair_coverage_ready(pair_counts):
        raise ValueError('insufficient unique local preference coverage: '
                         + json.dumps({'observed': pair_counts, 'minimum': MIN_PREFERENCE_PAIRS}, sort_keys=True))
    for left, right in arrays['pairs']:
        if not states_match(dict(history=arrays['history'][left], object_pose=arrays['current_object'][left],
                                 hand_keypoints=arrays['current_hand'][left]),
                            dict(history=arrays['history'][right], object_pose=arrays['current_object'][right],
                                 hand_keypoints=arrays['current_hand'][right]), require_history=True):
            raise ValueError('preference current H/object/hand states do not match')
    output.mkdir(parents=True)
    target = output / 'windows.npz'
    np.savez_compressed(target, **arrays)
    output_manifest = dict(schema=SCHEMA, horizon=K, execution_horizon=K_EXEC,
                           fps=30, units='m', rollout_kind='continuous', training_allowed=True,
                           preference_scope='local_window', progress_scope='episode_absolute',
                           action_semantics=ACTION_SEMANTICS,
                           future_semantics='object_effect_and_measured_hand_object_relative_keypoints',
                           hand_keypoint_links=list(HAND_LINKS),
                           history_contract=manifest['history_contract'],
                           label_provenance=pairs['label_provenance'],
                           windows_sha256=sha(target), source_episodes=source_hashes,
                           source_manifest_sha256=source_manifest_sha,
                           preference_annotations_sha256=preferences_sha,
                           route_sha256=provenance.get('route_sha256'),
                           label_report_sha256=provenance.get('report_sha256'),
                           all_experts_operationally_qualified=provenance.get('all_experts_operationally_qualified'),
                           minimum_pair_coverage=MIN_PREFERENCE_PAIRS,
                           pair_coverage_required=pair_coverage_required,
                           unique_episode_pair_groups=pair_counts,
                           producer_sha256=sha(Path(__file__)), status='PREPARED')
    metadata = output / 'manifest.json'
    metadata.write_text(json.dumps(output_manifest, indent=2) + '\n')
    data = Windows(output, require_complete=False)
    if sha(source_manifest_path) != source_manifest_sha or sha(preferences) != preferences_sha:
        raise RuntimeError('source manifest or preference annotation changed during preparation')
    output_manifest.update(status='CONTRACT_PASS', windows=len(arrays['history']),
                           pairs={name: len(ids) for name, ids in data.pair_ids.items()})
    metadata.write_text(json.dumps(output_manifest, indent=2) + '\n')
    return output_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--preferences', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-windows', type=int, default=20000)
    p.add_argument('--seconds', type=int, default=600)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.preferences, a.output, a.max_windows, a.seconds), indent=2))


if __name__ == '__main__':
    main()
