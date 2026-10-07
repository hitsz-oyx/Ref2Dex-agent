"""Slice continuous episodes after splitting; require explicit local preferences."""
import argparse
import json
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))

import numpy as np
from consequence_evaluator.data import K, K_EXEC, SCHEMA, Windows, object_effect, sha


def prepare(source, preferences, output, max_windows=20000, seconds=600):
    source, preferences, output = map(Path, (source, preferences, output))
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= max_windows <= 20000 or not 1 <= seconds <= 600:
        raise ValueError('bounded preparation: <=20000 windows and <=600 seconds')
    started = time.monotonic()
    manifest = json.loads((source / 'manifest.json').read_text())
    if (manifest.get('schema') != 'ref2dex.consequence-evaluator.episodes.v1'
            or manifest.get('status') != 'COMPLETED'
            or manifest.get('rollout_kind') != 'continuous'
            or manifest.get('training_allowed') is not True
            or manifest.get('fps') != 30 or manifest.get('units') != 'm'
            or not manifest.get('history_contract')):
        raise ValueError('continuous, real 30Hz robot episode provenance required')
    pairs = json.loads(preferences.read_text())
    if pairs.get('scope') != 'local_window' or not pairs.get('label_provenance'):
        raise ValueError('explicit local-window annotations required; do not inherit episode outcomes')
    windows = {key: [] for key in ('history', 'action', 'effect', 'progress', 'progress_mask',
                                  'episode', 'split', 'task', 'phase', 'quality')}
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
            expected = {'history', 'action', 'object_pose', 'timestamps', 'phase', 'progress', 'progress_mask'}
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
        if not np.allclose(np.diff(a['timestamps']), 1 / 30, atol=1e-6, rtol=0):
            raise ValueError('episode clock is not contiguous 30 Hz')
        for tick in range(steps - K + 1):
            if len(windows['history']) >= max_windows:
                raise ValueError('window budget exhausted; explicitly subset source episodes')
            if time.monotonic() - started >= seconds:
                raise TimeoutError('bounded episode preparation deadline')
            index = len(windows['history'])
            lookup[(episode, tick)] = index
            values = dict(history=a['history'][tick], action=a['action'][tick:tick + K],
                          effect=object_effect(a['object_pose'][tick:tick + K + 1]),
                          progress=a['progress'][tick + 1:tick + K + 1],
                          progress_mask=a['progress_mask'][tick + 1:tick + K + 1],
                          episode=episode, split=record['split'], task=record['task'],
                          phase=str(a['phase'][tick]), quality=record['quality'])
            for key, value in values.items():
                windows[key].append(value)
    indices, annotations = [], []
    for pair in pairs['pairs']:
        try:
            indices.append([lookup[(pair[name]['episode'], int(pair[name]['tick']))]
                            for name in ('chosen', 'rejected')])
        except KeyError as error:
            raise ValueError('annotation refers to a missing/incomplete 24-step window') from error
        annotations.append(pair['annotation'])
    arrays = {key: np.asarray(values) for key, values in windows.items()}
    arrays.update(pairs=np.asarray(indices, dtype='int64').reshape(-1, 2),
                  pair_annotation=np.asarray(annotations, dtype='U'))
    output.mkdir(parents=True)
    target = output / 'windows.npz'
    np.savez_compressed(target, **arrays)
    output_manifest = dict(schema=SCHEMA, horizon=K, execution_horizon=K_EXEC,
                           fps=30, units='m', rollout_kind='continuous', training_allowed=True,
                           preference_scope='local_window', progress_scope='episode_absolute',
                           action_semantics='executed_native_control',
                           future_semantics='current_anchor_frame_rigid_effect',
                           history_contract=manifest['history_contract'],
                           label_provenance=pairs['label_provenance'],
                           windows_sha256=sha(target), source_episodes=source_hashes,
                           source_manifest_sha256=sha(source / 'manifest.json'),
                           preference_annotations_sha256=sha(preferences),
                           producer_sha256=sha(Path(__file__)), status='PREPARED')
    metadata = output / 'manifest.json'
    metadata.write_text(json.dumps(output_manifest, indent=2) + '\n')
    data = Windows(output, require_complete=False)
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
