"""Audit frozen weak-value predictions and optional cached same-H forks.

Cached fork scoring has no full-task outcomes and cannot authorize Gate1.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / 'src'))
from consequence_evaluator.data import sha, object_effect, interaction_future
from consequence_evaluator.contracts import HAND_LINKS, is_within


def exhaustive_rank(pred, label, tick, mask):
    correct = []
    for bucket in np.unique(tick[mask] // 16):
        pos = np.flatnonzero(mask & (tick // 16 == bucket) & (label > 0))
        neg = np.flatnonzero(mask & (tick // 16 == bucket) & (label < 0))
        if len(pos) and len(neg):
            delta = pred[pos, None] - pred[neg][None, :]
            correct.extend(((delta > 0) + .5 * (delta == 0)).ravel())
    return float(np.mean(correct)) if correct else None


def cached_scores(root, run, gpu, hashes):
    if subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    from consequence_evaluator.temporal_value import TemporalValue
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    load = lambda path: torch.load(path, map_location='cpu', weights_only=False)
    meta_path = root / 'initial-scores.json'
    meta = json.loads(meta_path.read_text())
    hashes[str(meta_path)] = sha(meta_path)
    folders = [root / ('initial-k' + str(k)) for k in range(7)]
    paths = [folder / name for folder in folders for name in ('panel.pt', 'trace.pt')]
    for path in paths:
        hashes[str(path)] = sha(path)
        if hashes[str(path)] != meta['input_sha256'][str(path)]:
            raise ValueError('frozen fork identity drift')
    panels = [load(folder / 'panel.pt') for folder in folders]
    rows = np.asarray(meta['rows']); query = int(meta['query']); first = panels[0]
    histories = []; actions = []; futures = []; prefix = None
    for k, (panel, folder) in enumerate(zip(panels, folders)):
        if panel['candidate'] != k or panel['post_window'] != 32:
            raise ValueError('fixed candidate identity required')
        for key in ('initial_fingerprint', 'simulation_contract', 'model_fingerprint', 'rms_fingerprint'):
            if panel[key] != first[key]:
                raise ValueError('different actor/world provenance')
        for key in ('triggers', 'delta', 'motion_id', 'start_frame'):
            if not torch.equal(panel[key], first[key]):
                raise ValueError('candidate cohort drift')
        for key in ('before', 'history', 'actor_obs', 'hand_root'):
            if not torch.equal(panel[key][rows], first[key][rows]):
                raise ValueError('same-H candidate identity is not exact')
        if (not panel['valid_steps'][rows].all()
                or not torch.equal(panel['triggers'][rows], torch.full((len(rows),), query))
                or panel['clipped_steps'][rows].any()
                or not (panel['full_world_prefix_errors'] <= torch.tensor([1e-4]*5+[1e-5, 0.])).all()
                or not np.isclose(panel['control_dt'], 1/30)):
            raise ValueError('incomplete or clipped cached candidate')
        trace = load(folder / 'trace.pt'); geometry = trace['progress_geometry']
        if tuple(geometry['hand_links']) != tuple(HAND_LINKS):
            raise ValueError('hand point ordering differs')
        if trace['done'][query:query+24, rows].any():
            raise ValueError('terminal lookahead')
        if prefix is None:
            prefix = geometry
        for key in ('object_pose', 'hand_keypoints'):
            if not torch.equal(geometry[key][:query+1, rows], prefix[key][:query+1, rows]):
                raise ValueError('candidate geometry prefix differs')
        # The historical native collector executes this delta for eight steps,
        # then returns to the frozen reactive actor. No future executed action
        # is exposed as the requested plan.
        plan = np.zeros((len(rows), 24, 18), np.float32)
        plan[:, :8] = panel['delta'][k].numpy()
        measured = []
        for row in rows:
            pose = geometry['object_pose'][query:query+25, row].numpy()
            hand = geometry['hand_keypoints'][query:query+25, row].numpy()
            measured.append(np.concatenate((object_effect(pose)[:, :3].reshape(24, 12),
                                            interaction_future(pose, hand).reshape(24, 33)), -1))
        histories.append(panel['actor_obs'][rows].numpy())
        actions.append(plan); futures.append(np.asarray(measured))
    values = dict(history=np.concatenate(histories), action=np.concatenate(actions),
                  future=np.concatenate(futures))
    scores = {}; ranges = {}
    for arm in ('HA', 'HAZ'):
        saved = load(run / (arm + '.pt'))
        model = TemporalValue(**saved['architecture']).to('cuda:0').eval()
        model.load_state_dict(saved['model'])
        normalized = {}
        for key, value in values.items():
            mean, std = saved['statistics'][key]
            x = (torch.from_numpy(value).float() - mean) / std
            normalized[key] = x.to('cuda:0')
            if arm == 'HAZ':
                ranges[key] = dict(abs_p99=float(torch.quantile(x.abs().flatten(), .99)),
                                   abs_max=float(x.abs().max()))
        with torch.inference_mode():
            pred = model(normalized['history'], normalized['action'], normalized['future'],
                         arm == 'HAZ')['score'].cpu().numpy().reshape(7, len(rows)).T
        scores[arm] = pred
    return dict(scope='cached same-H model transfer diagnostic; no outcomes, no executed control',
        full_task_gate1=False, query=query, rows=rows.tolist(),
        actor_fingerprint=first['model_fingerprint'], normalized_input_ranges=ranges,
        scores={arm: value.tolist() for arm, value in scores.items()},
        choices={arm: value.argmax(-1).tolist() for arm, value in scores.items()},
        selection_counts={arm: np.bincount(value.argmax(-1), minlength=7).tolist()
                          for arm, value in scores.items()},
        differing_choices=int((scores['HA'].argmax(-1) != scores['HAZ'].argmax(-1)).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cached-forks', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args(); started = time.monotonic()
    run = args.run.resolve(); output = run / 'audit.json'
    if output.exists() or not is_within(run, ROOT / 'outputs/consequence-evaluator'):
        raise ValueError('fresh task-owned audit output required')
    result = json.loads((run / 'result.json').read_text()); hashes = dict(result['input_sha256'])
    for path, digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('fixed training input drift')
    for path in (run / 'result.json', run / 'test-predictions.npz', Path(__file__).resolve()):
        hashes[str(path)] = sha(path)
    for arm, digest in result['checkpoint_sha256'].items():
        path = run / (arm + '.pt'); hashes[str(path)] = digest
        if sha(path) != digest:
            raise ValueError('checkpoint drift')
    with np.load(args.data / 'windows.npz', allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    with np.load(run / 'test-predictions.npz', allow_pickle=False) as source:
        prediction = {key: source[key] for key in source.files}
    ids = prediction['indices']; tick = data['tick'][ids]; label = data['label'][ids]
    audit = dict(scope='post-freeze semantic audit; no selection or retraining', splits={}, time_ranges=[])
    active = np.abs(data['action']).max((1, 2)) > 1e-7
    for split in ('train', 'val', 'test'):
        rows = (data['split'] == split) & data['supervised']
        mask = (data['split'] == split) & data['recovery_uncertain']
        audit['splits'][split] = dict(supervised=int(rows.sum()), active_plans=int((rows & active).sum()),
            active_fraction=float(active[rows].mean()), masked=int(mask.sum()),
            successful_recovery_episodes=np.unique(data['episode'][mask]).tolist())
    for low, high in ((0, 48), (48, 144), (144, 400), (400, 519)):
        mask = (tick >= low) & (tick < high)
        audit['time_ranges'].append(dict(tick_range=[low, high], windows=int(mask.sum()),
            exhaustive_same_time_rank={arm: exhaustive_rank(prediction[arm], label, tick, mask)
                                       for arm in ('HA', 'HAZ', 'shuffled')}))
    if args.cached_forks is not None:
        if args.gpu is None:
            raise ValueError('GPU required for batched frozen model inference')
        audit['cached_forks'] = cached_scores(args.cached_forks.resolve(), run, args.gpu, hashes)
    if time.monotonic() - started > 60:
        raise TimeoutError('bounded audit60s')
    if any(sha(path) != digest for path, digest in hashes.items()):
        raise ValueError('audit input drift')
    audit.update(input_sha256=hashes, elapsed_s=time.monotonic() - started)
    output.write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('input_sha256', 'cached_forks')}, indent=2))
    if 'cached_forks' in audit:
        print(json.dumps({key: audit['cached_forks'][key] for key in
                          ('scope', 'selection_counts', 'differing_choices', 'normalized_input_ranges')}, indent=2))


if __name__ == '__main__':
    main()
