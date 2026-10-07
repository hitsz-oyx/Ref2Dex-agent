"""Frozen held-out ranking and a task/phase-matched future donor control."""
import time

import numpy as np
import torch
from .supervision import states_match


def future_donors(data, ids, seed):
    """Sample with replacement from other test episodes in the same stratum.

    This is a reassignment diagnostic, not an action-conditioned simulator and
    not a bijective permutation. H, requested residual plan and labels are never reassigned.
    """
    a = data.arrays
    if (a['split'][ids] != 'test').any():
        raise ValueError('future control is restricted to the frozen test split')
    rng = np.random.default_rng(seed)
    donors = []
    for index in ids:
        eligible = np.flatnonzero((a['split'] == 'test') & (a['task'] == a['task'][index])
                                  & (a['phase'] == a['phase'][index])
                                  & (a['expert']==a['expert'][index]) & (a['motion']==a['motion'][index])
                                  & (a['episode'] != a['episode'][index]))
        eligible=np.asarray([other for other in eligible if states_match(
            dict(history=a['history'][index], object_pose=a['current_object'][index],
                 hand_keypoints=a['current_hand'][index]),
            dict(history=a['history'][other], object_pose=a['current_object'][other],
                 hand_keypoints=a['current_hand'][other]), require_history=True)],dtype='int64')
        if not len(eligible):
            raise ValueError('no cross-episode future donor in the test stratum')
        donors.append(rng.choice(eligible))
    return np.asarray(donors, dtype=np.int64)


@torch.no_grad()
def predict(model, data, ids, statistics, device, use_future, batch=128,
            donors=None, deadline=None):
    """Score each unique window once; preserve the caller's model mode."""
    if donors is not None and len(donors) != len(ids):
        raise ValueError('one future donor per evaluated window is required')
    was_training = model.training
    model.eval()
    scores, progress = [], []
    try:
        for start in range(0, len(ids), batch):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('fixed held-out evaluation deadline')
            inputs, _ = data.batch(ids[start:start + batch], device)
            if donors is not None:
                other, _ = data.batch(donors[start:start + batch], device)
                inputs['future'] = other['future']
            inputs = {key: (value - statistics[key][0].to(device)) / statistics[key][1].to(device)
                      for key, value in inputs.items()}
            result = model(**inputs, use_future=use_future)
            if any(not torch.isfinite(value).all() for value in result.values()):
                raise FloatingPointError('nonfinite held-out predictions')
            scores.append(result['score'].cpu().numpy())
            bins = torch.linspace(0, 1, 10, device=device)
            progress.append((result['progress_logits'].softmax(-1) @ bins).cpu().numpy())
    finally:
        model.train(was_training)
    return dict(score=np.concatenate(scores), progress=np.concatenate(progress))


def ranking_metrics(margin, episodes):
    """Ties count as incorrect for the primary strict preference metric."""
    correct = margin > 0
    groups = {}
    for outcome, pair in zip(correct, episodes):
        key = tuple(sorted(map(str, pair)))
        groups.setdefault(key, []).append(bool(outcome))
    return dict(pairs=len(margin), strict_accuracy=float(correct.mean()),
                ties=int((margin == 0).sum()), bt_loss=float(np.logaddexp(0, -margin).mean()),
                episode_pair_groups=len(groups),
                macro_episode_pair_accuracy=float(np.mean([np.mean(v) for v in groups.values()])))


def compare(baseline, oracle):
    first, second = baseline > 0, oracle > 0
    return dict(accuracy_gain=float(second.mean() - first.mean()),
                both_correct=int((first & second).sum()),
                both_incorrect=int((~first & ~second).sum()),
                oracle_only_correct=int((~first & second).sum()),
                baseline_only_correct=int((first & ~second).sum()))


def summarize(data, ids, predictions):
    """Use only test pairs, with descriptive task/stage/quality breakdowns."""
    a = data.arrays
    pair_ids = data.pair_ids['test']
    if not len(pair_ids):
        raise ValueError('held-out test preferences are required')
    pairs = a['pairs'][pair_ids]
    expected = np.unique(pairs)
    if not np.array_equal(ids, expected):
        raise ValueError('predictions must cover exactly the unique test-pair windows')
    positions = np.searchsorted(ids, pairs)
    margins = {}
    progress = {}
    for name, values in predictions.items():
        if (values['score'].shape != (len(ids),)
                or values['progress'].shape != (len(ids), 24)
                or not all(np.isfinite(v).all() for v in values.values())):
            raise ValueError('invalid prediction arrays')
        margins[name] = values['score'][positions[:, 0]] - values['score'][positions[:, 1]]
        mask = a['progress_mask'][ids]
        progress[name] = dict(labeled_frames=int(mask.sum()),
                              unique_windows=int(mask.any(axis=1).sum()),
                              mae=float(np.abs(values['progress'][mask] - a['progress'][ids][mask]).mean())
                              if mask.any() else None)

    def panel(selected):
        result = {name: ranking_metrics(value[selected], a['episode'][pairs[selected]])
                  for name, value in margins.items()}
        result['object_minus_baseline'] = compare(margins['baseline'][selected], margins['oracle_object'][selected])
        result['interaction_minus_object'] = compare(margins['oracle_object'][selected], margins['oracle_interaction'][selected])
        result['interaction_minus_baseline'] = compare(margins['baseline'][selected], margins['oracle_interaction'][selected])
        return result

    by_task, by_task_phase, by_quality = [], [], []
    for task in np.unique(a['task'][pairs[:, 0]]):
        selected = a['task'][pairs[:, 0]] == task
        by_task.append(dict(task=str(task), metrics=panel(selected)))
        for phase in np.unique(a['phase'][pairs[selected, 0]]):
            group = selected & (a['phase'][pairs[:, 0]] == phase)
            by_task_phase.append(dict(task=str(task), phase=str(phase), metrics=panel(group)))
    qualities = a['quality'][pairs]
    for chosen, rejected in sorted(set(map(tuple, qualities.tolist()))):
        selected = (qualities[:, 0] == chosen) & (qualities[:, 1] == rejected)
        by_quality.append(dict(chosen=chosen, rejected=rejected, metrics=panel(selected)))
    report = dict(split='test', unique_windows=len(ids), metrics=panel(np.ones(len(pairs), dtype=bool)),
                  by_task=by_task, by_task_phase=by_task_phase, by_quality=by_quality,
                  progress_on_reliable_frames=progress,
                  interpretation='descriptive single-seed probe; overlapping windows are not independent')
    report['aligned_minus_reassigned']={}
    for arm in ('oracle_object','oracle_interaction'):
        if arm+'_reassigned_future' in margins:
            report['aligned_minus_reassigned'][arm]=compare(margins[arm+'_reassigned_future'],margins[arm])
    return report
