"""Frozen ref13 U32 teacher and geometry-only evaluator/PW adapters."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'src/task/cm-interaction-oracle/src'))
from rolling_y_audit import short_y
from oracle_y_utility import utility
from .model import Evaluator

SCHEMA = 'ref2dex.consequence-old-utility.v1'
PW_ORDER = [0, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10]


def teacher(current_height, current_pair, height, pair, rest):
    tensors = [torch.as_tensor(x) for x in (current_height, current_pair, height, pair, rest)]
    y, _ = short_y(*tensors)
    return y.numpy(), utility(y).numpy()


class OldUtility(nn.Module):
    def __init__(self, history_dim=1442, width=128, layers=2):
        super().__init__()
        base = Evaluator(history_dim, width, layers)
        for key in ('history', 'action', 'future', 'time', 'encoder', 'score'):
            setattr(self, key, getattr(base, key))

    def forward(self, history, action, future, use_future=True):
        if action.shape[1:] != (24, 18) or future.shape[1:] != (24, 45):
            raise ValueError('decision-known plan24 / geometric future24 required')
        if not use_future:
            future = torch.zeros_like(future)
        tokens = torch.cat((self.history(history)[:, None],
                            self.action(action) + self.future(future)), 1) + self.time
        return self.score(self.encoder(tokens)[:, 1:].mean(1)).squeeze(-1)


def panel_metrics(target, prediction):
    target = np.asarray(target); prediction = np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 7:
        raise ValueError('complete seven-candidate panels required')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('finite scores required')
    i, j = np.triu_indices(7, 1)
    truth = target[:, i] - target[:, j]; delta = prediction[:, i] - prediction[:, j]
    strict = np.abs(truth) > .02
    ties = np.abs(delta) <= 1e-7
    correct = np.where(ties, .5, (truth * delta > 0).astype(float))
    choice = prediction.argmax(1); best = target.argmax(1)
    regret = target.max(1) - target[np.arange(len(target)), choice]
    informative = strict.any(1); correlations = []
    for y, q in zip(target[informative], prediction[informative]):
        yr, qr = rankdata(y), rankdata(q)
        correlations.append(float(np.corrcoef(yr, qr)[0, 1]) if np.std(qr) > 0 else 0.)
    return dict(pairwise_accuracy=float(correct[strict].mean()) if strict.any() else None,
        strict_pairs=int(strict.sum()), informative_anchors=int(informative.sum()),
        anchors=len(target), top1_agreement=float((choice == best).mean()),
        top1_in_teacher_tie_set=float(np.isclose(target[np.arange(len(target)), choice], target.max(1), atol=1e-7).mean()),
        mean_regret=float(regret.mean()), median_regret=float(np.median(regret)),
        informative_mean_regret=float(regret[informative].mean()) if informative.any() else None,
        spearman_informative=float(np.mean(correlations)) if correlations else None,
        choices=choice.tolist(), selection_counts=np.bincount(choice, minlength=7).tolist())


def pw_sample(object_history, hand_history, hand_future, canonical, identity=0):
    """Inputs already use the CURRENT object's fixed frame. No future object input.

    Native interleaved wrist/MCP/tip points are mapped to the PW semantic order;
    right-hand only, with the absent left hand explicitly masked.
    """
    history = np.asarray(object_history); hh = np.asarray(hand_history); hf = np.asarray(hand_future)
    if history.shape != (4, 4, 4) or hh.shape != (4, 11, 3) or hf.shape != (24, 11, 3):
        raise ValueError('four current/past states and24 observed right-hand states required')
    if not np.allclose(history[-1], np.eye(4), atol=1e-5):
        raise ValueError('PW requires one fixed current-object frame')
    p, normal = canonical['points'], canonical['normals']
    ph = np.einsum('tij,pj->tpi', history[:, :3, :3], p) + history[:, None, :3, 3]
    h = np.zeros((4, 22, 3), np.float32); h[:, :11] = hh[:, PW_ORDER]
    future = np.zeros((24, 22, 3), np.float32); future[:, :11] = hf[:, PW_ORDER]
    action = np.concatenate((future, future-h[-1],
                             np.diff(np.concatenate((h[-1:], future)), axis=0)), -1)
    valid = np.zeros((24, 22), bool); valid[:, :11] = True
    action[~valid] = 0
    static = float(np.max(np.linalg.norm(ph[-1]-ph[0], axis=-1)) < .002)
    obj_extra = np.tile([1, static, 0, 0, 0, 0], (len(p), 1))
    obj_features = np.concatenate((normal, (ph[-1]-ph[-2])*30,
        (ph[-1]-2*ph[-2]+ph[-3])*900, ph[-1]-ph[0], obj_extra), -1)
    extra = np.stack((np.zeros(22), np.zeros(22), np.ones(22), np.repeat([0, 1], 11),
                      np.tile(np.arange(11)/10, 2), np.zeros(22)), -1)
    hand_features = np.concatenate((np.zeros((22, 3)), (h[-1]-h[-2])*30,
        (h[-1]-2*h[-2]+h[-3])*900, h[-1]-h[0], extra), -1)
    radius = float(canonical['radius'])
    return dict(xyz=np.concatenate((p, h[-1, :11])).astype('float32'),
        features=np.concatenate((obj_features, hand_features[:11])).astype('float32'),
        scene_object=np.r_[np.zeros(len(p), np.int64), np.full(11, -1, np.int64)],
        action=action.astype('float32'), action_valid=valid, points=p[None].astype('float32'),
        radius=np.array([radius], np.float32),
        object_features=np.r_[canonical['center'], np.eye(3).ravel(), radius, 1, 0][None].astype('float32'),
        effect=np.tile(np.eye(4, dtype=np.float32), (1, 24, 1, 1)), category=0,
        sample_id=np.array([identity, 0, 0], np.int64), hand_presence=np.array([True, False]))


def future_from_prediction(rotation, translation, hand_future):
    """Both arms contain the same measured hand future; only object future differs."""
    rotation = np.asarray(rotation); translation = np.asarray(translation)
    hand_future = np.asarray(hand_future)
    if rotation.shape != (24, 3, 3) or translation.shape != (24, 3) or hand_future.shape != (24, 11, 3):
        raise ValueError('full24step prediction required')
    relative = np.einsum('tji,tkj->tki', rotation, hand_future-translation[:, None])
    effect = np.concatenate((rotation, translation[..., None]), -1).reshape(24, 12)
    result = np.concatenate((effect, relative.reshape(24, 33)), -1).astype('float32')
    if not np.isfinite(result).all():
        raise ValueError('nonfinite PW future')
    return result
