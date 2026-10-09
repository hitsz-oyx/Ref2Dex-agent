"""Native action-chunk proposal contract for the physical consequence route.

This module is intentionally independent of Isaac Gym.  It turns a complete
continuous rollout into non-crossing ``H_t -> action[t:t+24]`` examples and
provides a small ACT-like proposal head.  The target is the action captured at
the native ``pre_physics_step`` boundary, not a residual plan or a hand flow.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .contracts import K


ACTION_CHUNK_SCHEMA = 'ref2dex.consequence-evaluator.native-action-chunks.v1'
ACTION_DIM = 18
EXECUTED_ACTION_SEMANTICS = 'native_post_noise_pre_physics_control'
DEFAULT_STRIDE = 4
NATIVE_RMS_EPSILON = 1e-5
ACTION_CHUNK_MODES = ('open_loop24', 'receding8', 'overlap8', 'temporal1')


class ActionChunkExecutor:
    """Dispatch native controls from causal, absolute-tick chunk predictions.

    ``overlap8`` changes aggregation only relative to ``receding8``;
    ``temporal1`` also queries every step. Exponential weights follow the
    official ACT implementation: oldest covering prediction first, weight
    exp(-decay * rank). Membership uses timestamps, never a zero sentinel.
    One executor belongs to one synchronous episode; reset before reuse.
    """

    def __init__(self, mode, *, decay=.01):
        if mode not in ACTION_CHUNK_MODES or not np.isfinite(decay) or decay < 0:
            raise ValueError('invalid action chunk execution mode/decay')
        self.mode = mode
        self.decay = float(decay)
        self.period = {'open_loop24': 24, 'receding8': 8,
                       'overlap8': 8, 'temporal1': 1}[mode]
        self.reset()

    def reset(self):
        self._chunks = []
        self._last_query = -1
        self.active_count = 0

    def should_query(self, tick):
        return tick >= 0 and tick % self.period == 0

    def add(self, tick, prediction):
        if not self.should_query(tick) or tick <= self._last_query:
            raise ValueError('chunk query must be on schedule and strictly increasing')
        if (prediction.ndim != 3 or prediction.shape[1:] != (K, ACTION_DIM)
                or not torch.isfinite(prediction).all() or prediction.abs().max() > 1+1e-6):
            raise ValueError('finite bounded [N,24,18] native chunk required')
        if self._chunks and prediction.shape != self._chunks[-1][1].shape:
            raise ValueError('chunk batch changed within episode')
        if self.mode in ('open_loop24', 'receding8'):
            self._chunks.clear()
        self._chunks.append((int(tick), prediction.detach().clone()))
        self._last_query = int(tick)

    def action(self, tick):
        # At most24 predictions are retained even for arbitrarily long episodes.
        self._chunks = [(start, plan) for start, plan in self._chunks if start+K > tick]
        covering = [(start, plan) for start, plan in self._chunks if start <= tick]
        self.active_count = len(covering)
        if not covering:
            raise RuntimeError('no causal action chunk covers tick%d' % tick)
        if self.mode in ('open_loop24', 'receding8'):
            start, plan = covering[-1]
            return plan[:, tick-start].clone()
        values = torch.stack([plan[:,tick-start] for start, plan in covering])
        weights = torch.exp(-self.decay * torch.arange(
            len(covering), device=values.device, dtype=values.dtype))
        weights = weights / weights.sum()
        return (values * weights[:,None,None]).sum(0)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def validate_action_episode(history, action, *, chunk=K):
    """Validate one complete raw episode without padding or reset stitching."""
    history = np.asarray(history)
    action = np.asarray(action)
    if history.ndim < 2 or history.shape[0] != action.shape[0] + 1:
        raise ValueError('history must contain exactly one more frame than action')
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError('native executed action must have shape [T,18]')
    if len(action) < int(chunk):
        raise ValueError('episode is shorter than one action chunk')
    if (history.dtype.kind not in 'biuf' or action.dtype.kind not in 'biuf'
            or not np.isfinite(history).all() or not np.isfinite(action).all()):
        raise ValueError('history/action must be finite numeric arrays')
    if np.abs(action).max() > 1 + 1e-6:
        raise ValueError('native executed action exceeds [-1,1]')
    return history.astype('float32', copy=False), action.astype('float32', copy=False)


def chunk_windows(history, action, *, chunk=K, stride=DEFAULT_STRIDE):
    """Return ``H_t`` and the native action chunk beginning at the same tick."""
    if not isinstance(chunk, (int, np.integer)) or int(chunk) != chunk or int(chunk) <= 0:
        raise ValueError('chunk must be a positive integer')
    if not isinstance(stride, (int, np.integer)) or int(stride) != stride or int(stride) <= 0:
        raise ValueError('stride must be a positive integer')
    history, action = validate_action_episode(history, action, chunk=int(chunk))
    starts = np.arange(0, len(action) - int(chunk) + 1, int(stride), dtype=np.int64)
    return history[starts].copy(), np.stack([action[t:t + int(chunk)] for t in starts]).astype('float32')


@dataclass(frozen=True)
class ActionChunkBatch:
    history: np.ndarray
    action: np.ndarray
    episode: np.ndarray
    tick: np.ndarray

    def __post_init__(self):
        n = len(self.history)
        if (self.history.ndim < 2 or self.action.shape != (n, K, ACTION_DIM)
                or self.episode.shape != (n,) or self.tick.shape != (n,)
                or not np.isfinite(self.history).all() or not np.isfinite(self.action).all()):
            raise ValueError('action-chunk batch contract mismatch')


def load_action_chunk_batches(root, *, qualities=('expert_success',), stride=DEFAULT_STRIDE,
                              splits=('train', 'val', 'test'), experts=None, tasks=None,
                              motions=None, allow_audit_only=False, clean_only=False):
    """Load episode-separated clean native chunks from a labeled continuous run.

    The manifest's historical ``action_semantics`` describes the evaluator's
    residual-plan field.  This route deliberately selects the separate
    ``action`` field and records its native capture semantics explicitly.
    """
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'ref2dex.consequence-evaluator.episodes.v2':
        raise ValueError('continuous episode schema required')
    if (manifest.get('status') != 'COMPLETED' or manifest.get('rollout_kind') != 'continuous'
            or manifest.get('action_semantics') != 'decision_known_requested_residual_plan'
            or manifest.get('fps') != 30 or manifest.get('units') != 'm'):
        raise ValueError('completed continuous episode run required')
    history_contract = manifest.get('history_contract')
    if (not isinstance(history_contract, dict) or history_contract.get('source') !=
            'raw native policy observation; per-expert RMS remains inside player'
            or history_contract.get('shape') != [1442]):
        raise ValueError('raw native history contract required')
    if manifest.get('training_allowed') is not True and not allow_audit_only:
        raise ValueError('audit-only or non-trainable episode source requires explicit engineering mode')
    qualities = tuple(str(value) for value in qualities)
    splits = tuple(str(value) for value in splits)
    if not qualities or not splits or any(value not in ('train', 'val', 'test') for value in splits):
        raise ValueError('nonempty quality and supported split filters required')
    batches = {}
    seen = set()
    for record in manifest.get('episodes', []):
        episode = str(record.get('episode'))
        split = str(record.get('split'))
        if (split not in splits or record.get('quality') not in qualities
                or (experts is not None and record.get('expert') not in set(experts))
                or (tasks is not None and record.get('task') not in set(tasks))
                or (motions is not None and record.get('motion') not in set(motions))):
            continue
        perturbation_tick = record.get('perturbation_tick', -1)
        if record.get('quality') == 'expert_success':
            try:
                success_tick = float(perturbation_tick)
            except (TypeError, ValueError):
                raise ValueError('expert_success must be an unperturbed clean episode')
            if (record.get('assigned_phase') != 'clean' or not np.isfinite(success_tick)
                    or success_tick != -1.0):
                raise ValueError('expert_success must be an unperturbed clean episode')
        if clean_only:
            # ``unlabeled`` is the normal quality for continuous collection, so
            # quality alone cannot prove that a trajectory was never intervened
            # on.  This is a filter: non-clean records are skipped, while a
            # record that claims to be clean is checked against its arrays below.
            try:
                clean_tick = float(perturbation_tick)
            except (TypeError, ValueError):
                continue
            if (not np.isfinite(clean_tick) or clean_tick != -1.0
                    or record.get('assigned_phase') != 'clean'):
                continue
        if episode in seen:
            raise ValueError('duplicate selected episode')
        seen.add(episode)
        path = root / str(record.get('path'))
        if not _within(path, root) or not path.exists():
            raise ValueError('episode path escapes or is missing: ' + str(path))
        if clean_only and (not isinstance(record.get('sha256'), str)
                           or len(record['sha256']) != 64):
            raise ValueError('clean-only episode requires a sha256 manifest entry: ' + episode)
        if record.get('sha256') and _sha(path) != record['sha256']:
            raise ValueError('episode file hash drift: ' + episode)
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {'history', 'action', 'residual_plan', 'plan_known',
                                   'hand_keypoints', 'object_pose', 'timestamps',
                                   'phase', 'progress', 'progress_mask'}:
                raise ValueError('continuous episode array whitelist mismatch')
            history, action = validate_action_episode(data['history'], data['action'])
            if history.shape[1] != int(history_contract['shape'][0]):
                raise ValueError('episode history dimension disagrees with manifest contract')
            # Clean demonstrations must not secretly carry a requested residual.
            if clean_only or record.get('quality') == 'expert_success':
                residual = data['residual_plan']
                if (residual.shape != (len(action), K, ACTION_DIM) or not np.isfinite(residual).all()
                        or np.abs(residual).max() > 1e-7):
                    label = 'clean' if clean_only else 'expert_success'
                    raise ValueError(label + ' episode contains nonzero or malformed residual plan: ' + episode)
            if clean_only:
                known = data['plan_known']
                if known.shape != (len(action),) or known.dtype.kind != 'b' or not known.all():
                    raise ValueError('clean episode has unknown residual-plan entries: ' + episode)
        h, a = chunk_windows(history, action, stride=stride)
        batches.setdefault(split, []).append((episode, h, a))
    result = {}
    for split in splits:
        rows = batches.get(split, [])
        if not rows:
            result[split] = ActionChunkBatch(
                np.empty((0, 0), dtype='float32'),
                np.empty((0, K, ACTION_DIM), dtype='float32'),
                np.empty((0,), dtype='<U1'), np.empty((0,), dtype=np.int64))
            continue
        hs, actions, episodes, ticks = [], [], [], []
        for episode, h, a in rows:
            hs.append(h); actions.append(a)
            episodes.extend([episode] * len(h)); ticks.extend(range(0, len(a) * int(stride), int(stride)))
        result[split] = ActionChunkBatch(
            np.concatenate(hs, axis=0), np.concatenate(actions, axis=0),
            np.asarray(episodes), np.asarray(ticks, dtype=np.int64))
    return result, manifest


@dataclass(frozen=True)
class HistoryStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    clip: Optional[float] = None

    @classmethod
    def fit(cls, history):
        history = np.asarray(history, dtype='float32')
        if history.ndim != 2 or len(history) == 0 or not np.isfinite(history).all():
            raise ValueError('nonempty finite [N,D] history required')
        mean = history.mean(axis=0, dtype=np.float64).astype('float32')
        scale = history.std(axis=0, dtype=np.float64).astype('float32')
        scale[scale < 1e-6] = 1.
        return cls(mean, scale)

    @classmethod
    def from_running_stats(cls, mean, variance, *, epsilon=NATIVE_RMS_EPSILON, clip=5.0):
        """Build the frozen normalizer used by a native policy checkpoint."""
        mean = np.asarray(mean, dtype='float32')
        variance = np.asarray(variance, dtype='float32')
        if (mean.ndim != 1 or variance.shape != mean.shape or not np.isfinite(mean).all()
                or not np.isfinite(variance).all() or (variance < 0).any() or epsilon <= 0):
            raise ValueError('invalid frozen running mean/variance')
        if clip is not None and (not np.isfinite(clip) or clip <= 0):
            raise ValueError('invalid native RMS clip')
        return cls(mean, np.sqrt(variance + float(epsilon)).astype('float32'), clip)

    def transform(self, history):
        history = np.asarray(history, dtype='float32')
        if history.shape[-1] != len(self.mean) or not np.isfinite(history).all():
            raise ValueError('history standardizer dimension/finite check failed')
        transformed = ((history - self.mean) / self.scale).astype('float32')
        return np.clip(transformed, -self.clip, self.clip) if self.clip is not None else transformed

    def as_dict(self):
        result = {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}
        if self.clip is not None:
            result['clip'] = float(self.clip)
        return result


class NativeActionChunkProposal(nn.Module):
    """Small one-shot chunk decoder; no future observation feedback is used."""

    def __init__(self, history_dim, *, width=128, layers=2, chunk=K):
        super().__init__()
        if (int(history_dim) <= 0 or int(width) <= 0 or int(width) % 4 != 0
                or int(layers) <= 0 or int(chunk) != K):
            raise ValueError('invalid native action chunk model dimensions')
        self.history_dim = int(history_dim)
        self.width = int(width)
        self.layers = int(layers)
        self.chunk = int(chunk)
        self.history = nn.Sequential(
            nn.Linear(self.history_dim, self.width), nn.LayerNorm(self.width), nn.GELU(),
            nn.Linear(self.width, self.width), nn.GELU())
        self.query = nn.Parameter(torch.zeros(1, self.chunk, self.width))
        self.time = nn.Parameter(torch.zeros(1, self.chunk, self.width))
        block = nn.TransformerEncoderLayer(self.width, 4, self.width * 4,
                                           dropout=0., batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(block, self.layers, enable_nested_tensor=False)
        self.output = nn.Linear(self.width, ACTION_DIM)
        nn.init.normal_(self.query, std=.02)
        nn.init.normal_(self.time, std=.02)

    def forward(self, history):
        if history.ndim != 2 or history.shape[-1] != self.history_dim:
            raise ValueError('expected [batch, history_dim] native history')
        context = self.history(history).unsqueeze(1)
        queries = self.query.expand(history.shape[0], -1, -1) + self.time
        decoded = self.decoder(torch.cat((context, queries), dim=1))[:, 1:]
        return torch.tanh(self.output(decoded))


def chunk_metrics(prediction, target, mean_target=None):
    """Return auditable action-space metrics; no task outcome is inferred."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != ACTION_DIM:
        raise ValueError('prediction/target action chunk shape mismatch')
    with torch.no_grad():
        error = prediction.detach().float() - target.detach().float()
        result = {'mse': float(error.square().mean()), 'mae': float(error.abs().mean()),
                  'first_action_mae': float(error[:, 0].abs().mean()),
                  'chunk_l2': float(error.flatten(1).norm(dim=1).mean())}
        if mean_target is not None:
            baseline = mean_target.to(prediction.device, prediction.dtype).view(1, K, ACTION_DIM)
            base_error = baseline - target.detach().to(prediction.device, prediction.dtype)
            result['mean_baseline_mse'] = float(base_error.square().mean())
            result['mse_ratio_to_mean'] = result['mse'] / result['mean_baseline_mse'] if result['mean_baseline_mse'] else None
        return result
