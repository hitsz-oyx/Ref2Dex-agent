"""Causal geometric reference progress; independent of old S/P/M labels."""
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .contracts import HAND_LINKS, K
from .data import validate_rigid
from .xirl_alignment import get_scaled_similarity, trailing_context_indices

RULE = 'causal-reference-clip-delta-progress-v1'
SCHEMA = 'ref2dex.consequence-reference-progress.labels.v1'
# Each group contributes equal mean-square distance. Floors have physical
# units and protect static coordinates; only reference data fit the scales.
GROUPS = (('object_translation', 3, .01), ('object_rotation', 9, .1),
          ('object_frame_hand_points', 33, .01), ('object_linear_motion', 3, .05),
          ('object_rotation_motion', 9, .5), ('hand_object_relative_motion', 33, .05))


@dataclass(frozen=True)
class AlignmentConfig:
    context: int = 8
    temperature: float = .01
    max_step: int = 4
    transition_scale: float = 1.5
    epsilon: float = .01
    max_cost: float = 1.0

    def __post_init__(self):
        if (not 2 <= self.context <= 32 or not 1 <= self.max_step <= 16
                or not 0 < self.temperature <= 1 or not 0 < self.transition_scale <= 16
                or not 0 < self.epsilon < 1 or not 0 < self.max_cost <= 100):
            raise ValueError('invalid bounded alignment configuration')

    def dictionary(self):
        return asdict(self)


def trajectory_features(poses, points, timestamps):
    """90D common 3D features, using frame-zero registration/backward motion.

    Register stationary task coordinates to the initial object pose, never
    to an episode-wide estimate. Hand points use each current object frame.
    Derivatives at t use only t and t-1, with zero motion at frame zero.
    No native reference clock, phase, force, outcome or quality is consumed.
    """
    poses = np.asarray(poses, np.float64)
    points = np.asarray(points, np.float64)
    timestamps = np.asarray(timestamps, np.float64)
    length = len(poses)
    if (poses.shape != (length, 4, 4) or length < 1
            or points.shape != (length, len(HAND_LINKS), 3)
            or timestamps.shape != (length,) or not np.isfinite(points).all()
            or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)):
        raise ValueError('finite aligned pose/11-point/clock trajectories required')
    validate_rigid(poses)
    relative = np.linalg.inv(poses[0]) @ poses
    local_points = np.einsum('tij,tkj->tki', poses[:, :3, :3].transpose(0, 2, 1),
                            points - poses[:, None, :3, 3]).reshape(length, -1)
    translation = relative[:, :3, 3]
    rotation = relative[:, :3, :3].reshape(length, -1)

    def motion(value):
        return np.concatenate((np.zeros((1, value.shape[1])),
                               np.diff(value, axis=0) / np.diff(timestamps)[:, None]))

    return np.concatenate((translation, rotation, local_points, motion(translation),
                           motion(rotation), motion(local_points)), axis=1)


class FeatureScale:
    """Fixed per-group scales fitted solely to supplied successful references."""
    def __init__(self, references):
        values = np.concatenate(references, axis=0)
        if values.ndim != 2 or values.shape[1] != sum(g[1] for g in GROUPS) or not np.isfinite(values).all():
            raise ValueError('finite common reference features required')
        self.center = values.mean(axis=0)
        self.scale = np.empty(values.shape[1], np.float64)
        self.weights = np.empty_like(self.scale)
        start = 0
        for _, size, floor in GROUPS:
            stop = start + size
            self.scale[start:stop] = max(float(np.sqrt(values[:, start:stop].var(axis=0).mean())), floor)
            self.weights[start:stop] = np.sqrt(values.shape[1] / (len(GROUPS) * size))
            start = stop

    def apply(self, features):
        result = (np.asarray(features) - self.center) / self.scale * self.weights
        if not np.isfinite(result).all():
            raise ValueError('nonfinite standardized geometry')
        return result

    def dictionary(self):
        return dict(center=self.center.tolist(), scale=self.scale.tolist(),
                    group_weights=self.weights.tolist(), fit_source='successful references only')


def clip_cost(features, reference, context=8, device='cpu'):
    """Upstream XIRL distance on causal flattened clips; one actual row per t."""
    actual = features[trailing_context_indices(len(features), context)].reshape(len(features), -1)
    ref = reference[trailing_context_indices(len(reference), context)].reshape(len(reference), -1)
    with torch.inference_mode():
        a = torch.as_tensor(actual, dtype=torch.float64, device=device)
        b = torch.as_tensor(ref, dtype=torch.float64, device=device)
        # -similarity at temperature=1 is dimension-normalized squared cost.
        return -get_scaled_similarity(a, b, 'l2', 1., True).cpu().numpy()


def track_cost(cost, config=AlignmentConfig()):
    """Filter q_t with an unbiased stay/forward/backward transition and clips.

    Start at reference zero because these collectors start at reference zero.
    All updates are forward filtering, never retrospective smoothing. Preserve
    alternative reachable phases in log space. If the observation would move
    the soft index more than max_step, project the posterior by exponential
    tilting onto that mean constraint (minimum KL change). Hard truncation
    around the mean would irreversibly delete alternative phase paths and can
    lock an otherwise useful matcher in an early local minimum. No t/N input.
    Backward motion is allowed, and there is no monotonic progress clamp.
    """
    cost = np.asarray(cost, np.float64)
    if (cost.ndim != 2 or cost.shape[0] < 1 or cost.shape[1] < 2
            or not np.isfinite(cost).all() or np.any(cost < -1e-8)):
        raise ValueError('finite nonnegative sequence cost matrix required')
    length, count = cost.shape
    index = np.arange(count, dtype=np.float64)
    offsets = np.arange(-config.max_step, config.max_step + 1)
    weights = np.exp(-np.abs(offsets) / config.transition_scale)
    normalizer = np.zeros(count)
    for offset, weight in zip(offsets, weights):
        normalizer += weight * ((index + offset >= 0) & (index + offset < count))
    q = np.zeros((length, count), np.float64)
    q[0, 0] = 1.
    log_previous = np.full(count, -np.inf)
    log_previous[0] = 0.

    def posterior(logits):
        peak = np.max(logits)
        probability = np.exp(logits - peak)
        probability /= probability.sum()
        return probability, float(probability @ index)

    for tick in range(1, length):
        center = float(q[tick - 1] @ index)
        log_prior = np.full(count, -np.inf)
        for offset, weight in zip(offsets, weights):
            source = np.flatnonzero((index + offset >= 0) & (index + offset < count))
            destination = source + offset
            log_prior[destination] = np.logaddexp(log_prior[destination],
                log_previous[source] + np.log(weight) - np.log(normalizer[source]))
        logits = log_prior - cost[tick] / config.temperature
        probability, mean = posterior(logits)
        target = float(np.clip(mean, max(0., center - config.max_step),
                               min(count - 1., center + config.max_step)))
        if abs(mean - target) > 1e-10:
            lower, upper = -1., 1.
            while posterior(logits + lower * index)[1] > target:
                lower *= 2
            while posterior(logits + upper * index)[1] < target:
                upper *= 2
            for _ in range(55):
                tilt = (lower + upper) / 2
                if posterior(logits + tilt * index)[1] < target:
                    lower = tilt
                else:
                    upper = tilt
            logits = logits + (lower + upper) / 2 * index
            probability, _ = posterior(logits)
        peak = np.max(logits)
        log_previous = logits - peak - np.log(np.exp(logits - peak).sum())
        q[tick] = probability
    progress = q @ (index / (count - 1))
    mean_index = progress * (count - 1)
    return dict(progress=progress, distribution=q,
                index_std=np.sqrt(np.sum(q * (index[None] - mean_index[:, None]) ** 2, axis=1)),
                matched_cost=np.sum(q * cost, axis=1),
                unconstrained_index=np.argmin(cost, axis=1))


class ReferenceProgress:
    def __init__(self, reference_features, config=AlignmentConfig(), device='cpu'):
        if len(reference_features) < 2:
            raise ValueError('at least two reference frames required')
        self.config, self.device = config, device
        self.normalizer = FeatureScale([reference_features])
        self.reference = self.normalizer.apply(reference_features)

    def align(self, features):
        return track_cost(clip_cost(self.normalizer.apply(features), self.reference,
                                   self.config.context, self.device), self.config)


def label_window(trace, tick, config=AlignmentConfig()):
    """Only prefix-filtered P[t] and P[t+24] define the executed candidate value."""
    end = tick + K
    if tick < 0 or end >= len(trace['progress']):
        return None
    value = float(trace['progress'][end] - trace['progress'][tick])
    return dict(value=value, progress_start=float(trace['progress'][tick]),
                progress_end=float(trace['progress'][end]),
                match_valid=bool(max(trace['matched_cost'][tick], trace['matched_cost'][end]) <= config.max_cost))


def preference(first, second, epsilon=.01):
    """Caller must separately establish common task/reference/current context."""
    if not 0 < epsilon < 1 or not np.isfinite([first, second]).all():
        raise ValueError('finite values and a positive abstention epsilon required')
    difference = first - second
    return 1 if difference > epsilon else -1 if difference < -epsilon else 0
