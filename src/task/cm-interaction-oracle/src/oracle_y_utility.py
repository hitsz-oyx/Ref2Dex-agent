"""Ref13 paired candidate utility, bounded stable-grasp target and gate logic."""
import numpy as np
import torch
from intervention import per_finger_residuals

CANDIDATES = ('baseline', 'thumb_yaw_plus', 'thumb_yaw_minus', 'middle_plus',
              'middle_minus', 'grip_plus', 'wrist_z_plus')
POST_WINDOW = 90


def candidate_deltas(device='cpu'):
    original = per_finger_residuals(.10, device)
    value = original[[0, 9, 10, 3, 4, 13]].clone()
    wrist = torch.zeros(1, 18, device=device); wrist[:, 2] = .01
    return torch.cat((value, wrist), 0)


def utility(y):
    """Fixed before outcomes: later held fraction + contact bonus - height failure."""
    return y[..., 7]+.25*y[..., 3]-y[..., 6]


def stable_grasp_z(height, pair, rest):
    """Qualify45 consecutive held steps bystep60; no subsequent drop tostep90.

    Guarantees >=30 post-qualification observations. Uses physical height and
    declared force-pair proxy; it is not episode-wide success or certified friction.
    """
    if height.shape[-1] != POST_WINDOW or pair.shape != height.shape:
        raise ValueError('Z requires exactly90post-decision steps')
    lifted = height-rest[..., None]
    held = (lifted >= .03) & pair.bool()
    count = torch.zeros(height.shape[:-1], dtype=torch.long, device=height.device)
    lost = torch.zeros_like(count); first = torch.full_like(count, -1)
    dropped = torch.zeros_like(count, dtype=torch.bool)
    for tick in range(POST_WINDOW):
        previous = first >= 0
        lost = torch.where(pair[..., tick].bool(), 0, lost+1)
        count = torch.where(held[..., tick], count+1, 0)
        first = torch.where((first < 0) & (count >= 45) & (tick < 60), tick+1, first)
        dropped |= previous & ((lifted[..., tick] < .02) | (lost >= 6))
    return (first >= 0) & ~dropped, dict(first_stable_step=first, dropped_after_qualification=dropped)


def paired_bootstrap(difference, seed=265, repeats=2000):
    difference = np.asarray(difference, dtype=float)
    if not len(difference): return dict(gain=None, lower95=None, upper95=None, anchors=0)
    means = difference[np.random.default_rng(seed).integers(len(difference), size=(repeats, len(difference)))].mean(1)
    return dict(gain=float(difference.mean()), lower95=float(np.quantile(means,.025)),
                upper95=float(np.quantile(means,.975)), anchors=len(difference))


def select(scores):
    """Baseline has index0; exact utility ties select baseline before alternatives."""
    return np.argmax(scores, axis=-1)


def oracle_gate(y, z, motion):
    score = utility(np.asarray(y)); z = np.asarray(z, dtype=float)
    if score.ndim != 2 or z.shape != score.shape or score.shape[1] != len(CANDIDATES):
        raise ValueError('complete same-state candidate panel required')
    chosen = select(score); rows = np.arange(len(score))
    oracle = z[rows, chosen]; baseline = z[:, 0]; upper = z.max(1)
    gain = paired_bootstrap(oracle-baseline)
    opportunity = paired_bootstrap(upper-baseline)
    adequate = len(score) >= 30 and len(np.unique(motion)) >= 2
    passed = adequate and gain['gain'] >= .05 and gain['lower95'] > 0
    return dict(status='PROMISING' if passed else ('UNPROMISING' if adequate else 'UNCLEAR'),
                passed=bool(passed), adequate_support=adequate, selection=chosen.tolist(),
                baseline_success=float(baseline.mean()) if len(z) else None,
                oracle_y_success=float(oracle.mean()) if len(z) else None,
                oracle_z_upper_success=float(upper.mean()) if len(z) else None,
                gain=gain, candidate_opportunity=opportunity,
                selection_counts=np.bincount(chosen, minlength=len(CANDIDATES)).tolist(),
                interpretation='Finite candidate/anchor Probe; absent candidate Z opportunity does not refute Y globally.')


def noise_curve(y, z, sigmas=(0,.025,.05,.1,.2,.4,.8), repeats=1000, seed=266):
    """GateB: raw GT Y plus independent Gaussian noise, no learned predictor."""
    y = np.asarray(y); z = np.asarray(z, dtype=float); score = utility(y)
    rows = np.arange(len(y)); base = z[:, 0]; oracle = z[rows, select(score)]
    denominator = float((oracle-base).mean())
    left, right = np.triu_indices(score.shape[1], 1)
    truth = score[:, left]-score[:, right]; active = np.abs(truth) >= .02
    rng = np.random.default_rng(seed); common_noise = rng.standard_normal((repeats, *y.shape))
    result = []
    for sigma in sigmas:
        noisy = y[None]+sigma*common_noise
        value = utility(noisy); picked = select(value)
        selected_z = z[rows[None], picked]
        products = (value[..., left]-value[..., right])*truth[None]
        ordering = float(((products>0)+.5*(products==0))[:,active].mean()) if active.any() else None
        regret = score.max(1)[None]-score[rows[None], picked]
        delta = selected_z.mean(0)-base
        result.append(dict(sigma=float(sigma),raw_y_rmse=float(np.sqrt(np.mean((noisy-y[None])**2))),
                           pairwise_accuracy=ordering,active_pairs=int(active.sum()),
                           top1_regret=float(regret.mean()),selected_z=float(selected_z.mean()),
                           gain=paired_bootstrap(delta),
                           retained_oracle_gain=float(delta.mean()/denominator) if denominator>0 else None))
    return dict(rows=result,repeats=repeats,seed=seed,noise='Independent raw-Y Gaussian components; unbounded/no clipping, common draws across sigma.',
                limits='Synthetic within-panel sensitivity; not a universal accuracy threshold or trained-model generalization guarantee.')
