"""Current SDK support observations; no future state or learned Cm input."""
from pathlib import Path
import json
import numpy as np
import torch
from scripts.analyze_static_hold_feasibility import rotation
from src.task.CmResidual.measured_geometry_barriers import relative_geometry
from src.task.CmResidual.truth_successor_value import read_panel as read_compact

SCHEMA = 'observed-support-value-current72-sdk80-v1'
VARIANTS = ('compact_value', 'support_value')


def read_panel(directory):
    directory = Path(directory)
    panel = read_compact(directory)
    mask = panel['tick'] >= 2
    total = panel['eligible_rows']
    panel = {k: v[mask] if torch.is_tensor(v) and v.shape[:1] == (total,) else v
             for k, v in panel.items()}
    initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=False)
    trace = torch.load(directory / 'trace.pt', map_location='cpu', weights_only=False)
    metadata = json.loads((directory / 'physical_metadata.json').read_text())
    names = [metadata['native_body_names'][k] for k in metadata['contact_body_ids']]
    assert names == ['index_intermediate', 'middle_intermediate', 'pinky_intermediate',
                     'ring_intermediate', 'thumb_distal']
    ticks, envs = panel['tick'].numpy(), panel['environment'].numpy()
    root = trace['object_root'].numpy()
    current, previous = root[ticks - 1, envs].astype(np.float64), root[ticks - 2, envs].astype(np.float64)
    assert np.max(np.abs(current - trace['context'].numpy()[ticks, envs, 36:49])) < 1e-7
    positions, orientations = relative_geometry(
        current, trace['hand_body_position'].numpy()[ticks - 1, envs],
        trace['hand_body_quaternion'].numpy()[ticks - 1, envs])
    older_positions, _ = relative_geometry(
        previous, trace['hand_body_position'].numpy()[ticks - 2, envs],
        trace['hand_body_quaternion'].numpy()[ticks - 2, envs])
    force = np.concatenate((trace['object_force'].numpy()[ticks - 1, envs, None],
                            trace['hand_force'].numpy()[ticks - 1, envs]), axis=1).astype(np.float64)
    weight = np.array([p['mass'] for p in metadata['object_body_properties']]) * np.linalg.norm(metadata['gravity'])
    assert (weight > 0).all()
    force = np.einsum('nij,nbj->nbi', rotation(current[:, 3:7]).swapaxes(-1, -2), force)
    force /= weight[envs, None, None]
    clearance = trace['clearance'].numpy()
    increments = np.stack((current[:, 2] - previous[:, 2],
                           clearance[ticks - 1, envs] - clearance[ticks - 2, envs]), -1) / .005
    extra = np.concatenate((positions.reshape(-1, 15), orientations, force.reshape(-1, 18),
                            increments, (positions - older_positions).reshape(-1, 15)), -1).astype(np.float32)
    assert extra.shape == (len(ticks), 80) and np.isfinite(extra).all()
    panel['extra'] = torch.from_numpy(extra)
    panel['eligible_rows'] = len(ticks)
    panel['episodes'] = int(panel['environment'].unique().numel())
    return panel


def features(panel, mean, std, variant):
    extra = np.clip((panel['extra'].numpy() - mean) / std, -10, 10)
    if variant == 'compact_value':
        extra = np.zeros_like(extra)
    assert variant in VARIANTS
    return np.concatenate((panel['current'].numpy(), extra), -1).astype(np.float32)


def initialized_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3541)
        return torch.nn.Sequential(torch.nn.Linear(152, 64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, 1), torch.nn.Sigmoid())


def phase_template(fit):
    labels = fit['target'].numpy()
    table = np.full((3, 16), labels.mean(), dtype=np.float32)
    for motion in range(3):
        for timebin in range(16):
            take = (fit['motion'].numpy() == motion) & (fit['timebin'].numpy() == timebin)
            if take.any():
                table[motion, timebin] = labels[take].mean()
    return table


def analyze(panel, predictions):
    labels = panel['target'].numpy().astype(np.float64)
    cluster = panel['cluster'].numpy()
    weight = np.bincount(cluster, minlength=192)
    risk = {k: (np.asarray(v, dtype=np.float64) - labels) ** 2 for k, v in predictions.items()}
    brier = {k: float(v.mean()) for k, v in risk.items()}
    indices = np.random.default_rng(3543).integers(0, 192, size=(2000, 192))
    intervals, gates = {}, {}
    for control in ('compact_value', 'motion_time'):
        diff = np.bincount(cluster, weights=risk['support_value'] - risk[control], minlength=192)
        ci = np.quantile(diff[indices].sum(1) / weight[indices].sum(1), [.025, .975]).tolist()
        intervals[control] = ci
        gates[control] = dict(gain1percent=brier['support_value'] <= .99 * brier[control],
                              paired_upper95_negative=ci[1] < 0)
    return dict(brier=brier, paired_difference_interval95=intervals, gates=gates,
                label='PROMISING' if all(all(g.values()) for g in gates.values()) else 'UNPROMISING')
