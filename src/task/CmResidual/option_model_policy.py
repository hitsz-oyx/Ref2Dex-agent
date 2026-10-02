"""Short physical successor and explicit held-option continuation value."""
from pathlib import Path
import json
import numpy as np
import torch
from scripts.analyze_static_hold_feasibility import rotation
from src.task.CmResidual.measured_geometry_barriers import relative_geometry

SCHEMA = 'option-model-policy-current152-physical131-known21-h8-v1'
HORIZON = 8
VARIANTS = ('cm', 'dynamics_off', 'direct_q')
DYNAMIC = np.r_[np.arange(51), np.arange(72, 152)].astype(np.int64)


def support_extra(root, older, body, older_body, quaternion, forces, weight, clearance_delta):
    root, older = np.asarray(root, np.float64), np.asarray(older, np.float64)
    positions, orientations = relative_geometry(root, body, quaternion)
    previous, _ = relative_geometry(older, older_body, quaternion)
    force = np.einsum('nij,nbj->nbi', rotation(root[:, 3:7]).swapaxes(-1, -2), forces.astype(np.float64))
    force /= np.asarray(weight, np.float64)[:, None, None]
    prior = np.stack((root[:, 2] - older[:, 2], clearance_delta), -1) / .005
    result = np.concatenate((positions.reshape(-1, 15), orientations, force.reshape(-1, 18),
                             prior, (positions - previous).reshape(-1, 15)), -1).astype(np.float32)
    assert result.shape == (len(root), 80) and np.isfinite(result).all()
    return result


def trace_extra(trace, ticks, environments, metadata):
    v = lambda key: trace[key].numpy()
    force = np.concatenate((v('object_force')[ticks - 1, environments, None],
                            v('hand_force')[ticks - 1, environments]), 1)
    weight = np.array([p['mass'] for p in metadata['object_body_properties']]) * np.linalg.norm(metadata['gravity'])
    return support_extra(v('object_root')[ticks - 1, environments], v('object_root')[ticks - 2, environments],
                         v('hand_body_position')[ticks - 1, environments], v('hand_body_position')[ticks - 2, environments],
                         v('hand_body_quaternion')[ticks - 1, environments], force, weight[environments],
                         v('clearance')[ticks - 1, environments] - v('clearance')[ticks - 2, environments])


def known_plan(initial, ticks, base):
    motion = initial['motion'].numpy()
    stop = initial['phase_stop'].numpy()[motion]
    progress = np.minimum(ticks + 1, stop)
    q = initial['native_reference_q'].numpy()[motion, progress]
    raw = np.concatenate((q, (progress / stop).astype(np.float32)[:, None]), -1)
    normalized = np.clip((raw - base['observation_mean'].numpy()[51:70]) /
                         base['observation_std'].numpy()[51:70], -10, 10)
    return np.concatenate((normalized, np.ones((len(ticks), 1), np.float32),
                           ((stop + 30 - ticks) / np.float32(202))[:, None]), -1).astype(np.float32)


def read_options(directory, base):
    directory = Path(directory)
    report = json.loads((directory / 'results.json').read_text())
    assert report['run_status'] == 'COMPLETED' and report['statistical_option_collection']
    assert json.loads((directory / 'panel_audit.json').read_text())['run_status'] == 'COMPLETED'
    initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=False)
    trace = torch.load(directory / 'trace.pt', map_location='cpu', weights_only=False)
    metadata = json.loads((directory / 'physical_metadata.json').read_text())
    env = np.arange(768)
    tick = initial['decision_steps'].numpy()
    stop = initial['phase_stop'].numpy()[initial['motion'].numpy()]
    assert (tick >= 2).all() and (tick + HORIZON < stop - 74).all()
    assert initial['independent_placements'] and not initial['paired_initial_offsets']
    roots = trace['object_root'].numpy()
    assert np.max(np.abs(roots[tick - 1, env] - trace['context'].numpy()[tick, env, 36:49])) <= 1e-7
    def compact(time):
        return np.concatenate((trace['normalized_context'].numpy()[time, env], np.ones((768, 1), np.float32),
                               ((stop + 30 - time) / np.float32(202))[:, None]), -1).astype(np.float32)
    current, future = compact(tick), compact(tick + HORIZON)
    known = known_plan(initial, tick + HORIZON, base)
    assert np.max(np.abs(future[:, 51:72] - known)) <= 2e-6
    labels = np.array([row['physical105'] for row in json.loads((directory / 'rows.json').read_text())], np.float32)
    progress = trace['progress'].numpy()
    valid = (roots[..., 2] - initial['initial_height'].numpy()[None] >= np.float32(.03)) & (trace['clearance'].numpy() >= np.float32(.02))
    window = (progress >= stop[None] - 74) & (progress <= stop[None] + 30)
    assert (window.sum(0) == 105).all() and np.array_equal(labels.astype(bool), ~(window & ~valid).any(0))
    return dict(current=current, extra=trace_extra(trace, tick, env, metadata),
                future=future, future_extra=trace_extra(trace, tick + HORIZON, env, metadata),
                known=known, raw=initial['option_raw12'].numpy(), reward=labels,
                environment=env, tick=tick, motion=initial['motion'].numpy(), arm=initial['policy_group'].numpy())


def state_features(compact, extra, mean, std):
    return np.concatenate((compact, np.clip((extra - mean) / std, -10, 10)), -1).astype(np.float32)


def initialized_network(kind):
    seed = 3551 if kind == 'dynamics' else (3553 if kind == 'value' else 3555)
    inputs, outputs = (164, 131) if kind == 'dynamics' else ((164, 1) if kind == 'value' else (152, 12))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = [torch.nn.Linear(inputs, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                  torch.nn.ReLU(), torch.nn.Linear(64, outputs)]
        if kind in ('actor', 'dynamics'):
            torch.nn.init.zeros_(layers[-1].weight)
            torch.nn.init.zeros_(layers[-1].bias)
        if kind == 'value':
            layers.append(torch.nn.Sigmoid())
        elif kind == 'actor':
            layers.append(torch.nn.Tanh())
        return torch.nn.Sequential(*layers)


def scores(current, raw, known, models, delta_mean, delta_std, variant):
    action = torch.tanh(raw)
    if variant == 'direct_q':
        return models['direct_q'](torch.cat((current, action), -1)).flatten()
    assert variant in ('cm', 'dynamics_off')
    model_action = action if variant == 'cm' else torch.zeros_like(action)
    residual = delta_mean + delta_std * models[variant](torch.cat((current, model_action), -1))
    successor = current.clone()
    successor[:, 51:72] = known
    successor[:, DYNAMIC] = (current[:, DYNAMIC] + residual).clamp(-10, 10)
    return models['value'](torch.cat((successor, action), -1)).flatten()


def numpy_forward(parameters, inputs, activation):
    x = np.asarray(inputs, dtype=np.float64)
    layers = sorted(int(k.split('.')[0]) for k in parameters if k.endswith('.weight'))
    for j, layer in enumerate(layers):
        x = x @ parameters[str(layer) + '.weight'].numpy().astype(np.float64).T + parameters[str(layer) + '.bias'].numpy()
        if j < len(layers) - 1:
            x = np.maximum(x, 0)
        elif activation == 'tanh':
            x = np.tanh(x)
        elif activation == 'sigmoid':
            x = 1 / (1 + np.exp(-x.clip(-700, 700)))
    return x
