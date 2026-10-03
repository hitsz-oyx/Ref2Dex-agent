"""Bounded common-surface motion prior; realized motion is offline input."""
import numpy as np
import torch
from torch import nn

SIZES = (512, 2048, 7168)
FLOW_COLUMNS = (12, 13, 14, 16, 17, 18)


def encode_raw(raw):
    rotation = raw['pose'][:, :3, :3]
    origin = raw['pose'][:, None, :3, 3]
    transform = lambda value: np.einsum('n...i,nij->n...j', value, rotation)
    obj = transform(raw['obj'] - origin)
    normal = transform(raw['normal'])
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    relative = transform(raw['hand'] - raw['obj'][:, :, None])
    hand_normal = transform(raw['hand_normal'])
    hand_normal /= np.maximum(np.linalg.norm(hand_normal, axis=-1, keepdims=True), 1e-8)
    flow = transform(raw['next_hand'] - raw['hand'])
    distance = np.linalg.norm(relative, axis=-1).min(-1)[..., None]
    global_flow = transform(raw['global_hand_flow'])[:, None]
    global_flow = np.broadcast_to(global_flow, obj.shape)
    previous = transform(raw['obj'] - raw['previous_obj'])
    features = np.concatenate((obj/.05, normal, relative.mean(2)/.05,
                               hand_normal.mean(2), flow.mean(2)/.01,
                               distance/.05, global_flow/.01, previous/.01), -1)
    target = transform(raw['next_obj'] - raw['obj'])/.01
    assert features.shape[-1] == 22
    assert np.isfinite(features).all() and np.isfinite(target).all()
    return features.astype(np.float32), target.astype(np.float32)


class SurfaceMotionPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(22, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, features):
        local = self.encoder(features)
        cm = local.mean(1, keepdim=True).expand_as(local)
        return self.decoder(torch.cat((local, cm), -1))


def numpy_predict(state, features):
    def linear(x, prefix):
        return x @ state[prefix+'.weight'].detach().cpu().numpy().T + state[prefix+'.bias'].detach().cpu().numpy()
    local = np.maximum(linear(features, 'encoder.0'), 0)
    local = np.maximum(linear(local, 'encoder.2'), 0)
    pooled = np.broadcast_to(local.mean(1, keepdims=True), local.shape)
    hidden = np.maximum(linear(np.concatenate((local, pooled), -1), 'decoder.0'), 0)
    return linear(hidden, 'decoder.2')


def metrics(prediction, target, parents):
    error = np.linalg.norm(prediction-target, axis=-1).mean(-1)*10
    grouped = {parent: float(error[np.asarray(parents)==parent].mean()) for parent in sorted(set(parents))}
    return dict(parent_epe_mm=float(np.mean(list(grouped.values()))), window_epe_mm=float(error.mean()),
                windows=len(error), parents=len(grouped), per_parent_epe_mm=grouped)


def classify(reports, persistence):
    gates = {hand+'_scale': reports[hand+'_7168']['parent_epe_mm'] <= .9*reports[hand+'_512']['parent_epe_mm']
             and reports[hand+'_2048']['parent_epe_mm'] <= 1.02*reports[hand+'_512']['parent_epe_mm'] for hand in ('mano', 'inspire')}
    error = reports['adapt_mano_7168']['parent_epe_mm']
    gates['cross_prior'] = (error <= .9*reports['adapt_scratch']['parent_epe_mm']
                            and error <= .95*reports['adapt_mano_motion_off']['parent_epe_mm']
                            and error <= .95*reports['adapt_mano_shuffled']['parent_epe_mm']
                            and error <= .95*persistence)
    label = 'PROMISING' if gates['cross_prior'] else ('UNCLEAR' if any(gates.values()) else 'UNPROMISING')
    return gates, label
