"""Fixed-endpoint current-only wrist MLP admission for diagnostic native execution."""
from pathlib import Path
import torch
from torch import nn


def load_wrist_probe(path, arm, device):
    packet = torch.load(Path(path), map_location='cpu', weights_only=False)
    if packet.get('step') != 6000 or packet.get('arm') != arm:
        raise ValueError('Fixed6000 current-wrist arm required')
    # Loading an override must not alter the native teacher/environment RNG stream.
    with torch.random.fork_rng(devices=[]):
        net = nn.Sequential(nn.Linear(811, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
                            nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 6))
    net.load_state_dict(packet['model'], strict=True)
    normalizers = {k: torch.as_tensor(packet[k], device=device) for k in ('xmean', 'xstd', 'ymean', 'ystd')}
    if any(not torch.isfinite(v).all() for v in normalizers.values()):
        raise ValueError('Finite normalizers required')
    if normalizers['xmean'].shape != (811,) or normalizers['ymean'].shape != (6,):
        raise ValueError('Wrist feature dimensions differ')
    if (normalizers['xstd'] <= 0).any() or (normalizers['ystd'] <= 0).any():
        raise ValueError('Positive scales required')
    return net.to(device).eval(), normalizers, arm


@torch.no_grad()
def predict_wrist(probe, x):
    net, stats, arm = probe
    if x.ndim != 2 or x.shape[1] != 811 or not torch.isfinite(x).all():
        raise ValueError('Current state/goal/delta0 feature811 required')
    z = (x-stats['xmean'])/stats['xstd']
    if arm == 'state_goal':
        z[:, 90:] = 0
    prediction = net(z)*stats['ystd']+stats['ymean']
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite wrist command')
    return prediction


def override_wrist(intended, wrist):
    if intended.shape != (3, 18) or wrist.shape != (2, 6):
        raise ValueError('Teacher plus two student command rows required')
    result = intended.clone()
    result[1:, :6] = wrist
    return result
