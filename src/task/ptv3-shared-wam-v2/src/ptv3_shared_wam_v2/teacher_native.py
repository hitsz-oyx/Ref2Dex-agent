"""Deploy the scratch teacher-input MLP without borrowing teacher actions."""
import torch
from torch import nn


class TeacherInputStudent(nn.Module):
    def __init__(self, packet):
        super().__init__()
        if packet['schema'] != 'ref2dex.full-teacher-MLP.v1' or packet['step'] != 24000:
            raise ValueError('Fixed final coverage checkpoint required')
        layers = []
        for n, m in zip([1442, 1024, 1024, 1024], [1024, 1024, 1024, 512]):
            layers.extend([nn.Linear(n, m), nn.ReLU()])
        self.model = nn.Sequential(*layers, nn.Linear(512, 12))
        self.model.load_state_dict(packet['model'], strict=True)
        for key, size in [('xmean', 1442), ('xscale', 1442), ('ymean', 12), ('yscale', 12)]:
            value = packet[key].detach().float().reshape(-1)
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError('Invalid normalization ' + key)
            if key.endswith('scale') and (value <= 0).any():
                raise ValueError('Nonpositive scale')
            self.register_buffer(key, value)
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, observation):
        if observation.ndim != 2 or observation.shape[-1] != 1442:
            raise ValueError('Raw native full1442 observations required')
        # Fixed 64-row GEMM avoids native-runtime singleton inference instability.
        if not 1 <= len(observation) <= 64:
            raise ValueError('At most 64 independent rows')
        padded = observation.repeat((63 // len(observation) + 1), 1)[:64]
        x = ((padded - self.xmean) / self.xscale).clamp(-5, 5)
        return (self.model(x) * self.yscale + self.ymean)[:len(observation)]
