"""Conditional residual flow for joint hand/object future trajectories.

The observation backbone is evaluated once. A small vector field mixes all74
hand/object points within each horizon token and attends across24 future ticks
and observed context. Supervised targets enter only the FM training path;
coordinate/contact losses always see a genuine noise-to-trajectory ODE sample.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

FLOW_ENCODING = 'v6.joint-hand42-object32.residual-cfm.linear-euler.v1'
OVERNIGHT_CAMPAIGN = 'v6.user-overnight-fm-20261022'


def validate_flow_config(config):
    if not config.get('flow_matching'):
        return
    if (config.get('flow_encoding') != FLOW_ENCODING
            or not 1 <= config.get('flow_steps', 0) <= 16
            or not 0 < config.get('flow_noise_std', 0) <= 1
            or not 0 < config.get('flow_matching_weight', 0) <= 2
            or not 0 < config.get('flow_lr', 0) <= .001
            or config.get('flow_width') not in (128, 256)
            or not 1 <= config.get('flow_layers', 0) <= 4
            or not isinstance(config.get('flow_eval_seed'), int)):
        raise ValueError('explicit bounded conditional flow configuration required')


def pack(hand, scene):
    """B,T,2,21,3 and B,1,T,32,3 -> B,T,222, preserving point identities."""
    return torch.cat((hand.flatten(2), scene[:, 0].flatten(2)), -1)


def unpack(value):
    return (value[..., :126].reshape(len(value), 24, 2, 21, 3),
            value[..., 126:].reshape(len(value), 1, 24, 32, 3))


def current_mask(batch):
    hand = batch['start_hand_valid'][..., None].expand(-1, -1, -1, 3).flatten(1)
    scene = batch['object_valid'][:, :1, None].expand(-1, 32, 3).flatten(1)
    return torch.cat((hand, scene), -1)[:, None]


def linear_path(target, noise, time):
    t = time[:, None, None]
    return (1-t)*noise+t*target, target-noise


def euler_sample(field, noise, steps):
    value = noise
    for i in range(steps):
        t = value.new_full((len(value),), i/steps)
        value = value+field(value, t)/steps
    return value


class TrajectoryFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_flow_config(config)
        self.config = config
        width = config['flow_width']
        self.input = nn.Linear(444, width)
        self.context = nn.Linear(config['shared_width'], width)
        self.horizon = nn.Embedding(24, width)
        self.time = nn.Sequential(nn.Linear(33, width), nn.SiLU(), nn.Linear(width, width))
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(width, 4, 4*width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
            for _ in range(config['flow_layers'])])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 222)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def velocity(self, value, time, base, context, context_valid, mask):
        frequency = torch.arange(16, device=value.device, dtype=torch.float32)
        angle = time[:, None]*(2.**(frequency/4))*math.pi
        time_features = torch.cat((time[:, None], angle.sin(), angle.cos()), -1)
        hidden = (self.input(torch.cat((value, base), -1)) + self.horizon.weight[None]
                  + self.time(time_features)[:, None])
        for layer in self.layers:
            if self.training and torch.is_grad_enabled() and self.config.get('flow_checkpoint', True):
                hidden = checkpoint(layer, hidden, context, memory_key_padding_mask=~context_valid,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                hidden = layer(hidden, context, memory_key_padding_mask=~context_valid)
        return torch.where(mask, self.output(self.norm(hidden)).float(), 0.)

    def noise(self, base, batch):
        if self.training:
            return torch.randn_like(base)*self.config['flow_noise_std']
        # Fixed per-sample draw for unbiased checkpoint comparisons; no global
        # RNG consumption during DEV and no dependence on future target values.
        ids = batch.get('sample_id')
        ids = ids.detach().cpu().tolist() if ids is not None else [[0, 0, 0]]*len(base)
        values = []
        for identity in ids:
            seed = self.config['flow_eval_seed']
            for number in identity:
                seed = (seed*1000003+int(number)) % (2**63-1)
            generator = torch.Generator(device=base.device).manual_seed(seed)
            values.append(torch.randn(base.shape[1:], device=base.device,
                                      dtype=base.dtype, generator=generator))
        return torch.stack(values)*self.config['flow_noise_std']

    def forward(self, hand_base, scene_base, shared, shared_valid, batch, normalizer):
        mask = current_mask(batch)
        base = torch.where(mask, pack(hand_base, scene_base), 0.)
        context = self.context(shared)
        noise = torch.where(mask, self.noise(base, batch), 0.)
        field = lambda value, time: self.velocity(value, time, base, context, shared_valid, mask)
        residual = euler_sample(field, noise, self.config['flow_steps'])
        hand, scene = unpack(torch.where(mask, base+residual, 0.))
        extras = {}
        if self.training and torch.is_grad_enabled():
            hv = batch['trajectory_valid'] & batch['start_hand_valid'][:, None]
            sv = batch['scene_valid'] & batch['object_valid'][:, :, None, None]
            htarget = normalizer.normalize('hand', torch.where(hv[..., None], batch['trajectory_target'].float(), 0.))
            starget = normalizer.normalize('scene', torch.where(sv[..., None], batch['scene_target'].float(), 0.))
            valid = pack(hv[..., None].expand_as(htarget), sv[..., None].expand_as(starget))
            target = torch.where(valid, pack(htarget, starget)-base.detach(), 0.)
            time = torch.rand(len(base), device=base.device)
            mixed, target_velocity = linear_path(target, noise, time)
            predicted_velocity = field(mixed, time)
            hp, sp = unpack(predicted_velocity);ht, st = unpack(target_velocity)
            extras = dict(flow_hand_velocity=hp, flow_scene_velocity=sp,
                          flow_hand_target_velocity=ht.detach(), flow_scene_target_velocity=st.detach())
        return hand, scene, extras
