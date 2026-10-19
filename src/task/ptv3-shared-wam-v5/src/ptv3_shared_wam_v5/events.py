"""Optional next planned CONTACT/RELEASE condition on the30Hz WAM timeline."""
import math
import torch
from torch import nn
import torch.nn.functional as F

EVENT_ENCODING='next-contact-release-plan.frame30.v1'
EVENT_TYPES={'NONE':0,'CONTACT':1,'RELEASE':2}


def event_packet(kind,delta_steps,valid,*,hz=30):
    """Boundary adapter for a scheduler; invalid means unknown, not known NONE."""
    if hz!=30 or kind not in EVENT_TYPES:
        raise ValueError('known event type and30Hz WAM event time required')
    if valid and (not math.isfinite(delta_steps) or delta_steps<0):
        raise ValueError('valid planned event requires nonnegative relative frames')
    return dict(event_type=EVENT_TYPES[kind],event_delta_steps=float(delta_steps) if valid else 0.,
                event_valid=bool(valid),event_hz=30)


def event_features(batch,horizon=24):
    required=('event_type','event_delta_steps','event_valid','event_hz')
    if any(k not in batch for k in required):
        raise ValueError('event-conditioned WAM requires explicit type/delta/valid/hz fields')
    kind,delta,valid,hz=(batch[k] for k in required)
    if kind.ndim!=1 or any(x.shape!=kind.shape for x in (delta,valid,hz)):
        raise ValueError('one next-event slot per target sample required')
    if 'xyz' in batch and len(kind)!=len(batch['xyz']):
        raise ValueError('one next-event slot per target sample required; no broadcast')
    if kind.dtype not in (torch.int32,torch.int64) or valid.dtype!=torch.bool:
        raise ValueError('integer event type and boolean validity required')
    if ((hz!=30)&valid).any() or ((kind<0)|(kind>2))[valid].any():
        raise ValueError('valid event type0..2 and30Hz required')
    if (not torch.isfinite(delta[valid]).all()) or (delta[valid]<0).any():
        raise ValueError('finite nonnegative planned relative frames required')
    if ((kind==0)&valid&(delta!=0)).any():
        raise ValueError('known NONE requires zero delta; far events retain their type')
    kind=torch.where(valid,kind,0).long()
    delta=torch.where(valid,delta.float(),0.)
    # Retain next events outside K24, capped at8s; do not label them NONE.
    values=torch.cat((F.one_hot(kind,3).float(),delta.clamp(max=240)[:,None]/horizon,
                      (delta>horizon).float()[:,None]),-1)
    return torch.where(valid[:,None],values,0.),valid


class EventGoalResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(5,64),nn.GELU(),nn.Linear(64,width))
        nn.init.zeros_(self.encoder[-1].weight);nn.init.zeros_(self.encoder[-1].bias)

    def forward(self,batch):
        features,valid=event_features(batch)
        residual=self.encoder(features)
        return torch.where(valid[:,None],residual,0.)


def initialize_event_model(model,base_state):
    """Explicit weights-only expansion; full optimizer migration is separate."""
    if not getattr(model,'event_conditioning',False):
        raise ValueError('event-enabled target required')
    target=model.state_dict()
    expected={k for k in target if not k.startswith('event_goal.')}
    if set(base_state)!=expected or any(target[k].shape!=v.shape for k,v in base_state.items()):
        raise ValueError('base checkpoint must match every original parameter/buffer')
    result=model.load_state_dict(base_state,strict=False)
    assert all(k.startswith('event_goal.') for k in result.missing_keys) and not result.unexpected_keys
    nn.init.zeros_(model.event_goal.encoder[-1].weight)
    nn.init.zeros_(model.event_goal.encoder[-1].bias)
    return dict(kind='weights-only-event-expansion',missing_keys=result.missing_keys,
                optimizer_restored=False,encoding=EVENT_ENCODING)
