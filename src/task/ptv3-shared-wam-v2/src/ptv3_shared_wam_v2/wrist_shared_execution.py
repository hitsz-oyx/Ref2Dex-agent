"""Live none/aligned wrist substitution with the exact frozen feature checkpoint."""
from pathlib import Path
import torch

from .wrist_shared_probe import MODEL_SCHEMA, WristSharedMLP

ARMS = ('none', 'aligned')  # Native rows1/2. Server request order is reversed.


def load_matched_wrist(path, arm, feature_checkpoint, device):
    if arm not in ARMS:raise ValueError('Only matched none/aligned arms are admitted')
    packet=torch.load(Path(path),map_location='cpu',weights_only=False)
    if packet.get('schema')!=MODEL_SCHEMA or packet.get('mode')!=arm or packet.get('step')!=6000:
        raise ValueError('Fixed6000 matched none/aligned checkpoint required')
    if Path(packet['feature_checkpoint']).resolve()!=Path(feature_checkpoint).resolve():
        raise ValueError('Live WAM must be the frozen feature source')
    if packet.get('feature_pooling')!='separate masked mean scene/hand/goal, concatenated2304D':
        raise ValueError('Exact2304D pooled feature contract required')
    # Loading the diagnostic wrist heads must preserve the native teacher RNG.
    with torch.random.fork_rng(devices=[]):
        model=WristSharedMLP()
    model.load_state_dict(packet['model'],strict=True)
    stats={k:torch.as_tensor(packet['stats'][k],device=device) for k in
           ('xmean','xstd','ymean','ystd','zmean','zstd')}
    expected={'xmean':811,'xstd':811,'ymean':6,'ystd':6,'zmean':2304,'zstd':2304}
    if any(v.shape!=(expected[k],) or not torch.isfinite(v).all() for k,v in stats.items()):
        raise ValueError('Finite matched wrist normalization shapes required')
    if any((stats[k]<=0).any() for k in ('xstd','ystd','zstd')):
        raise ValueError('Positive TRAIN-only scales required')
    return model.to(device).eval(),stats,arm


@torch.no_grad()
def predict_matched_wrist(probe, x, pooled):
    model,stats,arm=probe
    if (x.ndim!=2 or x.shape[1]!=811 or pooled.shape!=(len(x),2304)
            or not torch.isfinite(x).all() or not torch.isfinite(pooled).all()):
        raise ValueError('Own current811D and own frozen shared2304D required')
    raw=(x-stats['xmean'])/stats['xstd']
    z=(pooled-stats['zmean'])/stats['zstd'] if arm=='aligned' else torch.zeros_like(pooled)
    count=len(raw)
    # Keep the existing native-runtime safeguard for singleton CUDA Linear6.
    if raw.is_cuda and count==1:
        raw=raw.repeat(2,1);z=z.repeat(2,1)
    result=model(raw,z)[:count]*stats['ystd']+stats['ymean']
    if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite wrist command')
    return result


def order_server_features(value, device):
    pooled=torch.as_tensor(value,device=device,dtype=torch.float32)
    if pooled.shape!=(2,2304) or not torch.isfinite(pooled).all():
        raise ValueError('Two own-row server shared summaries required')
    return pooled[[1,0]]
