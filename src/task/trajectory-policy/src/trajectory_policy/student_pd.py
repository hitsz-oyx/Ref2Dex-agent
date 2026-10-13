"""Student-state action labels without promoting a failed future hand to a trajectory."""
from pathlib import Path

import torch
import torch.nn.functional as F

STUDENT_SCHEMA = 'ref2dex.student-pd-adapter.v1'
COMPETENT_SCHEMA = 'ref2dex.competent-pd-retargeter.v1'


def native_tau(prediction, order):
    """Right-hand displacement, reordered into the native links the competent R expects."""
    if prediction.ndim != 5 or prediction.shape[-3:] != (2, 11, 3):
        raise ValueError('WAM hand displacement B,24,2,11,3 required')
    if tuple(order.shape) != (11,):
        raise ValueError('native link order required')
    hand = prediction[:, :, 0].index_select(2, order)
    return hand.reshape(prediction.shape[0], prediction.shape[1], 33)


def trajectory_loss(prediction, target, valid, supervised, scale=0.05):
    """Rows with supervised False contribute nothing, even if a future hand is stored."""
    if prediction.shape != target.shape or valid.shape != prediction.shape[:-1]:
        raise ValueError('trajectory prediction, target and validity differ')
    if supervised.shape != (prediction.shape[0],) or supervised.dtype != torch.bool:
        raise ValueError('one trajectory-supervision bit per sample required')
    use = valid & supervised.view(-1, *([1] * (valid.ndim - 1)))
    error = F.huber_loss(prediction / scale, target / scale, reduction='none').mean(-1)
    return (error * use).sum() / use.sum().clamp_min(1).to(error.dtype)


def action_loss(predicted, target, has_pd):
    if predicted.shape != target.shape or predicted.shape[-1] != 12:
        raise ValueError('competent R emits one 12-D PD target')
    if has_pd.shape != (predicted.shape[0],) or has_pd.dtype != torch.bool:
        raise ValueError('one PD-supervision bit per sample required')
    error = (predicted - target).square().mean(-1)
    return (error * has_pd).sum() / has_pd.sum().clamp_min(1).to(error.dtype)


def competent_r_identity(path, digest):
    """Frozen successful-baseline R. One command per control, not a chunk."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('competent R file required')
    file_digest = digest(path)
    identity = dict(applied=True, kind='competent-pd-retargeter', schema=COMPETENT_SCHEMA,
                    checkpoint=str(path), execution_R_sha256=file_digest)
    return identity, {str(path): file_digest}


def validate_student_packet(packet, path, base, retargeter, digest):
    path, base, retargeter = (Path(path).resolve(), Path(base).resolve(), Path(retargeter).resolve())
    if packet.get('schema') != STUDENT_SCHEMA:
        raise ValueError('student PD packet required')
    if Path(packet.get('base_checkpoint', '')).resolve() != base:
        raise ValueError('student packet must continue the three-domain initialization')
    if Path(packet.get('retargeter', '')).resolve() != retargeter:
        raise ValueError('student packet must keep the competent R it was trained with')
    adapter = packet.get('adapter')
    if not isinstance(adapter, dict) or not {'alpha_p', 'alpha_g'} <= set(adapter) or not packet.get('trajectory'):
        raise ValueError('student packet is missing the reference adapter or trajectory')
    return dict(applied=True, schema=STUDENT_SCHEMA, checkpoint=str(path),
                checkpoint_sha256=digest(path), reference_offsets=[1, 16], effect_loss=False)


def student_control_identity(path, base, retargeter, digest):
    path = Path(path).resolve()
    packet = torch.load(path, map_location='cpu', weights_only=False)
    return packet, validate_student_packet(packet, path, base, retargeter, digest)
