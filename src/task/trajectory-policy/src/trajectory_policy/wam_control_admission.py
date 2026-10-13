"""Bind a robot adapter overlay to its unchanged C base and qualified R."""
import json
from pathlib import Path

from .wam_retargeter_admission import ROOT, sha

SCHEMA = 'ref2dex.robot-control-context.v1'


def control_admission(checkpoint, effect_identity, execution_R_identity=None):
    if checkpoint is None:
        return dict(applied=False), None
    import torch
    path = Path(checkpoint).resolve()
    if ROOT/'outputs/goal-point-wam' not in path.parents:
        raise ValueError('owned robot adapter required')
    packet = torch.load(path, map_location='cpu')
    result = json.loads((path.parent/'result.json').read_text())
    frozen = packet.get('contracts', {})
    if packet.get('manifest', {}).get('alignment'):
        expected = packet['manifest'].get('training_R_sha256')
        if not expected or (execution_R_identity or {}).get('execution_R_sha256') != expected:
            raise ValueError('aligned actor requires its exact frozen training R')
        if result.get('offline_gate') is not True:
            raise ValueError('aligned actor offline admission gate failed')
    if (packet.get('schema') != SCHEMA or result.get('status') != 'COMPLETED'
            or packet.get('manifest', {}).get('training_allowed') is not True
            or Path(result.get('final_checkpoint', '')).resolve() != path
            or not effect_identity.get('applied')
            or packet.get('manifest', {}).get('C_identity') != effect_identity
            or frozen.get('frozen_max_difference') != 0. or frozen.get('R_max_difference') != 0.
            or packet.get('stage') not in (1,2)
            or (packet['stage']==1 and frozen.get('decoder_difference') != 0.)):
        raise ValueError('completed robot adapter / exact C base / frozen R contract differs')
    for name in ('adapter','trajectory'):
        if not packet.get(name) or any(not torch.isfinite(v).all() for v in packet[name].values()):
            raise ValueError('finite robot adapter and decoder required')
    identity = dict(applied=True,checkpoint=str(path),stage=packet['stage'],step=packet['step'],
        checkpoint_sha256=sha(path),schema=SCHEMA,
        scope='robot-only measured q/dq/previous action and planned object reference conditioning')
    return identity, packet
