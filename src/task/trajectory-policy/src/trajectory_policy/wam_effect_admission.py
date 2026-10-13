"""Admit C's decoder-only continuation and the exact qualified native Chunk-R."""
import json
from pathlib import Path

from .wam_retargeter_admission import ROOT, sha


def chunk_admission(directory):
    directory = Path(directory).resolve()
    if ROOT/'outputs/trajectory-policy' not in directory.parents:
        raise ValueError('owned chunk directory required')
    checkpoint = directory/'chunk.pt'
    manifest = json.loads((directory/'manifest.json').read_text())
    result = json.loads((directory/'result.json').read_text())
    receipt = directory/'qualification.json'
    qualification = (Path(json.loads(receipt.read_text())['directory']).resolve() if receipt.exists()
                     else ROOT/'outputs/trajectory-policy/wam-v2-control-GT-20261012-r1')
    if ROOT/'outputs/trajectory-policy' not in qualification.parents:
        raise ValueError('owned qualification required')
    qm = json.loads((qualification/'manifest.json').read_text())
    qr = json.loads((qualification/'result.json').read_text())
    digest = sha(checkpoint)
    if (manifest.get('status') != 'COMPLETED' or result.get('status') != 'COMPLETED'
            or digest != result.get('checkpoint_sha256')
            or qm.get('status') != 'COMPLETED' or qr.get('GT_execution_gate') is not True
            or qm.get('checkpoint') != str(checkpoint) or qm.get('checkpoint_sha256') != digest):
        raise ValueError('exact completed and GT-qualified Chunk-R required')
    paths = [checkpoint, directory/'manifest.json', directory/'result.json',
             qualification/'manifest.json', qualification/'result.json']
    identity = dict(R_override=True, kind='qualified_native_chunk8_execute_first',
                    execution_R_sha256=digest, checkpoint=str(checkpoint),
                    scope='prior GT qualification; current rollout calibrates again')
    if receipt.exists():
        paths.append(receipt)
        identity.update(qualification_directory=str(qualification), repair_parent=manifest.get('parent'))
    return checkpoint, identity, {str(path): sha(path) for path in paths}


def effect_admission(checkpoint, head_directory, chunk_directory, allow_repaired_chunk=False):
    if checkpoint is None:
        return dict(applied=False), None
    import torch
    checkpoint = Path(checkpoint).resolve()
    if ROOT/'outputs/goal-point-wam' not in checkpoint.parents or chunk_directory is None:
        raise ValueError('owned C checkpoint and exact Chunk-R required')
    packet = torch.load(checkpoint, map_location='cpu')
    result = json.loads((checkpoint.parent/'result.json').read_text())
    manifest = packet.get('manifest', {})
    recipe = packet.get('recipe', {})
    training_chunk = Path(manifest.get('qualified_chunk', '')).resolve()
    execution_chunk = Path(chunk_directory).resolve()/'chunk.pt'
    chunk_matches = training_chunk == execution_chunk
    if not chunk_matches and allow_repaired_chunk:
        chunk_admission(chunk_directory)
        # Every repaired link must retain an exact parent and passing offline gate.
        node = Path(chunk_directory).resolve()
        for _ in range(3):
            fm = json.loads((node/'manifest.json').read_text())
            fr = json.loads((node/'result.json').read_text())
            parent = Path(fm.get('parent', '')).resolve()
            if (fm.get('experiment_id') not in ('P-20261013-goal-point-wam-ref31-own-R',
                    'P-20261013-goal-point-wam-visited-realign')
                    or ROOT/'outputs/trajectory-policy' not in parent.parents
                    or not parent.is_file() or fm.get('parent_sha256') != sha(parent)
                    or fr.get('offline_gate') is not True):
                break
            if parent == training_chunk:
                chunk_matches = True
                break
            node = parent.parent
    if (packet.get('schema') != 'ref2dex.effect-policy-patch.v1'
            or result.get('status') != 'COMPLETED' or packet.get('step') != result.get('step')
            or packet.get('C0_pass') is not True or recipe.get('train_modules') != ['trajectory']
            or Path(manifest.get('warm_policy', '')).resolve() != Path(head_directory).resolve()
            or not chunk_matches
            or not packet.get('curve') or any(row.get('model_frozen_max_difference') != 0.
                or row.get('R_max_difference') != 0. for row in packet['curve'])):
        raise ValueError('completed decoder-only C continuation identity differs')
    values = packet['trajectory']
    if not values or any(not torch.isfinite(value).all() for value in values.values()):
        raise ValueError('finite C decoder patch required')
    identity = dict(applied=True, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                    global_step=packet['step'], base_checkpoint=manifest['base_checkpoint'],
                    warm_policy=manifest['warm_policy'], qualified_chunk=manifest['qualified_chunk'],
                    scope='decoder overlay only; warm scene/goal/hand query retained')
    return identity, values
