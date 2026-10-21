"""Explicit local subset admission and weights-only initialization.

This scope never qualifies as the complete nine-source continuation recipe.
"""
import hashlib
from pathlib import Path
import torch

LOCAL_SCOPE = 'v6.local-prepared-subset.v1'
EXPANDED_SCOPE = 'v6.local-expanded-ddp.v1'
LOCAL_SCOPES = (LOCAL_SCOPE, EXPANDED_SCOPE)


def index_identity(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_local_parent(parent, recipe, normalizer):
    if (parent.get('local_subset_scope') not in LOCAL_SCOPES
            or recipe.get('local_subset_scope') != parent.get('local_subset_scope')
            or parent.get('status') != 'FROZEN'
            or parent.get('training_allowed') is not True
            or parent.get('engineering_subset') is not True
            or parent.get('original_frozen_windows_recovered') is not False):
        raise ValueError('explicit independent local subset admission required')
    if not parent.get('sequences') or len(parent['sequences']) > 1177:
        raise ValueError('local subset exceeds audited preparation')
    if normalizer.get('method') != 'checkpoint-normalizer-buffers; inherited TRAIN scales; no moments refit':
        raise ValueError('local warm start must retain checkpoint TRAIN scales')
    if normalizer.get('initialization_checkpoint') != recipe.get('initialization_checkpoint'):
        raise ValueError('normalizer initialization provenance changed')
    from ptv3_shared_wam_v5.storage import load_array
    train, dev = [load_array(parent['indices'][s], mmap_mode='r') for s in ('train', 'dev')]
    for array in (train, dev):
        if (array.ndim != 2 or array.shape[1] != 3 or not len(array)
                or array.dtype.kind not in 'iu' or (array < 0).any()
                or array[:, 0].max() >= len(parent['sequences'])):
            raise ValueError('invalid local window indices')
    if set(train[:, 0]) & set(dev[:, 0]):
        raise ValueError('local TRAIN/DEV sequence leakage')
    keys = [row['key'] for row in parent['sequences']]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate local sequence identity')
    if sorted(set(row['source_id'] for row in parent['sequences'])) != parent['source_ids']:
        raise ValueError('local source identities differ')
    for split, array in (('train', train), ('dev', dev)):
        if index_identity(parent['indices'][split]) != recipe['index_identities'][split]:
            raise ValueError('local frozen index changed')
        if index_identity(parent['strata'][split]) != recipe['strata_identities'][split]:
            raise ValueError('local frozen strata changed')
        if len(load_array(parent['strata'][split], mmap_mode='r')) != len(array):
            raise ValueError('local strata length mismatch')


def validate_local_execution(config, recipe, world, last, batch):
    scope = recipe.get('local_subset_scope')
    if scope not in LOCAL_SCOPES:
        if config.get('local_subset_scope'):
            raise ValueError('local config cannot train a full recipe')
        return
    if scope == EXPANDED_SCOPE:
        if (config.get('local_subset_scope') != scope or world not in (1, 3)
                or not 1 <= last <= config['updates'] <= 10000
                or not 0 < config['max_seconds'] <= 14400
                or not 1 <= batch <= 128
                or config['artifact_budget_bytes'] > 8*2**30):
            raise ValueError('expanded subset requires bounded world1 capacity probe or world3 training')
        return
    if (config.get('local_subset_scope') != LOCAL_SCOPE or world != 1
            or not 1 <= last <= config['updates'] <= 2000
            or not 0 < config['max_seconds'] <= 14400
            or not 1 <= batch <= 64
            or config['artifact_budget_bytes'] > 8*2**30):
        raise ValueError('local subset requires bounded single-GPU configuration')


def initialize_weights(model, saved, checkpoint, recipe):
    from .contracts import SCHEMA
    if (recipe.get('local_subset_scope') != LOCAL_SCOPE
            or str(checkpoint.resolve()) != recipe['initialization_checkpoint']
            or saved.get('schema') != SCHEMA or saved.get('step') != 23758):
        raise ValueError('explicit admitted step23758 weights-only source required')
    state = saved['model']
    for name, value in model.normalizer.state_dict().items():
        original = state['normalizer.'+name]
        if not torch.equal(value.cpu(), original.cpu()):
            raise ValueError('warm start changed pretrained normalization')
    if any(not torch.isfinite(value).all() for value in state.values() if value.is_floating_point()):
        raise ValueError('nonfinite initialization weights')
    model.load_state_dict(state, strict=True)
    return dict(kind='weights-only-independent-local-subset', checkpoint=str(checkpoint.resolve()),
                source_step=saved['step'], source_data_identity=saved['data_identity'],
                optimizer_restored=False, rng_restored=False, training_step_reset=True,
                normalization='exact inherited checkpoint buffers')
