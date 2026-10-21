import copy
import numpy as np
import pytest
import torch
from torch import nn

from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.local_subset import (
    LOCAL_SCOPE, index_identity, initialize_weights,
    validate_local_execution, validate_local_parent,
)
from ptv3_shared_wam_v6.contracts import SCHEMA


def local_packet(tmp_path):
    parent = dict(local_subset_scope=LOCAL_SCOPE, status='FROZEN', training_allowed=True,
                  engineering_subset=True, original_frozen_windows_recovered=False,
                  sequences=[dict(key='train', source_id=1), dict(key='dev', source_id=1)],
                  source_ids=[1], indices={}, strata={})
    recipe = dict(local_subset_scope=LOCAL_SCOPE, initialization_checkpoint='/checkpoint',
                  index_identities={}, strata_identities={})
    for split, seq_id in (('train', 0), ('dev', 1)):
        for field, data in (('indices', [[seq_id, 0, 3]]), ('strata', [0])):
            path = tmp_path/(split+'-'+field+'.npy')
            np.save(path, np.array(data, np.int32))
            parent[field][split] = str(path)
            recipe['index_identities' if field == 'indices' else 'strata_identities'][split] = index_identity(path)
    stats = dict(method='checkpoint-normalizer-buffers; inherited TRAIN scales; no moments refit',
                 initialization_checkpoint='/checkpoint')
    return parent, recipe, stats


def test_sequence_leakage_and_index_mutation_are_rejected(tmp_path):
    parent, recipe, stats = local_packet(tmp_path)
    validate_local_parent(parent, recipe, stats)
    np.save(parent['indices']['dev'], np.array([[0, 0, 3]], np.int32))
    with pytest.raises(ValueError, match='leakage'):
        validate_local_parent(parent, recipe, stats)
    np.save(parent['indices']['dev'], np.array([[1, 0, 4]], np.int32))
    with pytest.raises(ValueError, match='index changed'):
        validate_local_parent(parent, recipe, stats)


def test_subset_requires_explicit_scope_and_bounded_execution(tmp_path):
    parent, recipe, stats = local_packet(tmp_path)
    del parent['local_subset_scope']
    with pytest.raises(ValueError, match='admission'):
        validate_local_parent(parent, recipe, stats)
    config = dict(local_subset_scope=LOCAL_SCOPE, updates=2000, max_seconds=14400,
                  artifact_budget_bytes=8*2**30)
    validate_local_execution(config, recipe, 1, 8, 64)
    for world, last, batch in ((3, 8, 64), (1, 2001, 64), (1, 8, 128)):
        with pytest.raises(ValueError, match='single-GPU'):
            validate_local_execution(config, recipe, world, last, batch)


def test_weights_only_initialization_preserves_scale_and_rejects_changes(tmp_path):
    model = nn.Module()
    model.normalizer = nn.Module()
    model.normalizer.register_buffer('hand_mean', torch.zeros(24, 3))
    model.head = nn.Linear(3, 3)
    state = copy.deepcopy(model.state_dict())
    state['head.weight'].fill_(2.)
    checkpoint = tmp_path/'step023758.pt'
    recipe = dict(local_subset_scope=LOCAL_SCOPE, initialization_checkpoint=str(checkpoint.resolve()))
    saved = dict(schema=SCHEMA, step=23758, data_identity='original', model=state)
    binding = initialize_weights(model, saved, checkpoint, recipe)
    assert torch.equal(model.head.weight, state['head.weight'])
    assert binding['optimizer_restored'] is False and binding['training_step_reset'] is True
    model.normalizer.hand_mean.fill_(1.)
    with pytest.raises(ValueError, match='normalization'):
        initialize_weights(model, saved, checkpoint, recipe)
