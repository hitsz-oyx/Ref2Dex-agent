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
from test_v6_training import packet


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


def test_subset_admission_does_not_relax_complete_recipe(packet):
    import json
    from ptv3_shared_wam_v6.recipe import validate_recipe
    from ptv3_shared_wam_v5.cache_contract import metadata_identity
    root, recipe, _, _ = packet
    parent = json.loads((root/'parent.json').read_text())
    stats = json.loads((root/'stats.json').read_text())
    parent.update(local_subset_scope=LOCAL_SCOPE, engineering_subset=True,
                  original_frozen_windows_recovered=False, pending_sequences=38349)
    recipe.update(local_subset_scope=LOCAL_SCOPE, initialization_checkpoint='/checkpoint',
                  index_identities={}, strata_identities={}, windows=dict(train=1, dev=1))
    stats.update(method='checkpoint-normalizer-buffers; inherited TRAIN scales; no moments refit',
                 initialization_checkpoint='/checkpoint')
    for split, seq_id in (('train', 0), ('dev', 1)):
        for field, data in (('indices', [[seq_id, 0, 3]]), ('strata', [0])):
            path = root/(split+'-'+field+'.npy')
            np.save(path, np.array(data, np.int32))
            parent[field][split] = str(path)
            recipe['index_identities' if field == 'indices' else 'strata_identities'][split] = index_identity(path)
    (root/'parent.json').write_text(json.dumps(parent))
    (root/'stats.json').write_text(json.dumps(stats))
    events = json.loads((root/'events.json').read_text())
    events['parent_manifest_identity'] = metadata_identity(root/'parent.json')
    (root/'events.json').write_text(json.dumps(events))
    for key in ('parent_manifest', 'event_manifest', 'stats'):
        recipe[key+'_identity'] = metadata_identity(recipe[key])
    (root/'manifest.json').write_text(json.dumps(recipe))
    validate_recipe(root/'manifest.json')
    del recipe['local_subset_scope']
    (root/'manifest.json').write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='complete nine-source'):
        validate_recipe(root/'manifest.json')
