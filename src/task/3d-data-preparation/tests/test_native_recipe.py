"""Recipes keep source probabilities, frozen membership, and moments consistent."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pose_mesh_data.native_recipe import (RECIPE_SCHEMA,NATIVE_STATS_SCHEMA,NativeSourceSampler,
    source_probabilities,validate_stats_binding)


def test_capped_temperature_sampling_preserves_total_probability():
    counts={'4':264439,'5':30568,'6':71070,'8':351,'9':5691,'10':55547}
    p=source_probabilities(counts,.5,{'8':.01})
    assert sum(p.values())==pytest.approx(1) and p['8']==.01
    assert p['4']/p['5']==pytest.approx(np.sqrt(counts['4']/counts['5']))
    assert p['4']<.5  # Large simulator corpus does not retain its natural 62% share.
    with pytest.raises(ValueError,match='cannot allocate'):
        source_probabilities({'4':10,'5':10},.5,{'4':.3,'5':.3})
    with pytest.raises(ValueError,match='known sources'):
        source_probabilities(counts,.5,{'7':.01})


def fixture():
    rows=[dict(source_id=sid,split='train') for sid in (4,5,6,8,9,10)]
    data=SimpleNamespace(root=Path('/frozen'),meta={'sequences':rows},
        index=[(r,t) for r in range(6) for t in (3,11)],verified_clock_only=True,_index_stamp=(100,200))
    recipe=dict(schema=RECIPE_SCHEMA,training_allowed=False,permitted_use='engineering_only',
        sampling='source_probability_then_uniform_frozen_window.v1',data_manifest='/frozen/manifest.json',
        window_index_size_bytes=100,window_index_mtime_ns=200,
        source_counts={str(s):2 for s in (4,5,6,8,9,10)},
        source_probabilities={'4':.4,'5':.14,'6':.21,'8':.01,'9':.06,'10':.18})
    return data,recipe


def test_draw_is_reproducible_matches_probabilities_and_rejects_dev_membership():
    data,recipe=fixture();sampler=NativeSourceSampler(data,recipe)
    a=sampler.draw(30000,18,step=3)
    np.testing.assert_array_equal(a,sampler.draw(30000,18,step=3))
    assert not np.array_equal(a,sampler.draw(30000,18,step=4))
    sid=np.asarray([data.meta['sequences'][data.index[i][0]]['source_id'] for i in a])
    for source,p in recipe['source_probabilities'].items():
        assert abs((sid==int(source)).mean()-p)<.012
    assert set(a)==set(range(12))
    data.meta['sequences'][0]['split']='dev'
    with pytest.raises(ValueError,match='DEV'):
        NativeSourceSampler(data,recipe)


def test_recipe_and_statistics_cannot_change_weights_or_membership_independently():
    _,recipe=fixture()
    stats={**recipe,'schema':NATIVE_STATS_SCHEMA,'split':'train'}
    validate_stats_binding(recipe,stats)
    changed=deepcopy(stats);changed['source_probabilities']['4']+=.01
    with pytest.raises(ValueError,match='binding mismatch'):
        validate_stats_binding(recipe,changed)
    changed={**stats,'window_index_mtime_ns':201}
    with pytest.raises(ValueError,match='binding mismatch'):
        validate_stats_binding(recipe,changed)
