import json

import numpy as np
import pytest

from shadow_finetune.data import ShadowWindows
from shadow_finetune.scale_controls import STARTUP_SCHEMA,load_controls,window_key
from test_scale_controls import caches


def test_zero_mixture_preserves_original_draws_and_rng(tmp_path):
    _,full=caches(tmp_path);d=ShadowWindows(full,'train')
    a=np.random.default_rng(214);b=np.random.default_rng(214)
    assert d.draw(a,100)==d.draw_startup(b,100,0.)
    assert a.bit_generator.state==b.bit_generator.state


def test_startup_mixture_only_uses_admitted_early_ticks_and_keeps_later_support(tmp_path):
    _,full=caches(tmp_path);p=full/'manifest.json';m=json.loads(p.read_text())
    # Different temporal supports: no manufactured tick0 for the second record.
    m['records'][0]['ticks']=[2,7,18];m['records'][2]['ticks']=[0,9,20]
    p.write_text(json.dumps(m));d=ShadowWindows(full,'train');rng=np.random.default_rng(219)
    early=d.draw_startup(rng,200,1.)
    assert {d.keys[i] for i in early}=={(0,2),(0,7),(1,0)}
    mixed=d.draw_startup(rng,4000,.5)
    assert any(d.keys[i][1]>=8 for i in mixed)
    assert .65<sum(d.keys[i][1]<8 for i in mixed)/len(mixed)<.85
    with pytest.raises(ValueError,match='TRAIN-only'):
        ShadowWindows(full,'dev').draw_startup(rng,10,.5)
    with pytest.raises(ValueError,match='fraction'):d.draw_startup(rng,1,float('nan'))


def test_controls_reject_undeclared_sampler_change(tmp_path):
    small,full=caches(tmp_path);dev=ShadowWindows(full,'dev');parent=tmp_path/'parent.pt';parent.write_bytes(b'p')
    norm=ShadowWindows(small,'train').normalization()
    c=dict(schema=STARTUP_SCHEMA,status='PREPARED',startup_fraction=.5,
        settings=dict(parent=str(parent),seed=214,updates=1000,batch_size=64),
        training_caches=dict(startup=str(full)),validation_cache=str(full),validation_ids=[0],
        validation_keys=[window_key(dev,0)],input_stamps=[],normalization={k:v.tolist() for k,v in norm.items()})
    p=tmp_path/'controls.json';p.write_text(json.dumps(c))
    load_controls(p,full,parent,214,1000,64,.5)
    with pytest.raises(ValueError,match='sampling changed'):load_controls(p,full,parent,214,1000,64,0.)
