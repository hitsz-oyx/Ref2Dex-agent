"""Protect the PD-only collection split from duplicate-prefix label leakage."""
import importlib.util
from pathlib import Path

import numpy as np


def test_query_partition_excludes_duplicate_prefix_and_unsupported_roles():
    path=Path(__file__).resolve().parents[1]/'tools/run/fit_wam_visited_pd.py'
    spec=importlib.util.spec_from_file_location('wam_visited_pd_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ticks=np.repeat(np.arange(542),4)
    roles=np.tile(np.arange(4),542)
    train,val=module.query_partition(ticks,roles)
    assert not np.any(train & val)
    selected=train | val
    assert not selected[roles<2].any()
    assert not selected[(roles==3)&(ticks<16)].any()
    assert not selected[(roles==2)&(ticks>=64)].any()
    assert not selected[ticks>=519].any()
    assert np.all(train[(roles==2)&(ticks<48)])
    assert np.all(val[(roles==2)&(ticks>=48)&(ticks<64)])
    assert train.sum()==503 and val.sum()==64
    train,val=module.query_partition(ticks,roles,approach_only=True)
    assert not np.any(train&val)
    assert not (train|val)[(roles==2)&(ticks>=48)].any()
    assert not (train|val)[(roles==3)&((ticks<16)|(ticks>=128))].any()
    assert np.all(val[(roles==2)&(ticks>=32)&(ticks<48)])
    assert np.all(val[(roles==3)&(ticks>=64)&(ticks<80)])
    assert train.sum()==128 and val.sum()==32
