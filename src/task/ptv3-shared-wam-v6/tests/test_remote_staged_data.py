import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('remote_staged_data',
    Path(__file__).resolve().parents[1]/'tools/run/remote_staged_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_half_covers_every_source_and_inherited_split_without_order_bias():
    rows = [dict(key='%s-%s-%s'%(s,split,i),source_id=s,split=split,
        source_sequence='%s-%s'%(split,i)) for s in (0,1,8) for split in ('train','dev') for i in range(7)]
    chosen=module.stratified_keys(rows)
    assert chosen == module.stratified_keys(list(reversed(rows)))
    for source in (0,1,8):
        for split in ('train','dev'):
            assert sum(r['key'] in chosen for r in rows if r['source_id']==source and r['split']==split)==4
    assert len(chosen | ({r['key'] for r in rows}-chosen)) == len(rows)


def test_duplicate_mother_across_splits_is_rejected():
    with pytest.raises(ValueError,match='mother'):
        module.stratified_keys([dict(key=split,source_id=1,source_sequence='mother',split=split)
            for split in ('train','dev')])
