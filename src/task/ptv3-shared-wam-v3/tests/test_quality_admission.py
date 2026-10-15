from pathlib import Path
import json,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ptv3_shared_wam_v3.quality_admission import excluded_indices


def test_quality_index_is_bound_to_base_and_immutable_stamp(tmp_path):
    p=tmp_path/'index.npz';np.savez(p,train=np.array([1,4]),dev=np.array([2]));s=p.stat()
    r=dict(native_index_stamp=[1,2],main3_identity='fixed',quality_admission=dict(
        schema='ref2dex.strict-motion-window-admission.v1',index_path=str(p),index_stamp=[s.st_size,s.st_mtime_ns],
        base_native_index_stamp=[1,2],base_main3_identity='fixed',excluded_windows=dict(train=2,dev=1)))
    assert excluded_indices(r,'train',5).tolist()==[1,4]
    with pytest.raises(ValueError,match='in-range'):excluded_indices(r,'train',4)
    r['native_index_stamp']=[1,3]
    with pytest.raises(ValueError,match='identity'):excluded_indices(r,'train',5)


def test_unsorted_duplicates_cannot_be_used_as_a_quality_filter(tmp_path):
    p=tmp_path/'index.npz';np.savez(p,train=np.array([1,1]));s=p.stat()
    r=dict(native_index_stamp=[1,2],main3_identity='fixed',quality_admission=dict(
        schema='ref2dex.strict-motion-window-admission.v1',index_path=str(p),index_stamp=[s.st_size,s.st_mtime_ns],
        base_native_index_stamp=[1,2],base_main3_identity='fixed',excluded_windows=dict(train=2)))
    with pytest.raises(ValueError,match='sorted unique'):excluded_indices(r,'train',5)
