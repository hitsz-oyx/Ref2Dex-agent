import copy
from types import SimpleNamespace
import numpy as np
import pytest
from ptv3_shared_wam_v2.rollout_admission import DEFAULT_POLICY,admissible_ticks,restrict_rollout_admission


def data(length=160):
    obj=np.broadcast_to(np.eye(4,dtype=np.float32),(length,4,4)).copy()
    return dict(obj=obj,reference_obj=obj.copy())


def test_failure_preceding50_and_entire_h4_future24_support():
    d=data();d['obj'][100,0,3]=.1001
    ticks=np.arange(136);keep,report=admissible_ticks(d,ticks,DEFAULT_POLICY)
    assert ticks[keep].tolist()==list(range(26))+list(range(104,136))
    assert report['unsafe_frames']==51
    # Query25 ends49; query26 reaches unsafe50. Query103 sees bad100 in H4.
    assert keep[25] and not keep[26] and not keep[103] and keep[104]


def test_exact_threshold_and_start_end_boundaries():
    d=data();d['obj'][0,0,3]=.1
    # float32 .1 and configured .1 compare in float32, preserving old gate.
    keep,_=admissible_ticks(d,np.arange(136),DEFAULT_POLICY);assert keep.all()
    d['obj'][0,0,3]=.2
    keep,_=admissible_ticks(d,np.arange(136),DEFAULT_POLICY);assert np.flatnonzero(keep)[0]==4
    with pytest.raises(ValueError,match='incomplete'):admissible_ticks(d,[136],DEFAULT_POLICY)


def test_rebuilds_training_views_and_preserves_raw_records():
    d=data();d['obj'][100,0,3]=1
    records=[dict(sequence='a',admitted_ticks=list(range(136)))]
    original=copy.deepcopy(records)
    ds=SimpleNamespace(records=records,split='train',keys=[(i,0) for i in range(136)],
        ticks_by_sequence=[np.arange(136)],sequence=lambda _: (d,{}))
    report=restrict_rollout_admission(ds,DEFAULT_POLICY)
    assert records==original and report['retained_windows']==58
    assert ds.keys==[(i,0) for i in list(range(26))+list(range(104,136))]
    assert len(ds.pools[0])==58 and len(ds.phase_pools[0][0])==16


def test_nonfinite_overlapping_failures_and_short_sequence():
    d=data();d['obj'][100,0,3]=float('nan');d['obj'][110,0,3]=1
    keep,report=admissible_ticks(d,np.arange(136),DEFAULT_POLICY)
    assert report['unsafe_frames']==61
    assert np.flatnonzero(keep).tolist()==list(range(26))+list(range(114,136))
    d=data(24);keep,_=admissible_ticks(d,[],DEFAULT_POLICY);assert keep.size==0
