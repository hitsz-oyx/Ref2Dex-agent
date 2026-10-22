"""Research-critical event alignment and split support controls."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

AUDIT=Path(__file__).resolve().parents[1]/'tools/audit'
sys.path.insert(0,str(AUDIT))
from audit_event_aligned_variance import warp_trajectory,pooled
from audit_goal_chain import phase_ids,select_spaced


def test_event_warp_aligns_different_contact_times_and_preserves_endpoint():
    def curve(event):
        t=np.arange(1,25,dtype=float)
        return np.where(t<=event,t/event,1+(t-event)/(24-event))[:,None]
    first=warp_trajectory(curve(6),6,12)
    second=warp_trajectory(curve(9),9,12)
    np.testing.assert_allclose(first,second,atol=1e-7)
    assert first[11,0]==1 and first[-1,0]==2
    np.testing.assert_array_equal(warp_trajectory(curve(6),30,12),curve(6))


def test_phase_onset_hold_release_are_distinct():
    state=np.zeros((30,1),int);state[4:15]=1
    tr=np.array([[4,0,1],[15,0,2]])
    phase=phase_ids(state,tr,0,3)
    assert phase[:6]==[2]*6
    assert phase[6:11]==[3]*5
    assert phase[11:]==[4]*13


def test_spaced_samples_are_complete_support_disjoint():
    index=np.array([[1,0,t] for t in range(3,204)])
    picked=select_spaced(list(range(len(index))),index,np.random.default_rng(181),16)
    ticks=sorted(index[picked,2]);assert len(picked)>4
    assert min(np.diff(ticks))>=28


def test_pooled_variance_no_singleton_zero_evidence():
    hand=np.zeros((3,24,2,21,3));scene=np.zeros((3,24,32,3))
    hand[1,...,0]=2;scene[1,...,0]=4
    z=dict(hand=hand,scene=scene,present=np.ones((3,2),bool))
    sums,df,n=pooled([[0,1]],z)
    np.testing.assert_allclose(sums,[2e6,2e6,8e6,8e6]);assert (df,n)==(1,2)
