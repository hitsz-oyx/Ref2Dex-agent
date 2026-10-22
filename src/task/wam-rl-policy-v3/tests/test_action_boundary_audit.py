"""Negative controls detect omitted clip counts and conflated endpoints/limits."""
import numpy as np
import pytest
from wam_rl_policy_v3.action_boundary_audit import audit_action_boundaries


def inputs():
    raw=np.zeros((3,12)); raw[0,0]=1.02; raw[1,6]=1.; raw[2,2]=1.0000005
    action=np.clip(raw,-1,1)
    q=np.zeros((4,18)); q[1,6]=-.05; q[3,6]=-.11
    limits=dict(has_limits=np.ones(18,bool),lower=np.zeros(18),upper=np.ones(18)*3)
    return raw,action,q,limits


def test_substantive_clip_and_legal_endpoints_have_distinct_denominators():
    r=audit_action_boundaries(*inputs(),reported_clipped=1)
    assert r['clipped_controls']==1 and r['strict_raw_outside_controls']==2
    assert r['boundary_controls']==3 and r['boundary_controls_cartesian_or_wrist']==2
    assert r['nominal_joint_outside_states']==2 and r['joint_abort_states']==1
    assert r['first_joint_abort_tick']==3 and r['joint_abort_indices']==[6]
    assert r['source_clipped_controls'] is None


def test_injected_omission_of_clipped_count_is_rejected():
    with pytest.raises(ValueError,match='reported clipped_controls'):
        audit_action_boundaries(*inputs(),reported_clipped=0)


def test_injected_unclipped_action_log_is_rejected():
    raw,action,q,limits=inputs();action[0,0]=raw[0,0]
    with pytest.raises(ValueError,match='not clip'):
        audit_action_boundaries(raw,action,q,limits)


def test_source_rejects_substantive_overflow_but_accepts_roundoff_and_endpoints():
    raw,action,q,limits=inputs()
    with pytest.raises(ValueError,match='source chain'):
        audit_action_boundaries(raw,action,q,limits,source_chain=True)
    raw[0,0]=0; action=np.clip(raw,-1,1)
    r=audit_action_boundaries(raw,action,q,limits,source_chain=True,reported_source_rejections=0)
    assert r['clipped_controls']==0 and r['boundary_controls']==2
    assert r['source_clipped_controls']==0


def test_joint_limit_mask_and_margin_cannot_be_conflated_with_command_boundary():
    raw=np.zeros((1,12));q=np.zeros((2,18));q[1,4]=-100; q[1,6]=-.099
    limits=dict(has_limits=np.ones(18,bool),lower=np.zeros(18),upper=np.ones(18)*3)
    limits['has_limits'][4]=False
    assert audit_action_boundaries(raw,raw,q,limits)['joint_abort_states']==0
    q[1,6]=-.101
    assert audit_action_boundaries(raw,raw,q,limits)['joint_abort_states']==1
