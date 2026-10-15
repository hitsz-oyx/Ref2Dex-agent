"""Audit must detect missing contacts, wrong labels and frame misalignment."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('current_rollout_audit',
    Path(__file__).resolve().parents[1]/'tools/audit/audit_current_rollout.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    t,b=4,25
    c=dict(body_names=list(range(b)),reference_dim=428,contact_body_ids=[9,12,15,18,23],
           measured_hand_body_ids=[7,20,24,8,10,11,13,17,19,14,16])
    data={k:np.zeros(shape,np.float32) for k,shape in dict(q=(t,18),dq=(t,18),hand=(t,11,3),
        obj=(t,4,4),velocity=(t,6),reference_obj=(t,4,4),native_current_obs=(t,721),
        body_state=(t,b,13),body_contact_forces=(t,b,3),object_contact_force=(t,3),
        reference_current=(t,428),action=(t-1,18),applied=(t-1,18),pd_targets=(t-1,18)).items()}
    data['frame_tick']=np.arange(t)
    data['body_contact_forces'][2,9,0]=.2
    data['native_current_obs'][2,252]=1
    data['reference_current'][:,171]=1
    data['native_current_obs'][:,401]=1
    data['native_current_obs'][2,401]=0
    data['pd_targets'][:,6:]=.5
    for distal,parent,ratio in [(7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)]:
        data['pd_targets'][:,distal]=data['pd_targets'][:,parent]*ratio
    return data,dict(frames=t),c,dict(offset=[0]*18,scale=[1]*18)


def test_audit_accepts_known_same_tick_contact_pd_and_coupling():
    report=audit.audit_sequence(*fixture())
    assert report['frames_with_measured_contact']==1
    assert report['max_pd_error']==0


@pytest.mark.parametrize('kind',['contact','label','clock'])
def test_audit_rejects_semantic_corruption(kind):
    data,r,c,controller=fixture()
    if kind=='contact': data['native_current_obs'][2,252]=0
    if kind=='label': data['pd_targets'][1,0]=.01
    if kind=='clock': data['frame_tick'][1]=0
    with pytest.raises((AssertionError,ValueError)):
        audit.audit_sequence(data,r,c,controller)
