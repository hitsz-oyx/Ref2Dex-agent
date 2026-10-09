import numpy as np
from scipy.spatial.transform import Rotation
from consequence_evaluator.object_relative_servo import transport_wrist
from consequence_evaluator.retargeter import wrist_rotation


def test_transport_preserves_object_local_wrist_pose_and_fingers():
    q=np.array([.1,.2,.9,.3,1.8,-.2]+[.5]*12,dtype='float32')
    source=np.eye(4);source[:3,:3]=Rotation.from_rotvec([.1,-.2,.05]).as_matrix();source[:3,3]=[.2,.3,.6]
    live=np.eye(4);live[:3,:3]=Rotation.from_rotvec([-.2,.1,.3]).as_matrix();live[:3,3]=[.4,.1,.8]
    out=transport_wrist(q,source,live,q)
    np.testing.assert_allclose(live[:3,:3].T@(out[:3]-live[:3,3]),source[:3,:3].T@(q[:3]-source[:3,3]),atol=1e-7)
    np.testing.assert_allclose(live[:3,:3].T@wrist_rotation(out),source[:3,:3].T@wrist_rotation(q),atol=1e-6)
    np.testing.assert_array_equal(out[6:],q[6:])


def test_identity_transport_and_current_anchor_preserve_next_step_motion():
    source=np.eye(4);q=np.zeros(18,dtype='float32');q[:3]=[.2,.3,.4]
    np.testing.assert_allclose(transport_wrist(q,source,source,q),q,atol=1e-7)
    nextq=q.copy();nextq[2]+=.01
    out=transport_wrist(nextq,source,source,q)
    assert abs(out[2]-q[2]-.01)<1e-7
    future=source.copy();future[2,3]+=.01
    cancelled=transport_wrist(nextq,future,source,q)
    np.testing.assert_allclose(cancelled[:3],q[:3],atol=1e-7)
