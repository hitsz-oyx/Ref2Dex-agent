import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from trajectory_policy.decoder import TrajectoryDecoder, KNOTS
from consequence_evaluator.retargeter import wrist_rotation

URDF='third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'


def test_decoded_knots_retain_pose_in_rotated_query_frame_and_finger_bounds():
    decoder=TrajectoryDecoder(URDF,'cpu')
    current=np.zeros((1,18),np.float32);current[0,3]=3.13
    future=np.repeat(current[:,None],24,axis=1)
    future[0,:,0]=np.linspace(.01,.4,24)
    future[0,:,3]=np.linspace(3.14,3.4,24)
    future[0,:,[6,8,10,12,14,15]]=np.array([.8,.9,1.,1.1,.5,.2])[:,None]
    obj=np.eye(4,dtype=np.float32)[None];obj[0,:3,:3]=Rotation.from_euler('z',.6).as_matrix()
    c=decoder.encode(future,current,obj)
    decoded=decoder.decode(c,current,obj,.033333)
    actual=decoded['q'].numpy()[:,KNOTS]
    np.testing.assert_allclose(actual[:,:,:3],future[:,np.asarray(KNOTS)-1,:3],atol=2e-7)
    np.testing.assert_allclose(wrist_rotation(actual),wrist_rotation(future[:,np.asarray(KNOTS)-1]),atol=5e-7)
    np.testing.assert_allclose(actual[:,:, [6,8,10,12,14,15]],future[:,np.asarray(KNOTS)-1][:,:,[6,8,10,12,14,15]],atol=2e-7)
    assert decoded['hand'].shape==(1,24,11,3)
    assert np.max(abs(np.diff(decoded['q'].numpy()[0,:,3])))<.1


def test_decode_does_not_accept_future_reference_and_extreme_latent_is_bounded():
    decoder=TrajectoryDecoder(URDF,'cpu');current=np.zeros((1,18),np.float32)
    obj=np.eye(4,dtype=np.float32)[None];c=np.full((1,48),1000.,np.float32)
    q=decoder.decode(c,current,obj,.033333)['q'].numpy()
    assert np.isfinite(q).all()
    assert np.max(q[:,:, [6,8,10,12]])<=1.6+1e-7
    assert np.max(q[:,:,14])<=1.15+1e-7
    with pytest.raises(ValueError,match='finite'):decoder.decode(c*np.nan,current,obj,.033333)
