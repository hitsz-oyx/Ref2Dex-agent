"""Physical known answers for raw LOG direction, Euler axes, native IDs and validity."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
from convert_hoi4d_native import parse_camera_log,object_label,pair_native


def test_raw_redwood_log_remains_camera_to_world():
    text='7 7 8\n0 -1 0 1\n1 0 0 2\n0 0 1 3\n0 0 0 1\n'
    cameras=parse_camera_log(text)
    np.testing.assert_array_equal(cameras[7]@np.array([.1,.2,.3,1]),[.8,2.1,3.3,1])


def test_native_bbox_is_not_a_rigid_scale_matrix():
    native={'dataList':[dict(label='Mug',center=dict(x=.1,y=.2,z=.3),rotation=dict(x=0,y=0,z=np.pi/2),
        dimensions=dict(length=2,width=3,height=4))]}
    pose,dims,label=object_label(native,'Mug')
    np.testing.assert_allclose(pose@np.array([1,0,0,1]),[.1,1.2,.3,1],atol=1e-7)
    np.testing.assert_array_equal(dims,[2,3,4]);assert label=='Mug'
    np.testing.assert_allclose(np.linalg.det(pose[:3,:3]),1)


def test_filename_id_drives_pairing_and_camera_transform_cancels_in_relative_frame():
    camera=np.array([[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1.]])
    obj=np.eye(4);obj[:3,3]=[.2,.3,.4]
    hand={0:np.tile([.1,.101,.102],(2,21,1))};valid={0:np.ones((2,21),bool)}
    # Internal JSON ID 77 must never redirect the label to camera frame 77.
    result=pair_native([0],hand,valid,{0:camera},{0:(obj,np.ones(3),'Mug',77)})
    np.testing.assert_allclose(result['hand'][0,0,0],[.899,2.1,3.102],atol=1e-6)
    assert result['native_json_frame_id'][0]==77 and result['object_pose_valid'][0,0]
    local=np.linalg.inv(result['object_poses'][0,0])@np.r_[result['hand'][0,0,0],1]
    np.testing.assert_allclose(local[:3],[-.1,-.199,-.298],atol=1e-6)


def test_missing_or_reflected_camera_never_fabricates_valid_world_geometry():
    camera=np.diag([-1.,1,1,1]);obj=np.eye(4)
    hand={0:np.ones((2,21,3)),1:np.ones((2,21,3))};valid={i:np.ones((2,21),bool) for i in hand}
    result=pair_native([0,1],hand,valid,{0:camera},{i:(obj,np.ones(3),'Mug',i) for i in hand})
    assert not result['hand_valid'].any() and not result['object_pose_valid'].any()
