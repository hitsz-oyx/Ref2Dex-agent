"""Known camera transforms and split isolation for the native H2O importer."""
from collections import defaultdict
from pathlib import Path
import sys
import zipfile
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
from convert_h2o_bulk import convert_take, native_split


def native():
    data=defaultdict(dict)
    for i in range(30):
        camera=np.array([[0,-1,0,1+i*.01],[1,0,0,2],[0,0,1,3],[0,0,0,1.]])
        obj=np.eye(4);obj[:3,3]=[.2,.3,.4]
        joints=np.arange(63,dtype=float).reshape(21,3)/1000+.1
        data['hand_pose'][i]=np.r_[1,joints.ravel(),0,(joints+1).ravel()]
        data['cam_pose'][i]=camera.ravel();data['obj_pose_rt'][i]=np.r_[2,obj.ravel()]
    return data


def test_native_sides_units_points_and_moving_world_camera():
    data=native();s,oid=convert_take(data,list(range(30)))
    assert oid==2 and s['hand_valid'][:,1].all() and not s['hand_valid'][:,0].any()
    # Left native wrist (.100,.101,.102), then rotate z90 and translate (1,2,3).
    np.testing.assert_allclose(s['hand'][0,1,0],[.899,2.1,3.102],atol=1e-6)
    np.testing.assert_allclose(s['hand'][0,1,6],[.887,2.112,3.114],atol=1e-6)
    np.testing.assert_allclose(s['object_poses'][0,0,:3,3],[.7,2.2,3.4],atol=1e-6)
    np.testing.assert_allclose(s['hand'][29,1,0]-s['hand'][0,1,0],[.29,0,0],atol=1e-6)
    # Both transforms must cancel the same moving camera in the fixed current anchor.
    h=np.r_[s['hand'][29,1,0],1]
    local=np.linalg.inv(s['object_poses'][29,0])@h
    np.testing.assert_allclose(local[:3],[-.1,-.199,-.298],atol=1e-6)


def test_missing_camera_and_background_object_never_fabricate_valid_geometry():
    data=native();del data['cam_pose'][7];data['obj_pose_rt'][8][0]=0
    data['cam_pose'][9][0]=-1  # reflection: no valid geometry may enter this world frame
    s,_=convert_take(data,list(range(30)))
    assert not s['hand_valid'][[7,9]].any() and not s['object_pose_valid'][[7,8,9]].any()
    assert s['object_pose_valid'][6].all()


def test_official_split_refuses_session_leakage(tmp_path):
    path=tmp_path/'splits.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('label_split/pose_train.txt','subject3/k2/0/cam4/rgb/000000.png\n')
        z.writestr('label_split/pose_val.txt','subject3/k2/1/cam4/rgb/000000.png\n')
        z.writestr('label_split/pose_test.txt','subject4/h1/0/cam4/rgb/000000.png\n')
    with pytest.raises(ValueError,match='session split overlap'):native_split(path)
