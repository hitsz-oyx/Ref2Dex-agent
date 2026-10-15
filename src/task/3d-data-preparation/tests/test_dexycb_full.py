from pathlib import Path
import sys
import numpy as np
import pytest
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
from convert_dexycb_full import CAMERA,JOINTS,decode,split_for_subject


def test_official_s1_subject_partition():
    assert [i for i in range(1,11) if split_for_subject(i)=='train']==[1,2,3,4,5,6,10]
    assert split_for_subject(7)=='dev'
    assert [i for i in range(1,11) if split_for_subject(i)=='test']==[8,9]
    with pytest.raises(ValueError):split_for_subject(11)


def test_native_camera_labels_keep_missing_clock_and_hand_identity(tmp_path):
    folder=tmp_path/'session';(folder/CAMERA).mkdir(parents=True)
    (folder/'meta.yml').write_text(yaml.safe_dump(dict(num_frames=28,serials=[CAMERA],mano_sides=['left'],ycb_ids=[1],ycb_grasp_ind=0)))
    points=(np.arange(63).reshape(1,21,3)/1000+.2).astype(np.float32)
    pose=np.eye(4,dtype=np.float32)[None,:3]
    for frame in (0,1):
        np.savez(folder/CAMERA/f'labels_{frame:06d}.npz',joint_3d=points if frame==0 else np.full_like(points,-1),pose_m=np.ones((1,51),np.float32) if frame==0 else np.zeros((1,51),np.float32),pose_y=pose)
    _,data,missing=decode(folder)
    assert missing==26 and len(data['time'])==28
    np.testing.assert_allclose(np.diff(data['time']),1/30)
    np.testing.assert_array_equal(data['hand'][0,1],points[0,JOINTS])
    assert data['hand_valid'][0,1].all() and not data['hand_valid'][:,0].any()
    assert not data['hand_valid'][1:].any()
    assert data['object_pose_valid'][:2].all() and not data['object_pose_valid'][2:].any()
