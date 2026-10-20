import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.mano_preview import forecast_world,palm_scale


def test_prediction_world_uses_current_hand_and_rotation_only():
 current=np.arange(33).reshape(11,3)*.01;delta=np.tile([.1,0,0],(24,11,1))
 anchor=np.eye(4);anchor[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();anchor[:3,3]=[5,6,7]
 result=forecast_world(current,delta,anchor)
 np.testing.assert_allclose(result[0],current,atol=1e-7)
 np.testing.assert_allclose(result[1:],np.broadcast_to(current+[0,.1,0],(24,11,3)),atol=1e-7)
 assert result.shape==(25,11,3)


def test_bad_anchor_or_truncated_executor_tau_rejected():
 with pytest.raises(ValueError):forecast_world(np.zeros((11,3)),np.zeros((8,11,3)),np.eye(4))
 reflection=np.eye(4);reflection[0,0]=-1
 with pytest.raises(ValueError):forecast_world(np.zeros((11,3)),np.zeros((24,11,3)),reflection)


def test_scale_is_rigid_invariant_and_current_only():
 neutral=np.arange(33).reshape(11,3)*.01;rotation=Rotation.from_euler('xyz',[.2,.4,.1]).as_matrix()
 current=neutral@rotation.T*1.2+[8,3,4]
 scale,raw=palm_scale(current,neutral)
 assert scale==pytest.approx(1.2);assert raw==pytest.approx(1.2)
 with pytest.raises(ValueError):palm_scale(np.zeros((11,3)),neutral)


def test_visual_description_removes_collision_without_touching_source(tmp_path):
 import importlib.util
 from pathlib import Path
 import xml.etree.ElementTree as ET
 path=Path(__file__).resolve().parents[1]/'tools/run/render_wam_mano_preview.py'
 spec=importlib.util.spec_from_file_location('render_wam_mano_preview',path)
 renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
 source=tmp_path/'native.urdf';destination=tmp_path/'visual.urdf'
 text='<robot name="hand"><link name="base"><visual><geometry><mesh filename="meshes/hand.obj"/></geometry></visual><collision><geometry><box size="1 1 1"/></geometry></collision></link></robot>'
 source.write_text(text)
 renderer.visual_urdf(source,destination)
 result=ET.parse(destination).getroot()
 assert not list(result.iter('collision'))
 assert next(result.iter('mesh')).get('filename')==str(tmp_path/'meshes/hand.obj')
 assert source.read_text()==text


def test_rolling_windows_cover_future_once_and_trim_final_chunk():
 from wam_rl_policy_v3.mano_preview import rolling_windows
 windows=rolling_windows(22)
 assert [anchor for anchor,_ in windows]==[3,11,19]
 assert [len(frames) for _,frames in windows]==[8,8,2]
 np.testing.assert_array_equal(np.concatenate([frames for _,frames in windows]),np.arange(4,22))
 for anchor,frames in windows:
  assert frames[0]==anchor+1 and frames[-1]<=anchor+8
 with pytest.raises(ValueError):rolling_windows(4)


def test_native_png_writer_roundtrip(tmp_path):
 import importlib.util
 from pathlib import Path
 from PIL import Image
 path=Path(__file__).resolve().parents[1]/'tools/run/render_wam_mano_preview.py'
 spec=importlib.util.spec_from_file_location('png_renderer',path)
 renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
 pixels=np.arange(90,dtype=np.uint8).reshape(5,6,3)
 destination=tmp_path/'frame.png';renderer.write_rgb_png(destination,pixels)
 np.testing.assert_array_equal(np.asarray(Image.open(destination)),pixels)


def test_shared_camera_covers_distant_approach_and_zoomed_contact():
 import importlib.util
 from pathlib import Path
 path=Path(__file__).resolve().parents[1]/'tools/run/render_wam_mano_preview.py'
 spec=importlib.util.spec_from_file_location('camera_renderer',path)
 renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
 obj=np.eye(4);obj[:3,3]=[0,-1,.85]
 near=np.tile([0,-1,.9],(11,1));far=near+[0,1,0]
 eye1,target1=renderer.tracking_camera(far,obj,far+[.1,0,0])
 eye2,target2=renderer.tracking_camera(near,obj,near)
 assert np.linalg.norm(eye1-target1)>np.linalg.norm(eye2-target2)
 assert -.6<target1[1]<-.4
 assert np.isfinite(np.r_[eye1,target1,eye2,target2]).all()


def test_predicted_object_recovers_rigid_motion_in_current_anchor():
 from wam_rl_policy_v3.mano_preview import forecast_object_poses
 query=np.random.default_rng(246).normal(size=(32,3))*.03
 local=np.eye(4);local[:3,:3]=Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix();local[:3,3]=[.02,.03,.01]
 delta=query@local[:3,:3].T+local[:3,3]-query
 anchor=np.eye(4);anchor[:3,:3]=Rotation.from_euler('y',.8).as_matrix();anchor[:3,3]=[1,2,3]
 poses,residual=forecast_object_poses(query,np.tile(delta,(24,1,1)),anchor)
 np.testing.assert_allclose(poses,np.broadcast_to(anchor@local,(24,4,4)),atol=1e-10)
 assert residual.max()<1e-10
 # Chaining uses the OLD predicted pose as its next anchor, not original truth.
 next_poses,_=forecast_object_poses(query,np.tile(delta,(24,1,1)),poses[7])
 np.testing.assert_allclose(next_poses[0],anchor@local@local,atol=1e-10)


def test_object_projection_reports_nonrigid_shape_error_and_rejects_nan():
 from wam_rl_policy_v3.mano_preview import forecast_object_poses
 query=np.random.default_rng(246).normal(size=(32,3))*.03
 delta=np.tile(query*.5,(24,1,1))
 poses,residual=forecast_object_poses(query,delta,np.eye(4))
 assert residual.min()>.005
 np.testing.assert_allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-10)
 delta[0,0,0]=np.nan
 with pytest.raises(ValueError):forecast_object_poses(query,delta,np.eye(4))
