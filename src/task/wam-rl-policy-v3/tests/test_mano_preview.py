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
