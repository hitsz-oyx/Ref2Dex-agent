"""Production source-reference geometry and acquisition horizon, with injected faults."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def worker_module():
    path=Path(__file__).resolve().parents[1]/'tools/run/run_source_ceiling_worker.py'
    spec=importlib.util.spec_from_file_location('source_PD_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def reference_state():
    q=np.zeros((13,1,18));q[:,:,2]=1.2;q[:,:,6:]=.2
    objects=np.zeros((13,1,13));objects[:,:,2]=1.1;objects[:,:,6]=1.
    palm=np.zeros((13,1,7));palm[:,:,2]=1.2;palm[:,:,3:7]=Rotation.from_euler('x',.7).as_quat()
    action=np.zeros((12,12));action[:,0]=.02;action[:,5]=.1;action[:,6:]=-.6
    ref=dict(q=q,objects=objects,palm_pose=palm,action=action)
    hand_pose=np.eye(4);hand_pose[2,3]=1.2;hand_pose[:3,:3]=Rotation.from_euler('x',.7).as_matrix()
    obj=np.eye(4);obj[2,3]=1.1
    return ref,q[0,0].copy(),hand_pose,obj


def test_production_source_target_follows_live_object_translation():
    worker=worker_module();ref,q,palm,obj=reference_state();live=obj.copy();live[0,3]+=.05
    original=worker.source_native_targets
    def check():
        raw,mapped=worker.source_native_targets(ref,0,np.zeros(18),np.ones(18),q,palm,live)
        np.testing.assert_allclose(mapped[:3]-raw[:3],[.05,0,0],atol=1e-7)
        np.testing.assert_array_equal(mapped[6:],raw[6:])
    check()
    def retain_world(*args):
        raw,_=original(*args);return raw,raw.copy()
    with patch.object(worker,'source_native_targets',retain_world):
        with pytest.raises(AssertionError):check()


def test_production_source_target_maps_rotation_through_native_mount():
    worker=worker_module();ref,q,palm,obj=reference_state();live=obj.copy();live[:3,:3]=Rotation.from_euler('z',.4).as_matrix()
    original=worker.source_native_targets
    def check():
        raw,mapped=worker.source_native_targets(ref,0,np.zeros(18),np.ones(18),q,palm,live)
        expected=live[:3,:3]@Rotation.from_euler('XYZ',raw[3:6]).as_matrix()@palm[:3,:3]
        actual=Rotation.from_euler('XYZ',mapped[3:6]).as_matrix()@palm[:3,:3]
        np.testing.assert_allclose(actual,expected,atol=1e-7)
        np.testing.assert_array_equal(mapped[6:],raw[6:])
    check()
    def retain_world(*args):
        raw,_=original(*args);return raw,raw.copy()
    with patch.object(worker,'source_native_targets',retain_world):
        with pytest.raises(AssertionError):check()


def test_live_acquisition_reference_cannot_run_into_recorded_postgrasp_or_release():
    worker=worker_module();ref,_,_,_=reference_state();key=SimpleNamespace(command_frame=4)
    def check():
        length=worker.source_pregrasp_length(ref,key);cursor=0
        guard=dict(unloading=False)
        for _ in range(100):cursor=worker.advance_source_reference(cursor,length,True,guard,True)
        assert cursor==4, 'While live acquisition is unfinished, retain the valid grasp command instead of replaying recorded departure/release'
        assert len(ref['action'])==12 and len(ref['q'])==13, 'Keep the entire raw recording'
    check()
    with patch.object(worker,'source_pregrasp_length',lambda ref,key:len(ref['action'])):
        with pytest.raises(AssertionError):check()
    with pytest.raises(ValueError):worker.source_pregrasp_length(ref,SimpleNamespace(command_frame=100))
