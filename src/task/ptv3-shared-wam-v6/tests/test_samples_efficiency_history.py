"""Fairness contracts for the information shared by every experimental arm."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src')]
path=Path(__file__).resolve().parents[1]/'tools/run/samples_efficiency_env.py'
spec=importlib.util.spec_from_file_location('samples_environment',path)
environment=importlib.util.module_from_spec(spec);spec.loader.exec_module(environment)


def test_history_startup_has_no_invented_motion_and_is_translation_invariant():
    objects=np.array([[.1,.2,.3,0,0,0,1.]])
    hands=np.arange(33,dtype=float).reshape(1,11,3)*.01
    packed=environment.history_features(objects,hands).reshape(4,40)
    assert np.count_nonzero(packed[:,:3])==0
    assert np.count_nonzero(packed[:,7:])==0
    shifted=objects.copy();shifted[:,:3]+=[4.,2.,1.]
    np.testing.assert_array_equal(packed.reshape(-1),environment.history_features(shifted,hands+[4.,2.,1.]))


def test_reference_is_full_horizon_and_translation_equivariant():
    hand=np.array([[i*.004,.01,.08] for i in range(11)])
    points=np.array([[a,b,c] for a in [-.02,.02] for b in [-.02,.02] for c in [0.,.04]])
    pose=np.eye(4);goal=pose.copy();goal[2,3]=.08
    reference=environment.geometric_reference(hand,pose,points,goal)
    assert reference.shape==(24,11,3)
    shift=np.array([.7,-.8,.2]);pose[:3,3]+=shift;goal[:3,3]+=shift
    np.testing.assert_allclose(reference,environment.geometric_reference(hand+shift,pose,points,goal),atol=1e-7)
