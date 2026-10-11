import importlib.util
from pathlib import Path

import numpy as np
import pytest

script = Path(__file__).resolve().parents[1] / 'tools/audit/audit_video_reprojection.py'
spec = importlib.util.spec_from_file_location('reprojection_audit', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_radial_depth_error_has_zero_reprojection_but_nonzero_world_error():
    target = np.asarray([[1., 0, 2]])
    values = module.endpoint_errors(2*target, target, np.eye(4), np.eye(3))
    np.testing.assert_allclose(values['world_epe'], np.sqrt(5))
    np.testing.assert_allclose(values['radial_abs'], np.sqrt(5))
    np.testing.assert_allclose(values['transverse'], 0, atol=1e-12)
    np.testing.assert_allclose(values['projected_epe'], 0)
    np.testing.assert_allclose(values['radial_sq'], values['total_sq'])


def test_transverse_error_and_rotation_preserve_full_metric_denominator():
    target, pred = np.asarray([[0.,0,2]]),np.asarray([[.2,0,2.]])
    w2c = np.eye(4)
    w2c[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
    k = np.diag([100.,100,1])
    values = module.endpoint_errors(pred,target,w2c,k)
    np.testing.assert_allclose(values['radial_abs'],0)
    np.testing.assert_allclose(values['transverse'],.2)
    np.testing.assert_allclose(values['world_epe'],.2)
    np.testing.assert_allclose(values['projected_epe'],10)
    assert len(values['world_epe'])==1 and not values['invalid_projection'].any()


def test_behind_camera_forecast_is_retained_and_invalid_target_fails():
    target = np.asarray([[0.,0,2],[0,0,2.]])
    pred = np.asarray([[1.,1,-1],[0,0,2.]])
    values=module.endpoint_errors(pred,target,np.eye(4),np.eye(3))
    assert values['invalid_projection'].tolist()==[True,False]
    assert len(values['world_epe'])==2
    np.testing.assert_allclose(values['world_epe'],[np.sqrt(11),0])
    with pytest.raises(ValueError):
        module.endpoint_errors(pred,pred,np.eye(4),np.eye(3))
