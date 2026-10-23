"""Control invariants and deliberate-error witnesses, not physical qualification."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from wam_rl_policy_v3.object_tracking import ObjectPoseTracker, ObjectSteadyCorrection, hand_setpoint


def fixture(**kwargs):
    q = np.full(18, .3);q[:6] = 0.
    mask = np.ones(18, bool);mask[3:6] = False
    limits = dict(has_limits=mask, lower=np.r_[[-1.]*6, [0.]*12], upper=np.r_[[1.]*6, [2.]*12])
    tracker = ObjectPoseTracker(q, np.eye(4), np.zeros(18), np.ones(18),
                                correction_joint_limits=limits, **kwargs)
    return tracker, q


def step(tracker, q, measured=None, desired=None, valid=True, **kwargs):
    measured = np.eye(4) if measured is None else measured
    desired = np.eye(4) if desired is None else desired
    palm = np.eye(4);palm[:3, 3] = q[:3]
    return tracker.step(q, np.zeros(18), palm, desired, np.zeros(3), np.zeros(3),
                        np.ones((5, 3)), valid, False,
                        measured_object=measured, desired_object=desired, hand_object=np.eye(4), **kwargs)


def test_opt_out_is_bitwise_same_command_and_missing_measurement_is_error():
    a, q = fixture();b, _ = fixture(object_steady_correction=False)
    np.testing.assert_array_equal(step(a, q)[0], step(b, q)[0])
    on, _ = fixture(object_steady_correction=True)
    with pytest.raises(ValueError, match='measured object'):
        on.step(q, q*0, np.eye(4), np.eye(4), q[:3]*0, q[:3]*0, np.ones((5, 3)), True, True)


def test_opt_out_bitwise_matches_frozen_prepatch_ff_and_pd_path():
    import subprocess
    import types
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    path = 'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/object_tracking.py'
    source = subprocess.check_output(['git','show','01ce376a:'+path],cwd=root,text=True)
    old = types.ModuleType('wam_rl_policy_v3.prepatch_object_tracking')
    old.__package__ = 'wam_rl_policy_v3';exec(compile(source, path, 'exec'), old.__dict__)
    rng = np.random.default_rng(531)
    for ff in (False, True):
        tracker, q = fixture()
        prior = old.ObjectPoseTracker(q, np.eye(4), np.zeros(18), np.ones(18))
        for _ in range(100):
            target = np.eye(4);target[:3,3] = rng.normal(0,.025,3)
            target[:3,:3] = Rotation.from_rotvec(rng.normal(0,.03,3)).as_matrix()
            args = (q, rng.normal(0,.01,18), np.eye(4), target, rng.normal(0,.1,3), rng.normal(0,.2,3), rng.normal(0,.5,(5,3)), True, ff)
            np.testing.assert_array_equal(tracker.step(*args)[0], prior.step(*args)[0])


def test_so3_shortest_geodesic_world_frame_and_not_euler_subtraction():
    measured = np.eye(4);desired = np.eye(4)
    measured[:3, :3] = Rotation.from_euler('ZYX', [179, 50, 30], degrees=True).as_matrix()
    world_axis = np.array([1., 2., 3.]);world_axis /= np.linalg.norm(world_axis)
    desired[:3, :3] = Rotation.from_rotvec(world_axis*.06).as_matrix() @ measured[:3, :3]
    corrector = ObjectSteadyCorrection();_, info = corrector.propose(measured, desired)
    np.testing.assert_allclose(info['rotation_error_world_rad'], world_axis*.06, atol=1e-12)
    np.testing.assert_allclose(corrector.rotation, world_axis*.06*.7/30, atol=1e-12)
    wrong_body = Rotation.from_matrix(measured[:3, :3].T @ desired[:3, :3]).as_rotvec()
    assert np.linalg.norm(wrong_body - world_axis*.06) > .01  # kills wrong-frame mutant


def test_norm_limits_are_isotropic_and_unwind_when_error_reverses():
    measured = np.eye(4);desired = np.eye(4);desired[:3, 3] = [1, 2, 3]
    desired[:3, :3] = Rotation.from_rotvec(np.array([1., 2., 3.])*.3).as_matrix()
    corrector = ObjectSteadyCorrection()
    for _ in range(1000):
        oldp, oldr = corrector.position.copy(), corrector.rotation.copy()
        corrector.propose(measured, desired)
        assert np.linalg.norm(corrector.position-oldp) <= .015/30+1e-12
        assert np.linalg.norm(corrector.rotation-oldr) <= .08/30+1e-12
    assert np.linalg.norm(corrector.position) == pytest.approx(.04)
    assert np.linalg.norm(corrector.rotation) == pytest.approx(.14)
    np.testing.assert_allclose(corrector.position/np.linalg.norm(corrector.position), np.array([1, 2, 3])/np.sqrt(14))
    # Per-axis clipping mutant would reach sqrt(3)*limit, caught above.
    before = corrector.position.copy()
    desired[:3, 3] *= -1
    corrector.propose(measured, desired)
    assert np.linalg.norm(corrector.position) < np.linalg.norm(before)


def test_contact_loss_clears_bias_and_new_episode_has_no_integral():
    tracker, q = fixture(object_steady_correction=True)
    desired = np.eye(4);desired[0, 3] = .01
    step(tracker, q, desired=desired)
    assert tracker.steady.position[0] > 0
    _, info = step(tracker, q, desired=desired, valid=False)
    assert info['object_steady_correction']['reason'] == 'contact_invalid'
    np.testing.assert_array_equal(tracker.steady.position, np.zeros(3))
    new, _ = fixture(object_steady_correction=True)
    np.testing.assert_array_equal(new.steady.position, np.zeros(3))


def test_finite_joint_boundary_disables_bias_but_continuous_angles_do_not():
    tracker, q = fixture(object_steady_correction=True)
    desired = np.eye(4);desired[0, 3] = .01
    q[3:6] = [2*np.pi*50, 0, 0]
    _, info = step(tracker, q, desired=desired)
    assert info['object_steady_correction']['accepted']
    q[10] = .005
    _, info = step(tracker, q, desired=desired)
    assert info['object_steady_correction']['reason'] == 'finite_joint_boundary'
    assert np.linalg.norm(tracker.steady.position) == 0


def test_saturation_rejects_integral_growth_without_changing_original_action():
    tracker, q = fixture(object_steady_correction=True);off, _ = fixture()
    desired = np.eye(4);desired[0, 3] = .1
    raw, info = step(tracker, q, desired=desired)
    expected, _ = step(off, q, desired=desired)
    assert info['object_steady_correction']['reason'] == 'command_saturated'
    np.testing.assert_array_equal(raw, expected)
    for _ in range(200):step(tracker, q, desired=desired)
    assert np.linalg.norm(tracker.steady.position) == 0  # kills no-antiwindup mutant


def test_unreachable_corrected_target_never_drives_outer_boundary():
    tracker, q = fixture(object_steady_correction=True)
    q[0] = .9799
    desired = np.eye(4);desired[0, 3] = .9799
    measured = desired.copy();measured[0, 3] -= .01
    _, info = step(tracker, q, measured=measured, desired=desired)
    assert info['object_steady_correction']['reason'] == 'corrected_target_rejected'
    assert np.linalg.norm(tracker.steady.position) == 0


def test_finger_actions_unchanged_and_scoring_reference_not_mutated():
    tracker, q = fixture(object_steady_correction=True);off, _ = fixture()
    desired = np.eye(4);desired[0, 3] = .01;snapshot = desired.copy()
    raw, info = step(tracker, q, desired=desired)
    baseline, _ = step(off, q, desired=desired)
    np.testing.assert_array_equal(raw[6:], baseline[6:])
    np.testing.assert_array_equal(desired, snapshot)
    assert info['wrist_target'][0][3] > desired[0, 3]
    assert info['original_wrist_target'][0][3] == desired[0, 3]


def test_retained_ff_includes_corrected_rotation_lever_derivative():
    tracker, q = fixture(object_steady_correction=True)
    measured = np.eye(4);desired = np.eye(4);relative = np.eye(4);relative[0, 3] = .1
    tracker.steady.rotation[:] = [.01, -.02, .03]
    v = np.array([.01, .02, .03]);w = np.array([.1, -.2, .3])
    target, hv, hw = hand_setpoint(desired, v, w, relative)
    q[:3] = target[:3, 3];palm = target.copy()
    _, info = tracker.step(q, q*0, palm, target, hv, hw, np.ones((5, 3)), True, True,
                           measured_object=measured, desired_object=desired, hand_object=relative)
    assert info['object_steady_correction']['accepted']
    eps = 1e-6
    def mapped(t):
        obj = desired.copy();obj[:3, 3] += v*t
        obj[:3, :3] = Rotation.from_rotvec(w*t).as_matrix() @ desired[:3, :3]
        return tracker.steady.apply(obj) @ np.linalg.inv(relative)
    derivative = (mapped(eps)[:3, 3]-mapped(-eps)[:3, 3])/(2*eps)
    np.testing.assert_allclose(info['velocity_ff_world'], derivative, atol=1e-9)


@pytest.mark.parametrize('enabled', [False, True])
def test_biased_surrogate_static_plant_witness(enabled):
    # A deterministic engineering witness, NOT the native physical plant.
    tracker, q = fixture(object_steady_correction=enabled)
    desired = np.eye(4);desired[0, 3] = .025
    disturbance = np.array([.003, -.002, .001])
    for _ in range(300):
        measured = np.eye(4);measured[:3, 3] = q[:3]
        raw, _ = step(tracker, q, measured=measured, desired=desired)
        q[:3] += raw[:3] - disturbance
    error = np.linalg.norm(q[:3]-desired[:3, 3])
    assert error < .0001 if enabled else error > .008  # kills missing-error-feedback mutant


@pytest.mark.parametrize('mutant', ['missing_feedback','component_clip','no_antiwindup','stale_contact','body_frame'])
def test_deliberately_broken_implementations_are_detected(monkeypatch, mutant):
    import wam_rl_policy_v3.object_tracking as module
    if mutant == 'missing_feedback':
        def broken(self, measured, desired):
            return (self.position.copy(), self.rotation.copy()), {}
        monkeypatch.setattr(ObjectSteadyCorrection, 'propose', broken)
        witness = lambda: test_biased_surrogate_static_plant_witness(True)
    elif mutant == 'component_clip':
        monkeypatch.setattr(module, '_norm_cap', lambda vector, maximum: np.clip(vector, -maximum, maximum))
        witness = test_norm_limits_are_isotropic_and_unwind_when_error_reverses
    elif mutant == 'no_antiwindup':
        monkeypatch.setattr(ObjectSteadyCorrection, 'restore', lambda self, saved: None)
        witness = test_saturation_rejects_integral_growth_without_changing_original_action
    elif mutant == 'stale_contact':
        monkeypatch.setattr(ObjectSteadyCorrection, 'clear', lambda self: None)
        witness = test_contact_loss_clears_bias_and_new_episode_has_no_integral
    else:
        real = ObjectSteadyCorrection.propose
        def broken(self, measured, desired):
            saved, info = real(self, measured, desired)
            info['rotation_error_world_rad'] = Rotation.from_matrix(measured[:3,:3].T @ desired[:3,:3]).as_rotvec()
            return saved, info
        monkeypatch.setattr(ObjectSteadyCorrection, 'propose', broken)
        witness = test_so3_shortest_geodesic_world_frame_and_not_euler_subtraction
    with pytest.raises(AssertionError):
        witness()
