"""Source semantics and physical clocks; no learned-quality assertions."""
import sys
import ast
import importlib.machinery
import json
import runpy
import signal
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools/run/ref5_data_expansion'
sys.path.insert(0, str(TOOLS))
from native_data import arctic_poses, eligible_rows, MANO_MCP, MANO_TIPS
from prepare_egodex_hands import native_hands, JOINT_SUFFIXES, camera_object_to_origin
from object_geometry import exact_mesh_diameter


def test_arctic_top_is_negative_z_rotation_and_mm_translation():
    params = np.array([[np.pi/2, 0, 0, 0, 1000, 2000, 3000]])
    bottom, top = arctic_poses(params)[0]
    assert np.allclose(bottom[:3, 3], [1, 2, 3])
    assert np.allclose(bottom[:3, :3] @ [1, 0, 0], [1, 0, 0])
    assert np.allclose(top[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-7)
    assert MANO_MCP == [13, 1, 4, 10, 7]
    assert MANO_TIPS == [744, 320, 443, 554, 671]


def test_missing_confidence_is_not_invented_and_zero_is_masked():
    pose = np.broadcast_to(np.eye(4), (30, 4, 4)).copy()
    pose[:, :3, 3] = [1, 2, 3]
    transforms = {side + suffix: pose for side in ('right', 'left') for suffix in JOINT_SUFFIXES}
    confidence = np.ones(30); confidence[4] = 0
    xyz, valid, conf, known, names = native_hands({'transforms': transforms,
                                                'confidences': {'rightHand': confidence}})
    assert valid.shape == (30, 2, 11)
    assert not valid[4, 0, 0] and np.all(xyz[4, 0, 0] == 0)
    assert valid[:, 1].all() and not known[:, 1].any()
    assert np.isnan(conf[:, 1]).all()
    assert names[0][0] == 'rightHand'


def test_stationary_object_has_zero_world_effect_with_moving_camera():
    camera = np.broadcast_to(np.eye(4), (28, 4, 4)).copy()
    camera[:, 0, 3] = np.linspace(0, 1, 28)
    world_object = np.eye(4); world_object[:3, 3] = [1, 2, 3]
    object_camera = np.linalg.inv(camera) @ world_object
    recovered = camera_object_to_origin(camera, object_camera)
    assert np.allclose(recovered, world_object)
    assert np.allclose(np.linalg.inv(recovered[3]) @ recovered[27], np.eye(4))


def test_window_rejects_clock_gap_future_invalid_pose_and_presence_changes():
    n = 29
    hand = np.zeros((n, 2, 11, 3)); hv = np.ones((n, 2), bool)
    poses = np.broadcast_to(np.eye(4), (n, 1, 4, 4)).copy()
    valid = np.ones((n, 1), bool); near = valid.copy(); time = np.arange(n)/30
    centers = np.zeros((1, 3))
    rows = eligible_rows(hand, hv, poses, valid, near, time, centers)
    assert len(rows) == 2 and rows[0, 1] == 3
    gap = time.copy(); gap[15:] += 1/30
    assert len(eligible_rows(hand, hv, poses, valid, near, gap, centers)) == 0
    invalid = valid.copy(); invalid[20] = False
    assert len(eligible_rows(hand, hv, poses, invalid, near, time, centers)) == 0
    changed = hv.copy(); changed[20, 0] = False
    assert len(eligible_rows(hand, changed, poses, valid, near, time, centers)) == 0


def test_current_only_anchor_cannot_be_selected_from_future_proximity():
    n = 29
    hand = np.zeros((n, 2, 11, 3)); hv = np.ones((n, 2), bool)
    poses = np.broadcast_to(np.eye(4), (n, 1, 4, 4)).copy()
    valid = np.ones((n, 1), bool); near = np.zeros_like(valid); near[20:] = True
    rows = eligible_rows(hand, hv, poses, valid, near, np.arange(n)/30, np.zeros((1, 3)))
    assert len(rows) == 0


def test_hull_diameter_matches_all_pairs_for_spatial_planar_and_linear_geometry():
    from scipy.spatial.distance import pdist
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(17)
    for points in [rng.normal(size=(200, 3)),
                   np.c_[rng.normal(size=(100, 2)), np.zeros(100)],
                   np.c_[np.arange(12), np.zeros((12, 2))]]:
        reference = float(pdist(points).max())
        actual = exact_mesh_diameter(points)
        transformed = points @ Rotation.from_rotvec([.7, -.3, .2]).as_matrix().T + [3, 4, 5]
        assert np.isclose(actual, reference, atol=1e-10)
        assert np.isclose(exact_mesh_diameter(transformed), reference, atol=1e-10)


@pytest.mark.parametrize('motion', ['translation', 'rotation'])
def test_mid_horizon_motion_returning_to_start_is_moving(motion):
    from scipy.spatial.transform import Rotation
    n = 28
    hand = np.zeros((n, 2, 11, 3)); hv = np.ones((n, 2), bool)
    poses = np.broadcast_to(np.eye(4), (n, 1, 4, 4)).copy()
    if motion == 'translation':
        poses[10, 0, 0, 3] = .01
    else:
        poses[10, 0, :3, :3] = Rotation.from_rotvec([0, 0, .1]).as_matrix()
    valid = np.ones((n, 1), bool)
    rows = eligible_rows(hand, hv, poses, valid, valid, np.arange(n)/30, np.zeros((1, 3)))
    assert rows.tolist() == [[0, 3, 1]]  # Native near-hand moving category.


def test_deadline_escapes_real_upstream_candidate_exception_loop(tmp_path):
    """Execute actual handler, candidate loop and run boundary without CUDA imports."""
    vendor = (TOOLS.parents[5]/'outputs/cm-pointflow-effect-pretrain/'
              'ref5-data-expansion-20261007/vendor/ObjectForesight-Data/step10_fpose.py')
    if not vendor.exists():
        pytest.skip('requires the acquired original ObjectForesight source')
    upstream_tree = ast.parse(vendor.read_text())
    loop = next(n for n in upstream_tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'choose_best_init')
    wrapper_tree = ast.parse((TOOLS/'upstream_object.py').read_text())
    handler = next(n for n in ast.walk(wrapper_tree) if isinstance(n, ast.FunctionDef)
                   and n.name == 'deadline')
    classes = [n for n in wrapper_tree.body if isinstance(n, ast.ClassDef)]
    run = next(n for n in wrapper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    boundary = next(n for n in run.body if isinstance(n, ast.Try) and
                    any(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                        and call.func.attr=='process_single_object' for call in ast.walk(n)))
    seen = []
    def candidate(**kwargs):
        seen.append(kwargs['frame_idx'])
        if len(seen) == 1:
            signal.setitimer(signal.ITIMER_REAL, .02)
            time.sleep(.1)
        return 1., .9, np.eye(4), object()
    namespace = dict(np=np,traceback=traceback,evaluate_init_candidate=candidate)
    exec(compile(ast.Module(body=classes+[handler],type_ignores=[]),
                 str(TOOLS/'upstream_object.py'),'exec'),namespace)
    exec(compile(ast.Module(body=[loop],type_ignores=[]),str(vendor),'exec'),namespace)
    cfg = SimpleNamespace(init_lock_scale=False)
    def process(*args):
        return namespace['choose_best_init'](None,None,[0,1],None,cfg,None,None,None)
    manifest = tmp_path/'upstream_run_manifest.json'
    namespace.update(time=time,json=json,signal=signal,manifest=manifest,
                     started=time.monotonic(),report=dict(status='RUNNING'),initialization_errors=[],
                     a=SimpleNamespace(output=tmp_path),cfg=cfg,
                     upstream=SimpleNamespace(process_single_object=process))
    previous = signal.signal(signal.SIGALRM,namespace['deadline'])
    try:
        with pytest.raises(BaseException,match='bounded run deadline'):
            exec(compile(ast.Module(body=[boundary],type_ignores=[]),
                         str(TOOLS/'upstream_object.py'),'exec'),namespace)
        assert seen == [0]
        assert json.loads(manifest.read_text())['status'] == 'TIMED_OUT'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)


def test_audit_entrypoint_disables_external_bytecode_write_attempts(monkeypatch,tmp_path):
    source = tmp_path/'readonly_source.py'
    source.write_text('value = 17\n')
    attempts = []
    # Intercept before loader.set_data can create directories or write bytes.
    monkeypatch.setattr(importlib.machinery.SourceFileLoader,'set_data',
                        lambda self,path,data,**kwargs: attempts.append(path))
    monkeypatch.setattr(sys,'dont_write_bytecode',False)
    namespace = runpy.run_path(str(TOOLS/'audit_native.py'),run_name='audit_entrypoint_test')
    attempts.clear()
    loaded = namespace['module'](source,'external_fixture')
    assert loaded.value == 17
    assert attempts == []
    assert not (tmp_path/'__pycache__').exists()


@pytest.mark.parametrize('code,status,expected', [
    (-15,'RUNNING','PROCESS_FAILED'), (1,'COMPLETED','PROCESS_FAILED'),
    (0,'RUNNING','INCOMPLETE_RESULT'), (0,'COMPLETED','COMPLETED'),
    (1,'TIMED_OUT','TIMED_OUT'), (0,'NO_INITIALIZATION','NO_INITIALIZATION')])
def test_stage_exit_code_cannot_leave_a_dead_job_running_or_successful(code,status,expected):
    stage_status=runpy.run_path(str(TOOLS/'run_original_clip.py'))['stage_status']
    assert stage_status(code,{'status':status}) == expected
