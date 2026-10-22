"""CPU interface checks, with a small geometry seam; no PhysX or training."""
import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
sys.path.insert(0, str(ROOT/'src/task/wam-rl-policy-v3/src'))
from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime
from ptv3_shared_wam_v6.events import EVENT_ENCODING
from ptv3_shared_wam_v6.contracts import SCHEMA
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v5.contracts import SPARSE11_IN_21
from test_v5_contract import Geometry, config, stats


def runtime():
    torch.set_num_threads(2)
    torch.manual_seed(196)
    rt = object.__new__(OnlineV6Runtime)
    rt.config = dict(config(), schema=SCHEMA, event_conditioning=True,
                     event_encoding=EVENT_ENCODING, event_dropout=.15,
                     voxel_origin_m=[-2., -2., -2.], voxel_m=.005)
    rt.device = torch.device('cpu')
    rt.inference_batch = 1
    rt.forecast_only = False
    rt.model = SharedWAMV6(stats(), rt.config, Geometry(8)).eval()
    rt.geometries = {'cube@1': dict(surface=np.zeros((2048, 3), np.float32),
        normals=np.tile([0., 0., 1.], (2048, 1)), query=np.zeros((32, 3), np.float32))}
    rt.seconds = 0.
    rt.calls = rt.samples = 0
    return rt


def inputs():
    rotation = Rotation.from_euler('z', .73)
    objects = np.tile(np.r_[[.2, .3, .8], rotation.as_quat()], (2, 4, 1))
    hand = np.arange(2*4*11*3).reshape(2, 4, 11, 3)*.0001 + objects[:, :, None, :3]
    goals = np.tile(np.eye(4), (2, 1, 1))
    goals[:, :3, :3] = rotation.as_matrix()
    goals[:, :3, 3] = objects[:, -1, :3] + [0., 0., .08]
    events = [dict(event_type=2, event_delta_steps=8., event_valid=True, event_hz=30),
              dict(event_type=1, event_delta_steps=40., event_valid=True, event_hz=30)]
    return objects, hand, goals, events, ['cube', 'cube']


@pytest.mark.parametrize('missing', ['voxel_origin_m', 'voxel_m'])
def test_missing_voxel_config_is_explicit_before_inference(missing):
    rt = runtime()
    del rt.config[missing]
    with pytest.raises(ValueError, match='v6 runtime config.*'+missing):
        rt.predict(*inputs())
    assert rt.calls == 0


@pytest.mark.parametrize('bad', [dict(voxel_m=0), dict(voxel_m=float('nan')),
    dict(voxel_m=[.005]), dict(voxel_origin_m=[0, 0]), dict(voxel_origin_m=[0, 0, float('inf')])])
def test_malformed_voxel_config_is_explicit(bad):
    rt = runtime()
    rt.config.update(bad)
    with pytest.raises(ValueError, match='v6 runtime config'):
        rt.predict(*inputs())


def test_real_runtime_full21_event_goal_mask_and_executor_clock():
    from wam_rl_policy_v3.executor_contract import ExecutorLayout
    from wam_rl_policy_v3.inspire_contract import WAM_TO_NATIVE
    from ptv3_shared_wam_v6.executor_bridge import full21_to_sparse11, world_tau_to_executor
    rt = runtime()
    captures, event_calls = [], []
    original = rt._trajectory
    hook = rt.model.event_goal.register_forward_pre_hook(lambda module, args: event_calls.append(args[0]))
    def capture(batch):
        value = original(batch)
        captures.append((batch, value.detach().numpy()))
        return value
    rt._trajectory = capture
    args = inputs()
    actual = rt.predict(*args)
    hook.remove()
    assert actual.shape == (2, 24, 11, 3) and len(captures) == len(event_calls) == 2
    for i, (batch, full) in enumerate(captures):
        assert full.shape == (1, 24, 2, 21, 3)
        sparse = full21_to_sparse11(full)
        assert sparse.shape == (1, 24, 2, 11, 3)
        np.testing.assert_array_equal(sparse, full[..., list(SPARSE11_IN_21), :])
        np.testing.assert_allclose(actual[i], sparse[0, :, 0] @ Rotation.from_quat(args[0][i, -1, 3:]).as_matrix().T, atol=1e-7)
        assert batch['start_hand_valid'].sum() == 11
        assert not batch['start_hand_valid'][:, 1].any() and not full[:, :, 1].any()
        assert not batch['dense_hand_valid'].any()
        np.testing.assert_allclose(batch['goal_object_delta'][0], np.tile([0, 0, .08], (32, 1)), atol=1e-6)
        assert event_calls[i]['event_type'].item() == args[3][i]['event_type']
        assert event_calls[i]['event_delta_steps'].item() == args[3][i]['event_delta_steps']
        assert event_calls[i]['event_hz'].item() == 30
    tau = world_tau_to_executor(actual, native_hz=30, native_order=True)
    np.testing.assert_allclose(tau, actual[:, :8, WAM_TO_NATIVE], atol=1e-7)
    layout = ExecutorLayout('inspire', 411, 12, 30)
    task = np.zeros((2, 10), np.float32); task[:, -3] = 1
    canonical_tau = world_tau_to_executor(actual, native_hz=30)
    packed = layout.pack(np.zeros((2, 411)), task, tau=canonical_tau, tau_valid=[True, False])
    assert packed.shape == (2, 686)
    np.testing.assert_allclose(packed[0, 421:685], canonical_tau[0].reshape(-1))
    assert packed[0, -1] == 1 and not packed[1, 421:].any()
    np.testing.assert_allclose(layout.packet()['tau_times_s'], np.arange(1, 9)/30)


def test_two_hand_mapping_keeps_slots_and_60hz_interpolates_from_zero():
    from ptv3_shared_wam_v6.executor_bridge import full21_to_sparse11, world_tau_to_executor
    full = np.arange(24*2*21*3).reshape(24, 2, 21, 3)
    mapped = full21_to_sparse11(full)
    assert mapped.shape == (24, 2, 11, 3)
    np.testing.assert_array_equal(mapped[:, 1], full[:, 1, list(SPARSE11_IN_21)])
    world = np.broadcast_to(np.arange(1, 25)[:, None, None]/30, (24, 11, 3))
    tau = world_tau_to_executor(world, native_hz=60)
    np.testing.assert_allclose(tau[:, 0, 0], np.arange(1, 9)/60, atol=1e-8)


def test_real_runtime_rejects_wrong_valid_event_clock():
    rt = runtime()
    args = inputs(); args[3][0]['event_hz'] = 60
    with pytest.raises(ValueError, match='30Hz'):
        rt.predict(*args)


def test_official_ppo_is_still_unwired_source_assertion():
    source = (ROOT/'src/task/wam-rl-policy-v3/tools/run/inspire_measured_official_ppo.py').read_text()
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert any(n.module == 'wam_rl_policy_v3.wam_v4_runtime' and any(a.name == 'V4Runtime' for a in n.names) for n in imports)
    assert 'V4Runtime' in calls and 'OnlineV6Runtime' not in calls
    assert not any('ptv3_shared_wam_v6' in (n.module or '') for n in imports)
