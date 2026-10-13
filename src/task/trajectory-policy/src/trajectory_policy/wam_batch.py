"""One WAM forward for several independent environments.

A control step used to collate and run each environment by itself. The rows
in one packet are independent samples: they may sit at different demonstration
phases. The reply keeps env_id and tick so a row cannot be applied to the
wrong environment.
"""
import time

import numpy as np

MAX_BATCH = 32
ROW_CORE = ('tick', 'object_history', 'hand_history', 'planned_object_goal')


def request_rows(request, goal_key, adapter_on, open_queries, query_index):
    """Return (rows, batched). A legacy one-env packet stays a single row."""
    core = set(ROW_CORE)
    if adapter_on:
        core |= {'proprio', goal_key}
    if 'batch' in request:
        rows = request.get('batch')
        if set(request) != {'batch'} or not isinstance(rows, list) or not rows:
            raise ValueError('batch envelope requires a nonempty batch list')
        if len(rows) > MAX_BATCH:
            raise ValueError('batch larger than 32 is outside this probe')
        seen = set()
        wants_state = any(isinstance(row, dict) and 'retarget_state' in row for row in rows)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('batch row must be an object')
            allowed = set(core) | {'env_id'}
            if wants_state:
                allowed |= {'retarget_state'}
            if set(row) != allowed:
                raise ValueError('live H4/object-goal-only request required')
            if not isinstance(row['env_id'], int) or isinstance(row['env_id'], bool) or row['env_id'] in seen:
                raise ValueError('batch rows need unique env ids')
            seen.add(row['env_id'])
            _check_tick(row['tick'], open_queries, query_index)
            if wants_state and (not isinstance(row['retarget_state'], list) or len(row['retarget_state']) != 87):
                raise ValueError('retarget state must be 87-D')
        return rows, True
    if set(request) != core:
        raise ValueError('live H4/object-goal-only request required')
    _check_tick(request['tick'], open_queries, query_index)
    return [request], False


def _check_tick(tick, open_queries, query_index):
    if open_queries:
        if not isinstance(tick, int) or isinstance(tick, bool) or not 0 <= tick <= 542:
            raise ValueError('open query tick outside the demonstration')
    elif tick != query_index:
        raise ValueError('live H4/object-goal-only sequential request required')


def collate_robot_batch(rows, canonical, goal_key, sample_base, adapter_on):
    """Stack independent H4 samples. sample_id is metadata, not a scene seed."""
    import torch
    from goal_point_wam.data import goal_collate
    from trajectory_policy.wam_shadow import robot_goal_sample

    samples = []
    for index, row in enumerate(rows):
        samples.append(robot_goal_sample(
            np.asarray(row['object_history'], np.float32),
            np.asarray(row['hand_history'], np.float32),
            np.asarray(row['planned_object_goal'], np.float32),
            canonical, sample_id=int(sample_base) + index))
    batch = {key: value.cuda() for key, value in goal_collate(samples).items()}
    if adapter_on:
        for key in ('proprio', goal_key):
            value = np.stack([np.asarray(row[key], np.float32) for row in rows])
            if value.shape[0] != len(rows) or not np.isfinite(value).all():
                raise ValueError('nonfinite robot context')
            batch[key] = torch.tensor(value, device='cuda')
        batch['robot_mask'] = torch.ones(len(rows), device='cuda', dtype=torch.bool)
    return batch


def predict_trajectory(model, adapter, batch, reference_mode, conditioned_actor):
    """Actor-only trajectory. World is not conditioned and is not differentiated."""
    import torch

    batch.pop('_dropped_per_row', None)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        scene = model.scene(batch)
        tokens, mask = model.goal(
            batch['start_object_points'], batch['goal_object_points'], batch['object_goal_valid'])
        raw = model.observed_hand_condition(batch)
        if reference_mode:
            residual = adapter(batch['proprio'], batch['reference_goal'], batch['robot_mask']).to(raw.dtype)
            prediction = model.trajectory(
                scene['scene_tokens'], scene['scene_mask'], tokens, mask,
                raw + residual[:, None, None]).float()
        elif adapter is not None:
            if conditioned_actor is None:
                raise ValueError('control adapter forward is missing')
            prediction = conditioned_actor(
                model, (scene, tokens, mask, raw), adapter, batch).float()
        else:
            prediction = model.trajectory(
                scene['scene_tokens'], scene['scene_mask'], tokens, mask, raw).float()
    return prediction


def chunk_base(chunk_net, rows, tau):
    """Unclamped 18-D first command. Residual clipping stays on the native side."""
    import torch
    from trajectory_policy.retargeter_data import ACTIVE

    state = np.stack([np.asarray(row['retarget_state'], np.float32) for row in rows])
    if state.shape != (len(rows), 87) or tau.shape != (len(rows), 24, 33):
        raise ValueError('chunk state/tau batch contract differs')
    if not np.isfinite(state).all() or not np.isfinite(tau).all():
        raise ValueError('finite chunk state and tau required')
    with torch.inference_mode():
        encoded = chunk_net(
            torch.as_tensor(state, device='cuda'), torch.as_tensor(tau, device='cuda'))
    base = np.zeros((len(rows), 18), np.float32)
    base[:, list(ACTIVE)] = encoded[:, 0].float().detach().cpu().numpy()
    if not np.isfinite(base).all():
        raise ValueError('nonfinite chunk base')
    return base


def _rigid(translation):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray(translation, np.float32)
    return pose


def synthetic_rows(count, origin):
    """Distinct in-grid scenes. Phases differ on purpose."""
    from trajectory_policy.robot_context import control_context

    origin = np.asarray(origin, np.float32)
    center = origin + np.array([1.0, 1.0, 0.8], np.float32)
    rows = []
    for index in range(count):
        shift = np.array([0.01 * index, -0.008 * index, 0.0], np.float32)
        obj = np.stack([_rigid(center + shift) for _ in range(4)])
        hand = np.zeros((4, 11, 3), np.float32)
        hand[:] = obj[-1, :3, 3] + np.linspace(-0.02, 0.02, 11, dtype=np.float32)[:, None]
        goal = _rigid(center + shift + np.array([0.0, 0.0, 0.02], np.float32))
        ref = np.stack([goal, goal, goal])
        prop, robot_goal = control_context(np.zeros(18, np.float32), np.zeros(18, np.float32),
                                            np.linspace(-0.1, 0.1, 18, dtype=np.float32), obj[-1], ref)
        state = np.linspace(-0.2, 0.2, 87, dtype=np.float32) + 0.01 * index
        rows.append(dict(
            env_id=index, tick=int((0, 32, 48, 62)[index % 4]),
            object_history=obj.tolist(), hand_history=hand.tolist(),
            planned_object_goal=goal.tolist(), proprio=prop.tolist(),
            robot_goal=robot_goal.tolist(), retarget_state=state.tolist()))
    return rows


def _forward_rows(model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
                  rows, outlier_limit, sample_base=0):
    import torch
    from trajectory_policy.scene_workspace import mask_scene_outliers
    from trajectory_policy.wam_shadow import active_hand_tau

    torch.cuda.synchronize()
    started = time.perf_counter()
    batch = collate_robot_batch(rows, canonical, 'robot_goal', sample_base, adapter is not None)
    dropped = mask_scene_outliers(
        batch, model.scene.voxel_origin_m, float(model.scene.voxel_size_m), limit=outlier_limit)
    prediction = predict_trajectory(model, adapter, batch, reference_mode, conditioned_actor)
    tau = active_hand_tau(prediction.detach().cpu().numpy(), batch['start_hand_valid'].detach().cpu().numpy())
    base = chunk_base(chunk_net, rows, tau) if chunk_net is not None and 'retarget_state' in rows[0] else None
    listed = tau.tolist()
    if base is not None:
        base.tolist()
    torch.cuda.synchronize()
    return tau, base, time.perf_counter() - started, int(dropped), len(json_bytes(listed))


def json_bytes(payload):
    import json
    return json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()


def _median(values):
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def run_forward_benchmark(model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
                          repeats=8):
    """Time serial-4, two batch-2 forwards, and one batch-4 forward. Then grow B."""
    import subprocess
    import threading

    import torch

    origin = model.scene.voxel_origin_m.detach().float().cpu().numpy()
    rows = synthetic_rows(32, origin)
    # Warm the kernels on one row before any timed mode.
    _forward_rows(model, adapter, canonical, reference_mode, conditioned_actor, chunk_net, rows[:1], 64)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def equate(count):
        serial_tau, serial_base = [], []
        for index, row in enumerate(rows[:count]):
            tau, base, _elapsed, _dropped, _size = _forward_rows(
                model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
                [row], 64, sample_base=index)
            serial_tau.append(tau[0])
            serial_base.append(None if base is None else base[0])
        tau, base, _elapsed, dropped, _size = _forward_rows(
            model, adapter, canonical, reference_mode, conditioned_actor, chunk_net, rows[:count], 64)
        tau_error = float(np.max(np.abs(tau - np.stack(serial_tau))))
        base_error = None if base is None else float(np.max(np.abs(base - np.stack(serial_base))))
        return dict(batch=count, tau_max_abs_m=tau_error, base_max_abs=base_error,
                    workspace_points_dropped=dropped)

    samples = []
    stop = False

    def poll():
        while not stop:
            text = subprocess.check_output(
                ['nvidia-smi', '-i', '0', '--query-gpu=utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits'], text=True).strip()
            util, memory = text.split(',')
            samples.append((int(util), int(memory)))
            time.sleep(0.2)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    modes = []
    try:
        equivalence = [equate(4)]
        plans = (
            ('serial-4', 1, 4),
            ('batch-2x2', 2, 2),
            ('batch-4', 4, 1),
        )
        for name, width, groups in plans:
            modes.append(_time_mode(
                model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
                rows, name, width, groups, repeats, samples))
        for width in (8, 16, 32):
            try:
                modes.append(_time_mode(
                    model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
                    rows, f'batch-{width}', width, 1, max(3, repeats // 2), samples))
                equivalence.append(equate(width))
            except RuntimeError as error:
                if 'out of memory' not in str(error).lower():
                    raise
                torch.cuda.empty_cache()
                modes.append(dict(mode=f'batch-{width}', oom=True, error=str(error).splitlines()[-1]))
                break
    finally:
        stop = True
        watcher.join(timeout=1)
    peak_util = max((item[0] for item in samples), default=0)
    peak_memory = max((item[1] for item in samples), default=0)
    ipc = {width: _ipc_roundtrip(rows[:width]) for width in (1, 2, 4, 8, 16)}
    for mode in modes:
        measured = ipc.get(mode.get('batch'))
        if mode.get('oom') or measured is None:
            continue
        roundtrip_s, packet_bytes = measured
        mode['ipc_roundtrip_s'] = roundtrip_s
        mode['packet_bytes'] = packet_bytes
        mode['query_s'] = mode['group_wall_s'] + roundtrip_s * mode['groups']
        mode['queries_per_s'] = mode['samples'] / mode['query_s']
    return dict(
        equivalence=equivalence, modes=modes, ipc_roundtrip_s=ipc and {
            str(width): item[0] for width, item in ipc.items()},
        gpu_util_max_percent=peak_util, nvidia_memory_max_mib=peak_memory,
        torch_peak_allocated_mib=max(
            (mode.get('torch_peak_allocated_mib', 0) for mode in modes), default=0),
        card_mib=24576)


def _ipc_roundtrip(rows):
    """Request plus reply on a socketpair, without a GPU forward."""
    import socket

    from trajectory_policy.wam_ipc import SCHEMA, receive, send

    request = dict(batch=rows)
    reply = dict(schema=SCHEMA, batch=True, elapsed_s=0.0, responses=[
        dict(env_id=row['env_id'], tick=row['tick'], tau=np.zeros((24, 33), np.float32).tolist(),
             anchor_object_pose=row['object_history'][-1], base_action=[0.0] * 18,
             workspace_points_dropped=0)
        for row in rows])
    left, right = socket.socketpair()
    try:
        send(left, request)
        receive(right)
        send(right, reply)
        receive(left)
        elapsed = []
        for _ in range(20):
            started = time.perf_counter()
            send(left, request)
            receive(right)
            send(right, reply)
            receive(left)
            elapsed.append(time.perf_counter() - started)
        return _median(elapsed), len(json_bytes(request)) + len(json_bytes(reply))
    finally:
        left.close()
        right.close()


def _time_mode(model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
               rows, name, width, groups, repeats, gpu_samples):
    import torch

    _forward_rows(model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
                  rows[:width], 64)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    begin = len(gpu_samples)
    walls = []
    forwards = []
    nbytes = 0
    for _ in range(repeats):
        started = time.perf_counter()
        for group in range(groups):
            _tau, _base, elapsed, _dropped, size = _forward_rows(
                model, adapter, canonical, reference_mode, conditioned_actor, chunk_net,
                rows[group * width:(group + 1) * width], 64, sample_base=group * width)
            forwards.append(elapsed)
            nbytes = size
        walls.append(time.perf_counter() - started)
    count = width * groups
    wall = _median(walls)
    seen = gpu_samples[begin:]
    return dict(mode=name, batch=width, groups=groups, samples=count, repeats=repeats,
                forward_s=_median(forwards), group_wall_s=wall,
                samples_per_s=count / wall, reply_tau_bytes=nbytes, oom=False,
                gpu_util_max_percent=max((item[0] for item in seen), default=0),
                nvidia_memory_max_mib=max((item[1] for item in seen), default=0),
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20)
