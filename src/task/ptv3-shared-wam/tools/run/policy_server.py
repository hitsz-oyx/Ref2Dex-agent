"""Serve the full shared-WAM/R pair from graspenv to the native physics peer."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from ptv3_shared_wam.paths import ROOT, add_dependencies

add_dependencies()
from trajectory_policy.wam_ipc import send, receive, peer_pid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--socket", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--controls", type=int, required=True)
    p.add_argument("--sequence")
    p.add_argument("--retargeter", type=Path)
    p.add_argument("--model-version", choices=("v1", "v2"), default="v1")
    a = p.parse_args()
    execution_task = TASK if a.model_version == "v1" else TASK.parent / "ptv3-shared-wam-v2"
    if a.model_version == "v2":
        sys.path.insert(0, str(execution_task / "src"))
    a.output, a.socket, a.checkpoint = a.output.resolve(), a.socket.resolve(), a.checkpoint.resolve()
    if (a.output.exists() or ROOT / "outputs" / execution_task.name not in a.output.parents
            or ROOT / "tmp" not in a.socket.parents or a.socket.exists()
            or len(os.fsencode(str(a.socket))) >= 108):
        raise ValueError("fresh Task output and short project-local Unix socket required")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(a.socket.parent),
                      TRITON_CACHE_DIR=str(a.socket.parent / "triton"))
    import numpy as np
    import torch
    from ptv3_shared_wam.checkpoint import SCHEMA, atomic_json, load_checkpoint
    from ptv3_shared_wam.model import SharedWAM
    from ptv3_shared_wam.robot import RobotPair
    from ptv3_shared_wam.data import point_labels, shared_collate
    from trajectory_policy.wam_shadow import robot_goal_sample
    torch.set_num_threads(2)
    if a.model_version == "v2":
        from ptv3_shared_wam_v2.native_execution import load_robot_checkpoint as load_checkpoint
        from ptv3_shared_wam_v2.paths import SCHEMA
    packet = load_checkpoint(a.checkpoint)
    if packet["stage"] != "robot":
        raise ValueError("jointly fitted robot pair required")
    from ptv3_shared_wam.native_inputs import resolve_native_inputs, load_native_arrays
    selected = resolve_native_inputs(packet, a.sequence)
    canonical = load_native_arrays(selected)["canonical"]
    if a.model_version == "v2":
        from ptv3_shared_wam_v2.robot_model import SharedWAMV2Robot
        model = SharedWAMV2Robot(packet["stats"], packet["config"], packet["config"]["action_representation"])
    else:
        model = RobotPair(SharedWAM(packet["stats"], packet["config"]),action_representation=packet['config'].get('action_representation','native-command.v1'))
    model.load_state_dict(packet["model"], strict=True)
    from ptv3_shared_wam.native_inputs import load_execution_retargeter
    model.retargeter = load_execution_retargeter(packet, a.retargeter)
    model = model.cuda().eval()
    a.output.mkdir()
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(a.socket)); listener.listen(1); listener.settimeout(120)
    status = dict(schema=SCHEMA, status="READY", checkpoint=str(a.checkpoint), stage="robot", queries=0,
                  native_controller=packet["config"]["native_controller"])
    status['action_representation']=model.retargeter.action_representation
    status["native_input"] = selected
    status["retargeter_override"] = str(a.retargeter.resolve()) if a.retargeter else None
    atomic_json(a.output / "status.json", status)
    try:
        connection, _ = listener.accept()
        with connection:
            connection.settimeout(120)
            hello = receive(connection)
            if hello.get("schema") != SCHEMA or hello.get("pid") != peer_pid(connection):
                raise ValueError("owned shared-WAM peer required")
            send(connection, dict(status, pid=os.getpid()))
            previous = -1
            while True:
                request = receive(connection)
                if request.get("stop"):
                    break
                tick = request["tick"]
                if not previous < tick < a.controls:
                    raise ValueError("query ticks must increase inside the episode")
                previous = tick
                started = time.monotonic()
                sample = robot_goal_sample(request["object_history"], request["hand_history"],
                    request["planned_object_goal"], canonical, packet["config"]["object_points"], tick)
                # Dummy labels are collate placeholders; no label enters encode.
                sample.update(point_labels(sample, packet["config"]["object_points"]))
                batch = {key: value.cuda() for key, value in shared_collate([sample]).items()}
                state = torch.as_tensor(request["native_state"], device="cuda", dtype=torch.float32)[None]
                if state.shape != (1, 87) or not torch.isfinite(state).all():
                    raise ValueError("finite own87D native state required")
                from ptv3_shared_wam.robot import native_trajectory
                with torch.inference_mode():
                    prediction = model.forecaster(batch)
                    tau = native_trajectory(prediction["trajectory"], batch["start_hand_valid"])
                    chunk = model.retargeter(state, tau)
                if not torch.isfinite(chunk).all():
                    raise FloatingPointError("nonfinite action chunk")
                send(connection, dict(tick=tick, action_chunk=chunk[0].cpu().tolist(),
                                      action_representation=model.retargeter.action_representation,
                                      native_tau=tau[0].cpu().tolist(), elapsed_s=time.monotonic() - started))
                status["queries"] += 1
                atomic_json(a.output / "status.json", status)
        status["status"] = "COMPLETED"
        atomic_json(a.output / "status.json", status)
    except BaseException as error:
        status.update(status="FAILED", error=repr(error))
        atomic_json(a.output / "status.json", status)
        raise
    finally:
        listener.close()
        a.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
