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
    a = p.parse_args()
    a.output, a.socket, a.checkpoint = a.output.resolve(), a.socket.resolve(), a.checkpoint.resolve()
    if (a.output.exists() or ROOT / "outputs/ptv3-shared-wam" not in a.output.parents
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
    packet = load_checkpoint(a.checkpoint)
    if packet["stage"] != "robot":
        raise ValueError("jointly fitted robot pair required")
    model = RobotPair(SharedWAM(packet["stats"], packet["config"]))
    model.load_state_dict(packet["model"], strict=True)
    model = model.cuda().eval()
    canonical_path = packet["provenance"]["canonical"]
    with np.load(canonical_path, allow_pickle=False) as stream:
        canonical = {key: stream[key].copy() for key in stream.files}
    a.output.mkdir()
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(a.socket)); listener.listen(1); listener.settimeout(120)
    status = dict(schema=SCHEMA, status="READY", checkpoint=str(a.checkpoint), stage="robot", queries=0,
                  native_controller=packet["config"]["native_controller"])
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
