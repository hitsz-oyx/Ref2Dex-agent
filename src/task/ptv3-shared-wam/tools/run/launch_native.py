"""Own both runtime peers, GPU and deadline for K_exec=1/4/8 execution."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--controls", type=int, default=542)
    p.add_argument("--execution-steps", type=int, choices=(1, 4, 8), default=4)
    p.add_argument("--seed", type=int, default=292)
    p.add_argument("--deadline-seconds", type=int, default=600)
    p.add_argument("--experiment-card", type=Path)
    p.add_argument("--smoke", action="store_true", help="engineering execution, at most16 controls")
    p.add_argument("--sequence", help="required exact corpus record for RL660 checkpoints")
    p.add_argument("--retargeter", type=Path, help="compatible R-only eight-action candidate")
    p.add_argument("--preflight-only", action="store_true", help="validate inputs and R on CPU without starting GPU peers")
    p.add_argument("--model-version", choices=("v1", "v2"), default="v1")
    p.add_argument('--shared-actor', action='store_true', help='Explicit v2 shared bridge; both student rows use their own measured inputs')
    a = p.parse_args()
    execution_task = TASK if a.model_version == "v1" else TASK.parent / "ptv3-shared-wam-v2"
    if a.model_version == "v2":
        sys.path.insert(0, str(execution_task / "src"))
    out = a.output.resolve()
    if (out.exists() or ROOT / "outputs" / execution_task.name not in out.parents
            or not 1 <= a.controls <= 542 or not 1 <= a.deadline_seconds <= 600):
        raise ValueError("fresh Task output and bounded controls/deadline required")
    engineering = a.smoke or a.controls <= 4
    if a.smoke and a.controls > 16:
        raise ValueError("engineering execution capped at16 controls")
    if not engineering and a.experiment_card is None:
        raise ValueError("non-smoke native execution requires an experiment card")
    if a.experiment_card:
        import yaml
        card = a.experiment_card.resolve()
        if execution_task / "docs/experiments" not in card.parents:
            raise ValueError("Task-local experiment card required")
        experiment = yaml.safe_load(card.read_text().split("---", 2)[1])
        if experiment.get("task") != execution_task.name or a.seed not in experiment.get("seeds", []):
            raise ValueError("task and assigned rollout seed must match the card")
    from ptv3_shared_wam.checkpoint import directory_bytes, load_checkpoint,check_shared_storage
    if a.model_version == "v2":
        from ptv3_shared_wam_v2.native_execution import load_robot_checkpoint as load_checkpoint
    if a.shared_actor and (a.model_version != 'v2' or a.retargeter):
        raise ValueError('shared bridge requires v2 and its original jointly trained actor')
    packet = load_checkpoint(a.checkpoint, allow_shared=True) if a.shared_actor else load_checkpoint(a.checkpoint)
    if packet["stage"] != "robot" or (not engineering and packet["provenance"].get("engineering_only")):
        raise ValueError("robot pair required; engineering weights only admit engineering smoke")
    from ptv3_shared_wam.native_inputs import resolve_native_inputs, load_native_arrays
    selected = resolve_native_inputs(packet, a.sequence)
    if (packet["config"].get("action_objective") == "control-aware-first.v1"
            and a.execution_steps != 1):
        raise ValueError("first-control objective requires explicit execution-steps=1")
    if a.retargeter:
        from ptv3_shared_wam.native_inputs import load_execution_retargeter
        candidate=load_execution_retargeter(packet, a.retargeter)
        if candidate.first_only and a.execution_steps != 1:
            raise ValueError("R-only first-control repair must first be evaluated with execution-steps=1")
    arrays = load_native_arrays(selected)
    if selected["kind"] == "rl660":
        if "motion" not in selected:
            raise ValueError("RL660 native execution requires a resolved geometric motion")
        if a.controls + 24 > len(arrays["oracle"]):
            raise ValueError("controls must leave a complete 24-step oracle horizon")
    if a.preflight_only:
        print(json.dumps(dict(status="PREFLIGHT_PASS", native_input=selected,
                             retargeter_override=str(a.retargeter.resolve()) if a.retargeter else None,
                             execution_steps=a.execution_steps, controls=a.controls,
                             scope="CPU input/weight contract only; physics untested")))
        return
    if subprocess.check_output(["nvidia-smi", "-i", str(a.gpu), "--query-compute-apps=pid",
                                "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU occupied before execution")
    storage_config = dict(packet['config'])
    if a.model_version == 'v2':
        storage_config['min_free_disk_bytes'] = max(20 * 2**30, storage_config.get('min_free_disk_bytes', 0))
    check_shared_storage(storage_config)
    out.mkdir(parents=True)
    # Short socket name avoids the Linux108-byte AF_UNIX limit.
    scratch = ROOT / "tmp" / ("swam-%d" % os.getpid())
    scratch.mkdir(mode=0o700)
    address = scratch / "s.sock"
    if len(os.fsencode(str(address))) >= 108:
        raise ValueError("project socket address too long")
    checkpoint = a.checkpoint.resolve()
    common = ["--model-version", a.model_version, "--checkpoint", str(checkpoint), "--gpu", str(a.gpu), "--controls", str(a.controls)]
    if a.shared_actor:
        common.append('--shared-actor')
    if a.sequence:
        common.extend(["--sequence", a.sequence])
    if a.retargeter:
        common.extend(["--retargeter", str(a.retargeter.resolve())])
    status = dict(model_version=a.model_version, status="STARTING", checkpoint=str(checkpoint), controls=a.controls,
                  execution_steps=a.execution_steps, physical_gpu=a.gpu, seed=a.seed,
                  scratch=str(scratch), engineering_only=engineering)
    status["native_input"] = selected
    status['shared_actor'] = a.shared_actor
    status["retargeter_override"] = str(a.retargeter.resolve()) if a.retargeter else None
    def write():
        from ptv3_shared_wam.checkpoint import atomic_json
        atomic_json(out / "group_status.json", status)
    processes, handles = [], []
    started = time.monotonic()
    def guard():
        if time.monotonic() - started > a.deadline_seconds:
            raise TimeoutError("owned group deadline")
        pids = set(map(int, subprocess.check_output(["nvidia-smi", "-i", str(a.gpu),
            "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split()))
        if pids - {process.pid for process in processes}:
            raise RuntimeError("foreign GPU process during execution group")
        if directory_bytes(out, scratch) > 1_000_000_000:
            raise RuntimeError("execution artifacts exceed1GB")
    try:
        for kind, python in (("server", "/home2/wyy/miniconda3/envs/graspenv/bin/python"),
                             ("native", "/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python")):
            script = TASK / "tools/run" / ("policy_server.py" if kind == "server" else "execute_native.py")
            log = (out / (kind + ".log")).open("w"); handles.append(log)
            extra = [] if kind == "server" else ["--server-pid", str(processes[0].pid),
                "--execution-steps", str(a.execution_steps), "--seed", str(a.seed),
                "--deadline-seconds", str(a.deadline_seconds)]
            if kind == "native" and engineering:
                extra.append("--engineering-only")
            process = subprocess.Popen([python, str(script), *common, "--socket", str(address),
                "--output", str(out / kind), *extra], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, TMPDIR=str(scratch), PYTHONDONTWRITEBYTECODE="1"))
            processes.append(process)
            status[kind + "_pid"] = process.pid
            write()
            if kind == "server":
                while not address.exists():
                    guard()
                    if process.poll() is not None:
                        raise RuntimeError("server exited before readiness")
                    time.sleep(.5)
        status["status"] = "RUNNING"; write()
        while any(process.poll() is None for process in processes):
            guard()
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("execution peer failed; see immutable peer logs")
            time.sleep(1.)
        status.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                      exit_codes=[process.returncode for process in processes])
        write()
    except BaseException as error:
        status.update(status="FAILED", error=repr(error)); write()
        raise
    finally:
        # Only process groups started by this launcher are stopped.
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
