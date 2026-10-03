#!/usr/bin/env python3
"""Collect Gate 1 transitions with explicit future-consequence tensors.

The existing physical-value collector already records the exact simulator
reward and action noise.  This wrapper adds measured object/hand tensors to
each transition; an offline assembler later forms grouped future windows.
No model is trained in this phase.
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_source() -> str:
    source = (ROOT / "scripts/run_cm_physical_value_environment.py").read_text()
    old_physical = """            gap_before = gap() if bridge is not None else None
            obs, base_reward, done, info = self.env_step(self.env, action)"""
    new_physical = """            gap_before = gap() if bridge is not None else None
            # Capture measured tensors before env_step so row t is aligned
            # with the pre-action state and action a_t.
            physical_before = dict(
                object_root=task._target_states.clone(),
                hand_body_position=task._rigid_body_pos[:, task._contact_body_ids].clone(),
                hand_body_quaternion=task._rigid_body_rot[:, task._contact_body_ids].clone(),
                hand_force=task._contact_forces[:, task._contact_body_ids].clone(),
                object_force=task._tar_contact_forces.clone(),
            )
            obs, base_reward, done, info = self.env_step(self.env, action)"""
    if source.count(old_physical) != 1:
        raise ValueError("physical collector env_step contract drift")
    source = source.replace(old_physical, new_physical)
    old = """                              step=step.clone(), progress=phase, start_frame=start_frame.clone(),
                              motion_id=motion.clone(), noise_std=noise.clone())"""
    new = """                              step=step.clone(), progress=phase, start_frame=start_frame.clone(),
                              motion_id=motion.clone(), noise_std=noise.clone(),
                              object_root=physical_before["object_root"],
                              hand_body_position=physical_before["hand_body_position"],
                              hand_body_quaternion=physical_before["hand_body_quaternion"],
                              hand_force=physical_before["hand_force"],
                              object_force=physical_before["object_force"])"""
    if source.count(old) != 1:
        raise ValueError("physical collector values contract drift")
    source = source.replace(old, new)
    old_payload = """            payload.update(schema=SCHEMA, gamma=gamma, control_dt=task.dt,
                           motion_names=list(task.motion_file), source_sha256=ARGS.checkpoint_sha256)"""
    new_payload = """            payload.update(schema=SCHEMA, gamma=gamma, control_dt=task.dt,
                           motion_names=list(task.motion_file), source_sha256=ARGS.checkpoint_sha256,
                           physical_timing="pre_env_step",
                           effect_definition=("future object root pose and twist in world frame"),
                           interaction_definition=("future contact-body poses/quaternions and measured "
                                                   "hand/object forces; pair identity unavailable"))"""
    if source.count(old_payload) != 1:
        raise ValueError("physical collector payload contract drift")
    source = source.replace(old_payload, new_payload)
    old_guard = 'if ARGS.mode == "collect" and ARGS.checkpoint_sha256 != SOURCE_SHA:'
    new_guard = ('if ARGS.mode == "collect" and ARGS.checkpoint_sha256 != SOURCE_SHA '
                 'and not os.environ.get("REF2DEX_ALLOW_DIAGNOSTIC_SOURCE"):')
    if source.count(old_guard) != 1:
        raise ValueError("physical collector source guard drift")
    source = source.replace(old_guard, new_guard)
    return source


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    if "--allow-diagnostic-source" in sys.argv:
        sys.argv.remove("--allow-diagnostic-source")
        os.environ["REF2DEX_ALLOW_DIAGNOSTIC_SOURCE"] = "1"
    exec(compile(build_source(), str(ROOT / "scripts/run_cm_physical_value_environment.py"), "exec"),
         {"__name__": "__main__", "__file__": str(ROOT / "scripts/run_cm_physical_value_environment.py")})
