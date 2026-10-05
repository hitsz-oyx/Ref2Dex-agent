#!/usr/bin/env python3
"""One randomized intervention per native episode, synchronized full resets only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/DExplore/dexplore")]
from isaacgym import gymapi  # import before torch
import torch
import evaluate as original
from src.task.CmResidual.physical_value_contract import HoldTracker
from src.task.CmResidual.physical_value_live import snapshot, context, contacts
from src.task.CmResidual.dexplore_cm_geometry import DExploreCmv2GeometryBridge, dexplore_root_pose
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/src"))
from intervention import CHUNK, WINDOW, HISTORY, ARM_NAMES, residuals, all_arms_have_headroom, update_predecision_hold, decode_assignment, decode_amplitude_assignment, apply_feedback_residual, per_finger_residuals, PER_FINGER_ARM_NAMES

SOURCE_SHA = "16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f"
ARGS = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InterventionPlayer(original.EvalPlayer):
    @torch.no_grad()
    def run(self):
        task, device = self.env.task, self.device
        task._enable_early_termination = False
        task._adaptive_kappa_enabled = False
        task._hybrid_init_prob = .5
        if abs(task.dt - 1/30) > 1e-8 or len(task.motion_file) != 3:
            raise ValueError("airplane motion/control interval drift")
        self.model.eval()
        n = task.num_envs
        per_finger = ARGS.intervention_set == "per-finger-range"
        arm_names = PER_FINGER_ARM_NAMES if per_finger else ARM_NAMES
        delta = per_finger_residuals(ARGS.finger_range_fraction, device) if per_finger else residuals(device)
        capture_geometry = per_finger or ARGS.amplitudes != [1.]
        rng = torch.Generator(device=device).manual_seed(ARGS.assignment_seed)
        asset = ROOT / "third_party/DExplore/dexplore/data/assets"
        bridge = DExploreCmv2GeometryBridge(hand_urdf=asset / "inspire_hand_new/inspire_hand_right.urdf",
                                          object_urdf=asset / "mjcf/airplane.urdf", device=device)
        if capture_geometry:
            sample = ARGS.run_dir / "object_surface_sample.pt"
            torch.save(dict(points=torch.as_tensor(bridge.geometry.object_local).cpu(),
                            normals=torch.as_tensor(bridge.geometry.object_normal).cpu(),
                            seed=42, stride=8), sample)
            (ARGS.run_dir / "geometry_provenance.json").write_text(json.dumps(dict(
                sample_sha256=sha(sample), seed=42, stride=8,
                contract="local object visual surface; transformed by measured world object pose"), indent=2)+"\n")
        tip_names = ("index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip")
        tip_ids = task._key_body_ids[[task.cfg["env"]["keyBodies"].index(name) for name in tip_names]]
        hand_base_id = task._key_body_ids[task.cfg["env"]["keyBodies"].index("hand_base_link")]
        tracker = HoldTracker(n, device)
        started = time.monotonic()
        packets, summaries = [], []
        full_ids = torch.arange(n, device=device)
        empty_ids = full_ids[:0]

        def physical():
            # Direct measured simulator body/root state, never identity-root FK.
            pos = task._rigid_body_pos[:, task._contact_body_ids]
            rot = task._rigid_body_rot[:, task._contact_body_ids]
            obj_points, obj_normals = bridge.geometry.object(dexplore_root_pose(task._target_states))
            distance, nearest = torch.cdist(pos, obj_points[:, ::8]).min(-1)
            normals = torch.gather(obj_normals[:, ::8], 1, nearest[:, :, None].expand(-1, -1, 3))
            return torch.cat((task._target_states.clone(), torch.cat((pos, rot), -1).flatten(1),
                              task._contact_forces[:, task._contact_body_ids].flatten(1),
                              task._tar_contact_forces.clone(), distance,
                              contacts(task).bool().all(-1, keepdim=True).float()), -1), normals

        for wave in range(ARGS.waves):
            obs = self.env_reset(full_ids)
            if self.get_batch_size(obs["obs"], 1) != n:
                raise ValueError("native player batch mismatch")
            if self.is_rnn:
                self.init_rnn()
            tracker.reset(full_ids, task._target_states[:, 2])
            rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
            terminal = torch.zeros(n, device=device, dtype=torch.bool)
            arms = torch.full((n,), -1, device=device, dtype=torch.long)
            durations = torch.zeros_like(arms)
            amplitudes = torch.ones(n, device=device)
            decision_tick = torch.full_like(arms, -1)
            hold_steps = torch.zeros_like(arms)
            hold_steps_at = torch.zeros_like(arms)
            previous = torch.zeros(n, 18, device=device)
            history = torch.zeros(n, HISTORY, 139, device=device)
            initial = torch.zeros(n, 72, device=device)
            history_at = torch.zeros_like(history)
            obs_at = torch.zeros(n, obs["obs"].shape[-1], device=device)
            context_at = torch.zeros(n, 435, device=device)
            base_at = torch.zeros(n, 18, device=device)
            root_at = torch.zeros(n, 13, device=device)
            motion_at = torch.zeros_like(arms)
            progress_at = torch.zeros_like(arms)
            start_at = task.start_times.clone()
            outcomes = torch.zeros(n, WINDOW, 72, device=device)
            normals_at = torch.zeros(n, 5, 3, device=device)
            normal_trajectory = torch.zeros(n, WINDOW, 5, 3, device=device)
            native_q = torch.zeros(n, WINDOW, 18, device=device)
            tips_at = torch.zeros(n, 5, 3, device=device)
            tip_trajectory = torch.zeros(n, WINDOW, 5, 3, device=device)
            hand_base_at = torch.zeros(n, 7, device=device)
            hand_base_trajectory = torch.zeros(n, WINDOW, 7, device=device)
            actual = torch.zeros(n, WINDOW, 18, device=device)
            bases = torch.zeros_like(actual)
            pd_targets = torch.zeros_like(actual)
            pd_base_targets = torch.zeros_like(actual)
            valid_steps = torch.zeros(n, WINDOW, device=device, dtype=torch.bool)
            eligibility_counts = dict(force_proxy=0, region=0, geometry=0, headroom=0, assigned=0)
            region_seen = torch.zeros(n, device=device, dtype=torch.bool)
            geometry_seen = torch.zeros_like(region_seen)
            headroom_seen = torch.zeros_like(region_seen)
            for tick in range(ARGS.max_steps):
                if time.monotonic() - started > ARGS.wall_seconds:
                    raise TimeoutError("bounded intervention collection deadline")
                obs = self.env_reset(empty_ids)  # adapter only, no physical reset
                state = snapshot(task, tracker)
                phys, normals = physical()
                hold_steps = update_predecision_hold(hold_steps, phys[:, 2], rest, phys[:, 71] > .5)
                compact = torch.cat((state, previous, task._humanoid_root_states.clone(),
                                     phys[:, 13:66]), -1)
                if compact.shape[-1] != 139:
                    raise ValueError("history physical contract drift")
                history = torch.cat((history[:, 1:], compact[:, None]), 1)
                base = self.get_action(obs, True).clamp(-1, 1)
                age = tick - decision_tick
                # Event trigger depends only on current state, before random draw.
                eligible = (~terminal) & (arms < 0) & (tick >= HISTORY-1)
                eligible &= (task.max_episode_length[task.data_id] - task.progress_buf > WINDOW + 1)
                eligible &= (task.rollout_length - (task.progress_buf-task.start_times) > WINDOW + 1)
                eligible &= phys[:, 71] > .5
                eligibility_counts["force_proxy"] += int(eligible.sum())
                if ARGS.decision_region == "early-hold":
                    eligible &= hold_steps >= 6
                eligibility_counts["region"] += int(eligible.sum())
                region_seen |= eligible
                eligible &= phys[:, 66:71].amin(-1) < .06
                eligibility_counts["geometry"] += int(eligible.sum())
                geometry_seen |= eligible
                eligible &= all_arms_have_headroom(base, delta*max(ARGS.amplitudes))
                eligibility_counts["headroom"] += int(eligible.sum())
                headroom_seen |= eligible
                chosen = eligible.nonzero(as_tuple=False).flatten()
                if len(chosen):
                    if len(ARGS.amplitudes) > 1:
                        draw = torch.randint(7*len(ARGS.amplitudes), (len(chosen),), generator=rng, device=device)
                        arms[chosen], amplitudes[chosen] = decode_amplitude_assignment(draw, ARGS.amplitudes)
                        durations[chosen] = ARGS.durations[0]
                    else:
                        draw = torch.randint(len(arm_names)*len(ARGS.durations), (len(chosen),), generator=rng, device=device)
                        arms[chosen], durations[chosen] = decode_assignment(draw, ARGS.durations, len(arm_names))
                        amplitudes[chosen] = ARGS.amplitudes[0]
                    decision_tick[chosen] = tick
                    hold_steps_at[chosen] = hold_steps[chosen]
                    initial[chosen], history_at[chosen] = phys[chosen], history[chosen]
                    normals_at[chosen] = normals[chosen]
                    tips_at[chosen] = task._rigid_body_pos[chosen][:, tip_ids]
                    hand_base_at[chosen] = torch.cat((task._rigid_body_pos[chosen, hand_base_id], task._rigid_body_rot[chosen, hand_base_id]), -1)
                    obs_at[chosen], context_at[chosen] = obs["obs"][chosen], context(task, tracker)[chosen]
                    base_at[chosen], root_at[chosen] = base[chosen], task._humanoid_root_states[chosen]
                    motion_at[chosen], progress_at[chosen] = task.data_id[chosen], task.progress_buf[chosen]
                    eligibility_counts["assigned"] += len(chosen)
                age = tick - decision_tick
                action = apply_feedback_residual(base, arms, age, durations, terminal, delta, amplitudes)
                targets = task._action_to_pd_targets(action.clone()).clone()
                base_targets = task._action_to_pd_targets(base.clone()).clone()
                active_window = (arms >= 0) & (age < WINDOW) & ~terminal
                env_ids = active_window.nonzero(as_tuple=False).flatten()
                before_terminal = terminal.clone()
                _, _, done, _ = self.env_step(self.env, action)
                done = done.to(device).bool().reshape(-1)
                tracker.step(task._target_states[:, 2], contacts(task).bool().all(-1))
                after, after_normals = physical()
                if len(env_ids):
                    offsets = age[env_ids]
                    outcomes[env_ids, offsets] = after[env_ids]
                    normal_trajectory[env_ids, offsets] = after_normals[env_ids]
                    native_q[env_ids, offsets] = task._dof_pos[env_ids]
                    tip_trajectory[env_ids, offsets] = task._rigid_body_pos[env_ids][:, tip_ids]
                    hand_base_trajectory[env_ids, offsets] = torch.cat((task._rigid_body_pos[env_ids, hand_base_id], task._rigid_body_rot[env_ids, hand_base_id]), -1)
                    actual[env_ids, offsets], bases[env_ids, offsets] = action[env_ids], base[env_ids]
                    pd_targets[env_ids, offsets] = targets[env_ids]
                    pd_base_targets[env_ids, offsets] = base_targets[env_ids]
                    valid_steps[env_ids, offsets] = True
                newly_done = done & ~before_terminal
                for env_id in newly_done.nonzero(as_tuple=False).flatten().tolist():
                    summaries.append(dict(wave=wave, env_id=env_id, arm=int(arms[env_id]),
                        motion_id=int(task.data_id[env_id]), start_frame=int(start_at[env_id]),
                        decision_tick=int(decision_tick[env_id]), steps=tick+1,
                        stable_success=bool(tracker.stable[env_id]),
                        drop_after_success=bool(tracker.drop_after_success[env_id]),
                        max_hold_seconds=float(tracker.max_run[env_id])))
                terminal |= done
                previous = action.clone()
                if tick % 250 == 0:
                    print(json.dumps(dict(wave=wave, tick=tick, assigned=int((arms>=0).sum()),
                                          complete=int(terminal.sum()))), flush=True)
                if terminal.all():
                    break
            else:
                raise RuntimeError("wave budget ended before all native episodes terminated")
            selected = (arms >= 0).nonzero(as_tuple=False).flatten()
            values = dict(arm=arms, duration=durations, decision_tick=decision_tick, before=initial, history=history_at,
                          actor_obs=obs_at, context=context_at, base_action=base_at, hand_root=root_at,
                          motion_id=motion_at, progress=progress_at, start_frame=start_at,
                          trajectory=outcomes, actions=actual, base_actions=bases, valid_steps=valid_steps,
                          pd_targets=pd_targets, pd_base_targets=pd_base_targets,
                          rest_height=rest, episode_id=full_ids + wave*n)
            values["pre_hold_steps"] = hold_steps_at
            if capture_geometry:
                values.update(amplitude=amplitudes, before_surface_normals=normals_at, surface_normals=normal_trajectory)
            if per_finger:
                values.update(native_q=native_q, before_fingertip_positions=tips_at, fingertip_positions=tip_trajectory,
                              before_hand_base_pose=hand_base_at, hand_base_pose=hand_base_trajectory)
            packets.append({key: value[selected].cpu() for key, value in values.items()})
            print(json.dumps(dict(wave_completed=wave, eligible_counts=eligibility_counts,
                                  unique_region_envs=int(region_seen.sum()), unique_geometry_envs=int(geometry_seen.sum()),
                                  unique_headroom_envs=int(headroom_seen.sum()),
                                  trials=len(selected), complete_windows=int(valid_steps[selected].all(-1).sum()))), flush=True)
            # Checkpoint bounded collection after every whole wave.
            payload = {key: torch.cat([p[key] for p in packets]) for key in values}
            payload.update(schema="ref2dex.randomized_intervention.v1" if ARGS.durations == [CHUNK] else "ref2dex.randomized_intervention.v2", arm_names=arm_names,
                           delta=delta.cpu(), chunk=CHUNK if ARGS.durations == [CHUNK] else "per-trial",
                           duration_levels=ARGS.durations, window=WINDOW, control_dt=task.dt,
                           motion_names=list(task.motion_file), source_sha256=SOURCE_SHA)
            payload["decision_region"] = ARGS.decision_region
            if capture_geometry:
                payload.update(schema="ref2dex.randomized_intervention.v3", amplitude_levels=ARGS.amplitudes,
                    normal_contract="nearest sampled object-surface normal at measured hand body center; aggregate force projection, not paired contact")
            if per_finger:
                payload.update(schema="ref2dex.randomized_intervention.v4", intervention_set=ARGS.intervention_set,
                    finger_range_fraction=ARGS.finger_range_fraction, fingertip_names=tip_names,
                    contact_body_names=tuple(task.cfg["env"]["contactBodies"]),
                    native_q_units=("m",)*3+("rad",)*15,
                    measured_motion_contract="native q0:3 metres, q3:18 radians; actual tip world positions metres; actual hand-base pose xyz metres/xyzw quaternion")
            torch.save(payload, ARGS.run_dir / "interventions.pt")
            (ARGS.run_dir / "episode_summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
        result = dict(trials=len(payload["arm"]), episodes=len(summaries),
                      complete_windows=int(payload["valid_steps"].all(-1).sum()),
                      arm_counts=torch.bincount(payload["arm"], minlength=len(arm_names)).tolist(),
                      elapsed_seconds=time.monotonic()-started,
                      dataset_sha256=sha(ARGS.run_dir / "interventions.pt"))
        (ARGS.run_dir / "result.json").write_text(json.dumps(result, indent=2)+"\n")


def main():
    global ARGS
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--assignment-seed", type=int, required=True)
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--wall-seconds", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--decision-region", choices=("contact", "early-hold"), default="contact")
    parser.add_argument("--durations", nargs="+", type=int, default=[CHUNK])
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[1.])
    parser.add_argument("--intervention-set", choices=("synergy", "per-finger-range"), default="synergy")
    parser.add_argument("--finger-range-fraction", type=float, default=.05)
    ARGS, remaining = parser.parse_known_args()
    if ARGS.intervention_set == "per-finger-range":
        per_finger_residuals(ARGS.finger_range_fraction)  # validate before loading native player
        if ARGS.durations != [8] or ARGS.amplitudes != [1.]:
            raise ValueError("per-finger range comparison fixes K8 and amplitude1; range fraction defines dose")
    if ARGS.durations != sorted(set(ARGS.durations)) or any(k not in (4, 8, 16) for k in ARGS.durations):
        raise ValueError("durations must be a sorted unique subset of4/8/16")
    if ARGS.amplitudes != sorted(set(ARGS.amplitudes)) or any(a not in (1., 2., 4.) for a in ARGS.amplitudes):
        raise ValueError("amplitudes must be a sorted unique subset of1/2/4")
    if len(ARGS.amplitudes) > 1 and ARGS.durations != [8]:
        raise ValueError("amplitude comparison fixes K8")
    checkpoint = Path(remaining[remaining.index("--checkpoint")+1])
    if sha(checkpoint) != SOURCE_SHA:
        raise ValueError("must use pinned self-trained source_e260")
    ARGS.run_dir = ARGS.run_dir.resolve()
    ARGS.run_dir.mkdir(parents=True, exist_ok=False)
    paths = [SCRIPT, ROOT / "src/task/cm-interaction-oracle/src/intervention.py", checkpoint]
    if ARGS.amplitudes != [1.] or ARGS.intervention_set != "synergy":
        paths.extend(ROOT / path for path in (
            "src/task/CmResidual/physical_value_live.py",
            "src/task/CmResidual/physical_value_contract.py",
            "third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py",
            "third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py",
            "src/task/CmResidual/dexplore_cm_geometry.py",
            "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py",
            "third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf",
            "third_party/DExplore/dexplore/data/assets/mjcf/airplane.urdf",
            "third_party/DExplore/dexplore/data/assets/mjcf/objects/airplane/airplane.obj"))
    for flag in ("--cfg_env", "--cfg_train"):
        paths.append(Path(remaining[remaining.index(flag)+1]))
    motion_root = Path(remaining[remaining.index("--motion_file")+1])
    # Path.rglob does not descend through the canonical motion directory links.
    for directory in sorted(motion_root.iterdir()):
        motion = directory / "interaction_hand_inspire.pt"
        if motion.is_file():
            paths.append(motion.resolve())
    manifest = dict(run_status="STARTED", command=sys.argv, assignment_seed=ARGS.assignment_seed,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input_hashes={str(p): sha(p) for p in paths}, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        created_at=datetime.now(timezone.utc).isoformat(), waves=ARGS.waves, wall_seconds=ARGS.wall_seconds,
        torch_version=torch.__version__, chunk_contract="per-trial assigned duration feedback residual; future base actions not available at decision",
        duration_levels=ARGS.durations, amplitude_levels=ARGS.amplitudes,
        intervention_set=ARGS.intervention_set, finger_range_fraction=ARGS.finger_range_fraction)
    manifest["decision_region"] = ARGS.decision_region
    manifest["reset_contract"] = "full batch only; no assigned trial excluded after execution"
    def save():
        (ARGS.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    save()
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = InterventionPlayer
    try:
        original.main()
        if sha(checkpoint) != SOURCE_SHA:
            raise ValueError("source checkpoint changed during collection")
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
