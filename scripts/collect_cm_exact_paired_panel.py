#!/usr/bin/env python3
"""Collect a small same-state native candidate panel with exact restores."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_paired_evaluator_resolution import sha


def make_player(original, args, torch, gymtorch):
    from collect_contact_consequences import build_player

    Parent = build_player(original, args, torch, gymtorch)

    class ExactPanelPlayer(Parent):
        @torch.no_grad()
        def run(self):
            from src.task.CmResidual.executable_contact_options import (
                TableClearance, hold_target, obj_vertices,
            )
            from src.task.CmResidual.native_pd_selector import (
                FrozenNativePDSelector, native_pd_targets,
            )
            from src.task.CmResidual.orientation_anchored_options import orientation_anchored_action
            from src.task.CmResidual.paired_sim_step import paired_sim_step
            from src.task.CmResidual.paired_evaluation import fingerprint
            from src.task.CmResidual.weight_normalized_contact import weight_normalized_contacts

            started = time.monotonic()
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            task = self.env.task
            if task.num_envs != 96 or abs(task.dt - 1 / 30) > 1e-8:
                raise ValueError("native panel contract requires 96 environments at 30 Hz")
            if self.is_rnn:
                raise ValueError("exact panel requires stateless expert candidates")
            task._hybrid_init_prob = 0.0
            task._enable_early_termination = False
            task._adaptive_kappa_enabled = False
            ids = torch.arange(task.num_envs, device=self.device)
            obs = self.env_reset(ids)
            if self.get_batch_size(obs["obs"], 1) != task.num_envs:
                raise ValueError("native observation batch mismatch")
            selector = FrozenNativePDSelector(args.native_pd_checkpoint, self.device)
            if selector.fixed != 7:
                raise ValueError("native selector fixed Cup index changed")
            mesh_root = ROOT / "third_party/DExplore/dexplore/data/assets/mjcf"
            geometry = TableClearance(
                obj_vertices(mesh_root / "objects/airplane/airplane.obj", self.device),
                obj_vertices(mesh_root / "objects/table/table.obj", self.device),
                getattr(task, "ball_size", 1.),
            )
            body_ids = task._contact_body_ids
            mass = torch.tensor([
                task.gym.get_actor_rigid_body_properties(env, handle)[0].mass
                for env, handle in zip(task.envs, task._target_handles)
            ], device=self.device, dtype=torch.float32)
            gravity = abs(float(task.sim_params.gravity.z))
            history = torch.zeros(task.num_envs, 10, 69, device=self.device)
            previous = torch.zeros(task.num_envs, 18, device=self.device)
            contact_run = torch.zeros(task.num_envs, dtype=torch.long, device=self.device)
            done_seen = torch.zeros(task.num_envs, dtype=torch.bool, device=self.device)
            panel_seen = torch.zeros(task.num_envs, dtype=torch.bool, device=self.device)
            reset = ids[:0]
            rows = []
            source_hash = sha(args.native_pd_checkpoint)
            selector_before = fingerprint({mode: [m.state_dict() for m in models]
                                            for mode, models in selector.models.items()})
            experts_before = fingerprint([dict(model=m.state_dict(), rms=r.state_dict())
                                          for m, r in self.frozen_experts])

            def contact_bits():
                bits, ratio = weight_normalized_contacts(
                    task._contact_forces[:, body_ids], task._tar_contact_forces,
                    mass, gravity)
                return bits, ratio

            def snapshot():
                # Restore every task tensor after setting the simulator's root
                # and DOF buffers. This keeps progress/reset/task caches aligned
                # with the visible physics state.
                return {name: value.clone() for name, value in vars(task).items()
                        if isinstance(value, torch.Tensor)}

            def restore(saved):
                task._root_states.copy_(saved["_root_states"])
                task._dof_state.copy_(saved["_dof_state"])
                task.gym.set_actor_root_state_tensor(task.sim, gymtorch.unwrap_tensor(task._root_states))
                task.gym.set_dof_state_tensor(task.sim, gymtorch.unwrap_tensor(task._dof_state))
                task._refresh_sim_tensors()
                for name, value in saved.items():
                    current = getattr(task, name)
                    if current.shape != value.shape or current.dtype != value.dtype:
                        raise ValueError(f"task tensor changed during paired panel: {name}")
                    current.copy_(value)

            for tick in range(args.max_steps):
                if time.monotonic() - started > args.wall_seconds:
                    raise TimeoutError("exact paired panel wall budget exceeded")
                obs = self.env_reset(reset)
                state = torch.cat((task._dof_pos.clone(), task._dof_vel.clone(),
                                   task._target_states.clone()), -1)
                contact, ratio = contact_bits()
                history = torch.cat((history[:, 1:],
                                     torch.cat((state, contact.float(), previous), -1)[:, None]), 1)
                contact_run = torch.where(contact.all(-1), contact_run + 1, torch.zeros_like(contact_run))
                bank = self.candidates(obs)
                if bank.shape != (task.num_envs, 6, 18):
                    raise ValueError("six frozen expert candidates required")
                base = bank[:, 4]
                clearance = geometry.clearance(task._target_states, task._table_states)
                rest = task.hoi_refs[task.data_id, task.ref_index, 0, 108].clone()
                eligible = (
                    ~panel_seen & ~done_seen & (contact_run >= 3)
                    & (state[:, 38] - rest >= .03) & (clearance >= .002)
                    & (tick >= 10) & (tick <= args.max_steps - 2)
                    & (task.max_episode_length[task.data_id] - task.progress_buf > 3)
                )
                chosen = eligible.nonzero().flatten()
                remaining = args.max_states - len(rows)
                if len(chosen) > remaining:
                    chosen = chosen[:remaining]
                if len(chosen):
                    anchor = hold_target(task._dof_pos[chosen], task._pd_action_offset,
                                         task._pd_action_scale)
                    base_hold = orientation_anchored_action(
                        bank[chosen, 4], anchor, task._dof_pos[chosen],
                        task._pd_action_offset, task._pd_action_scale)
                    cup_hold = orientation_anchored_action(
                        bank[chosen, 1], anchor, task._dof_pos[chosen],
                        task._pd_action_offset, task._pd_action_scale)
                    raw = torch.cat((bank[chosen], base_hold[:, None], cup_hold[:, None]), 1)
                    native = native_pd_targets(
                        raw, task._dof_pos[chosen, None].expand(-1, 8, -1),
                        task._pd_action_offset, task._pd_action_scale)
                    cm_choice, diagnostic = selector.choose(
                        "cm", history[chosen], obs["obs"][chosen], rest[chosen], native,
                        task._contact_forces[chosen][:, body_ids], task._tar_contact_forces[chosen],
                        mass[chosen], gravity, clearance[chosen])
                    saved = snapshot()
                    fixed = torch.zeros_like(base)
                    fixed[:] = base
                    fixed[chosen] = cup_hold
                    effects = torch.zeros(len(chosen), 8, device=self.device)
                    repeat_object = torch.zeros_like(effects)
                    repeat_joint = torch.zeros_like(effects)
                    valid = torch.ones_like(effects, dtype=torch.bool)
                    pair_errors = [None] * 8
                    for candidate in range(8):
                        alternate = fixed.clone()
                        alternate[chosen] = raw[:, candidate]
                        if torch.allclose(alternate[chosen], fixed[chosen], atol=1e-7, rtol=0):
                            # Candidate 7 is exactly Cup and is the zero baseline.
                            valid[:, candidate] = True
                            continue
                        try:
                            paired = paired_sim_step(
                                task, fixed, alternate, gymtorch.unwrap_tensor,
                                restore_tolerance=1e-6,
                                repeat_object_tolerance_m=5e-5,
                                repeat_joint_tolerance=1e-4,
                            )
                        except BaseException as error:
                            valid[:, candidate] = False
                            pair_errors[candidate] = repr(error)
                            continue
                        effects[:, candidate] = (
                            paired["alternate_next_object_state"][chosen, 2]
                            - paired["base_next_object_state"][chosen, 2]
                        ) * 1000
                        repeat_object[:, candidate] = paired["same_action_repeat_object_mm"][chosen]
                        repeat_joint[:, candidate] = paired["same_action_repeat_joint_max"][chosen]
                    restore(saved)
                    for local, env_id in enumerate(chosen.tolist()):
                        rows.append(dict(
                            env_id=int(env_id), motion_id=int(task.data_id[env_id]),
                            start_frame=int(task.start_times[env_id]), trigger=int(tick),
                            state=state[env_id].cpu(), history=history[env_id].cpu(),
                            rest_z=float(rest[env_id]), initial_clearance=float(clearance[env_id]),
                            candidate_actions=raw[local].cpu(), candidate_pd_targets=native[local].cpu(),
                            cm_choice=int(cm_choice[local]),
                            cm_diagnostics={key: diagnostic[key][local].cpu() for key in (
                                "score_mm", "relative_std_mm", "retention", "release")},
                            cm_ood=bool(diagnostic["ood"][local]),
                            cm_candidate_ood=diagnostic["candidate_ood"][local].cpu(),
                            actual_effect_z_mm=effects[local].cpu(),
                            repeat_object_mm=repeat_object[local].cpu(),
                            repeat_joint_max=repeat_joint[local].cpu(),
                            valid=valid[local].cpu(),
                            pair_errors=pair_errors,
                        ))
                    panel_seen[chosen] = True
                    if len(rows) >= args.max_states:
                        break
                # Advance all unpaired environments by the frozen base action.
                action = base
                _, _, done, _ = self.env_step(self.env, action)
                done = done.bool().reshape(-1)
                done_seen |= done
                reset = done.nonzero().flatten()
                previous = action.clone()
                if tick % 50 == 0:
                    print(json.dumps(dict(tick=tick, panels=len(rows))), flush=True)
            if len(rows) < args.min_states:
                raise ValueError(f"paired panel support {len(rows)} < {args.min_states}")
            if selector_before != fingerprint({mode: [m.state_dict() for m in models]
                                                for mode, models in selector.models.items()}):
                raise ValueError("Cm selector changed during paired panel")
            if experts_before != fingerprint([dict(model=m.state_dict(), rms=r.state_dict())
                                              for m, r in self.frozen_experts]):
                raise ValueError("frozen expert changed during paired panel")
            payload = dict(
                schema="ref2dex.cm_exact_paired_panel.v1", experiment_id=args.experiment_id,
                rows=rows, candidate_names=["expert0", "expert1", "expert2", "expert3",
                                            "expert4", "expert5", "base_hold", "fixed_cup"],
                fixed_index=7, checkpoint_sha256=source_hash, frozen_cm=True,
                frozen_experts=True, optimizer_used=False,
                paired_contract="same hot task state; base repeat; root/DOF restore; one native tick",
                utility_contract="one-step object z effect relative to fixed Cup; ten-step prefix stored only as engineering check",
            )
            torch.save(payload, args.output / "records.pt")
            result = dict(
                run_status="COMPLETED", rows=len(rows), min_states=args.min_states,
                checkpoint_sha256=source_hash,
                valid_rows=int(sum(int(r["valid"].all()) for r in rows)),
                max_repeat_object_mm=max(float(r["repeat_object_mm"].max()) for r in rows),
                max_repeat_joint=float(max(float(r["repeat_joint_max"].max()) for r in rows)),
                elapsed_seconds=time.monotonic() - started,
                record_sha256=sha(args.output / "records.pt"),
            )
            (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)

    return ExactPanelPlayer


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-dir", dest="output", type=Path, required=True)
    parser.add_argument("--native-pd-checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-id", default="P-20261003-cm-exact-paired-panel")
    parser.add_argument("--max-states", type=int, default=24)
    parser.add_argument("--min-states", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--wall-seconds", type=int, default=300)
    args, remaining = parser.parse_known_args()
    base = (ROOT / "src/task/CmResidual/research/contact_consequence/output").resolve()
    if base not in args.output.resolve().parents or args.output.exists():
        raise ValueError("output must be a new owned directory")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(run_status="STARTED", pid=os.getpid(), command=sys.argv,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), phases=[])
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        sys.path.insert(0, str(ROOT / "third_party/DExplore/dexplore"))
        from isaacgym import gymtorch
        import torch
        import evaluate as original
        original.EvalPlayer = make_player(original, args, torch, gymtorch)
        sys.argv = [sys.argv[0], *remaining]
        original.main()
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", error=repr(error))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
