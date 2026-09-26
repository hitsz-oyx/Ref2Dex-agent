"""Randomized, actually executed action interventions in DExplore.

Treatment is randomized among pre-contact environments at prespecified
global steps. This identifies a population treatment effect, not an
individual same-state counterfactual. Actor provenance is explicit; an
official checkpoint may only be used as a Cm data-collection diagnostic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - must precede torch
import joblib
import numpy as np
import torch

import evaluate_paired as pinned
from src.task.CmResidual.randomized_action import (
    FINGER_SYNERGY_INDICES, balanced_assignment, balanced_axis_assignment,
    balanced_three_arm_assignment, execute_finger_synergy_dose,
    execute_finger_primer_lift,
    execute_sustained_grip_lift,
    execute_signed_axis_dose, execute_sequence_axis_dose, execute_crossaxis_primer,
)
from src.task.CmResidual.randomized_source import validate as validate_object_split_source


original = pinned.original
BASE_PLAYER = original.EvalPlayer
CONFIG = None
OFFICIAL_DIAGNOSTIC_SHA256 = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"


def balanced_base_three_arm_assignment(mask, generator):
    """Assign eligible environments to -z, base, or +z arms."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("mask must be a one-dimensional boolean tensor")
    selected = mask.nonzero(as_tuple=False).reshape(-1)
    assignment = torch.zeros(len(mask), dtype=torch.int8, device=mask.device)
    if len(selected):
        labels = torch.tensor([-1, 0, 1], dtype=torch.int8).repeat(
            (len(selected) + 2) // 3)[:len(selected)]
        labels = labels[torch.randperm(len(labels), generator=generator)]
        assignment[selected] = labels.to(mask.device)
    return assignment


class ProbeDone(Exception):
    """Normal early stop after the last randomized intervention."""


class RandomizedPlayer(BASE_PLAYER):
    def __init__(self, config):
        super().__init__(config)
        if CONFIG is None:
            raise RuntimeError("randomized action configuration missing")
        self.probe_step = 0
        self.probe_records = []
        self.followup_pending = []
        self.sequence_pending = None
        self.macro_pending = None
        self.probe_error = None
        self.probe_generator = torch.Generator(device="cpu").manual_seed(CONFIG["assignment_seed"])
        self.policy_artifact = None

    def restore(self, filename):
        super().restore(filename)
        if CONFIG["policy_mode"] != "cm":
            return
        model_path = CONFIG["policy_model"]
        self.policy_artifact = joblib.load(model_path)
        required = {"state_scaler", "state_pca", "models"}
        if not required.issubset(self.policy_artifact):
            raise ValueError("Cm policy artifact is missing fitted preprocessing/model")
        if ("cm_aware", "binary") not in self.policy_artifact["models"]:
            raise ValueError("Cm policy artifact has no cm_aware binary head")

    @torch.no_grad()
    def _cm_assignment(self, task, action, valid):
        """Score +/- delta-z from the pre-action state with a frozen value head."""
        if self.policy_artifact is None:
            raise RuntimeError("Cm policy artifact was not loaded")
        q = task._dof_pos.detach().float().cpu()
        object_state = task._target_states.detach().float().cpu()
        q_relative = q.clone()
        q_relative[:, :3] -= object_state[:, :3]
        state = torch.cat((
            q_relative, task._dof_vel.detach().float().cpu(), object_state,
            action.detach().float().cpu(),
            task.progress_buf.detach().float().cpu().reshape(-1, 1) / 500.0,
            task.start_times.detach().float().cpu().reshape(-1, 1) / 500.0,
        ), dim=1).numpy()
        scaler = self.policy_artifact["state_scaler"]
        pca = self.policy_artifact["state_pca"]
        latent = pca.transform(scaler.transform(state))
        classifier = self.policy_artifact["models"][("cm_aware", "binary")]
        plus_features = torch.from_numpy(
            np.concatenate((latent, np.ones((len(latent), 1)), latent), axis=1))
        minus_features = plus_features.clone()
        minus_features[:, latent.shape[1]] = -1.0
        plus = classifier.predict_proba(plus_features.numpy())[:, 1]
        minus = classifier.predict_proba(minus_features.numpy())[:, 1]
        scores = torch.as_tensor(plus - minus, device=action.device, dtype=action.dtype)
        assignment = torch.where(scores >= 0,
                                  torch.ones_like(valid, dtype=torch.int8),
                                  -torch.ones_like(valid, dtype=torch.int8))
        assignment = torch.where(valid, assignment,
                                 torch.zeros_like(assignment, dtype=torch.int8))
        return assignment, torch.as_tensor(plus, device=action.device), \
            torch.as_tensor(minus, device=action.device)

    def env_step(self, env, action):
        self.probe_step += 1
        if CONFIG["sustained_grip_lift"]:
            return self._sustained_env_step(env, action)
        task = env.task
        selected_step = self.probe_step in CONFIG["steps"]
        if selected_step:
            pre_contact = ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                           (task._tar_contact_forces.norm(dim=-1) > .1))
            valid = pre_contact & (task.reset_buf.reshape(-1) == 0) & (task.progress_buf > 0)
            multiaxis = len(CONFIG["axes"]) > 1
            sequence = CONFIG["sequence_lengths"]
            crossaxis = CONFIG["crossaxis_primer"]
            finger = CONFIG["finger_synergy"]
            finger_primer = CONFIG["finger_primer_lift"]
            if CONFIG["three_arm_randomized"]:
                valid &= action[:, 2].abs() <= 1 - CONFIG["delta_z"]
                assignment = balanced_base_three_arm_assignment(
                    valid, self.probe_generator)
                policy_plus = policy_minus = None
            elif finger or finger_primer:
                valid &= (action[:, list(FINGER_SYNERGY_INDICES)].abs() <=
                          1 - CONFIG["delta_z"]).all(-1)
                if finger_primer:
                    valid &= action[:, 2].abs() <= 1 - CONFIG["second_delta"]
                assignment = balanced_assignment(valid, self.probe_generator)
            elif multiaxis or sequence or crossaxis:
                valid &= (action[:, CONFIG["axes"]].abs() <= 1 - CONFIG["delta_z"]).all(-1)
                assignment = (balanced_three_arm_assignment(valid, self.probe_generator)
                              if crossaxis else balanced_axis_assignment(
                                  valid, self.probe_generator,
                                  2 if sequence else len(CONFIG["axes"])))
            else:
                if CONFIG["policy_mode"] == "off":
                    assignment = torch.zeros_like(valid, dtype=torch.int8)
                    policy_plus = policy_minus = None
                elif CONFIG["policy_mode"] == "cm":
                    valid &= action[:, 2].abs() <= 1 - CONFIG["delta_z"]
                    assignment, policy_plus, policy_minus = self._cm_assignment(
                        task, action, valid)
                else:
                    assignment = balanced_assignment(valid, self.probe_generator)
                    policy_plus = policy_minus = None
            executed = action.detach().clone()
            if finger_primer:
                executed = execute_finger_primer_lift(
                    action, assignment, finger_delta=CONFIG["delta_z"],
                    lift_delta=CONFIG["second_delta"])
            elif finger:
                executed = execute_finger_synergy_dose(action, assignment,
                                                       CONFIG["delta_z"])
            elif crossaxis:
                executed = execute_crossaxis_primer(action, assignment, CONFIG["delta_z"])
            elif sequence:
                executed = execute_sequence_axis_dose(
                    action, assignment, CONFIG["delta_z"], CONFIG["axes"][0])
            elif multiaxis:
                executed = execute_signed_axis_dose(action, assignment,
                                                    tuple(CONFIG["axes"]), CONFIG["delta_z"])
            else:
                executed[:, 2] = (executed[:, 2] + CONFIG["delta_z"] * assignment).clamp(-1, 1)
            before = {
                "q": task._dof_pos.clone(),
                "dof_vel": task._dof_vel.clone(),
                "object_state": task._target_states.clone(),
                "env_id": torch.arange(len(action), device=action.device),
                "base_action": action.detach().clone(),
                "executed_action": executed.clone(),
                "assignment": assignment.clone(),
                "intervention_valid": valid.clone(),
                "pre_contact": pre_contact.clone(),
                "progress": task.progress_buf.clone(),
                "motion_id": task.data_id.clone(),
                "start_frame": task.start_times.clone(),
            }
            if CONFIG["policy_mode"] == "cm":
                before["policy_score_plus"] = policy_plus.clone()
                before["policy_score_minus"] = policy_minus.clone()
            result = BASE_PLAYER.env_step(self, env, executed)
            before.update(next_q=task._dof_pos.clone(),
                          next_object_state=task._target_states.clone(),
                          global_step=torch.full_like(task.progress_buf, self.probe_step))
            if CONFIG["followup_horizon"]:
                before["next_contact"] = self._contact(task).clone()
                before["followup_contact_count"] = torch.zeros_like(task.progress_buf)
                if finger or finger_primer:
                    before["followup_alive"] = torch.ones_like(pre_contact)
                self.followup_pending.append((self.probe_step, before))
            if sequence or crossaxis or finger_primer:
                self.sequence_pending = (self.probe_step, assignment, before)
            if not all(torch.isfinite(value).all() for value in before.values()):
                raise FloatingPointError("non-finite randomized physical transition")
            if not CONFIG["followup_horizon"]:
                self.probe_records.append({key: value.detach().cpu() for key, value in before.items()})
            print("REF2DEX_RANDOMIZED_STEP " + json.dumps({
                "step": self.probe_step, "selected": int(valid.sum()),
                "plus": int((assignment > 0).sum()),
                "minus": int((assignment < 0).sum()),
                "axis_counts": ({"finger_primer_lift" if finger_primer else "finger_synergy":
                                [int((assignment == 1).sum()),
                                 int((assignment == -1).sum())]}
                                if finger or finger_primer else {str(axis if not sequence else code):
                                [int((assignment == code).sum()),
                                            int((assignment == -code).sum())]
                                for code, axis in enumerate(
                                    (1, 2) if sequence else CONFIG["axes"], 1)}),
            }, sort_keys=True), flush=True)
        elif self.sequence_pending is not None and self.probe_step == self.sequence_pending[0] + 1:
            _, assignment, record = self.sequence_pending
            executed = (execute_finger_primer_lift(
                            action, assignment, finger_delta=CONFIG["delta_z"],
                            lift_delta=CONFIG["second_delta"], second=True)
                        if CONFIG["finger_primer_lift"] else
                        execute_crossaxis_primer(action, assignment, CONFIG["delta_z"],
                                                 second=True)
                        if CONFIG["crossaxis_primer"] else
                        execute_sequence_axis_dose(
                            action, assignment, CONFIG["delta_z"],
                            CONFIG["axes"][0], second=True))
            record["second_base_action"] = action.detach().clone()
            record["second_executed_action"] = executed.clone()
            result = BASE_PLAYER.env_step(self, env, executed)
            self.sequence_pending = None
        else:
            result = BASE_PLAYER.env_step(self, env, action)
        if CONFIG["followup_horizon"]:
            contact = self._contact(task)
            remaining = []
            for started, record in self.followup_pending:
                if CONFIG["finger_synergy"] or CONFIG["finger_primer_lift"]:
                    elapsed = self.probe_step - started + 1
                    record["followup_alive"] &= (
                        (task.reset_buf.reshape(-1) == 0) &
                        (task.progress_buf == record["progress"] + elapsed))
                    record["followup_contact_count"] += (
                        contact & record["followup_alive"]).long()
                else:
                    record["followup_contact_count"] += contact.long()
                if self.probe_step - started + 1 == CONFIG["followup_horizon"]:
                    record.update(
                        followup_object_state=task._target_states.clone(),
                        followup_contact=contact.clone(),
                        followup_progress=task.progress_buf.clone(),
                        followup_reset=task.reset_buf.reshape(-1).clone(),
                    )
                    self.probe_records.append({key: value.detach().cpu()
                                               for key, value in record.items()})
                else:
                    remaining.append((started, record))
            self.followup_pending = remaining
        if (self.probe_step >= CONFIG["stop_step"] and
                not CONFIG["record_final_outcome"]):
            raise ProbeDone
        return result

    def _sustained_env_step(self, env, action):
        task = env.task
        started = self.probe_step in CONFIG["steps"]
        if started:
            if self.macro_pending is not None:
                raise RuntimeError("overlapping sustained options")
            pre_contact = self._contact(task)
            valid = pre_contact & (task.reset_buf.reshape(-1) == 0) & (
                task.progress_buf > 0)
            assignment = balanced_assignment(valid, self.probe_generator)
            record = {
                "q": task._dof_pos.clone(), "dof_vel": task._dof_vel.clone(),
                "object_state": task._target_states.clone(),
                "env_id": torch.arange(len(action), device=action.device),
                "base_action": action.detach().clone(),
                "assignment": assignment.clone(),
                "pre_contact": pre_contact.clone(),
                "progress": task.progress_buf.clone(),
                "motion_id": task.data_id.clone(),
                "start_frame": task.start_times.clone(),
                "global_step": torch.full_like(task.progress_buf, self.probe_step),
                "followup_contact_count": torch.zeros_like(task.progress_buf),
                "followup_alive": torch.ones_like(pre_contact),
                "option_wrist_increment_sum": torch.zeros_like(task.progress_buf,
                                                                dtype=action.dtype),
                "option_finger_increment_sum": torch.zeros(
                    (len(action), len(FINGER_SYNERGY_INDICES)),
                    dtype=action.dtype, device=action.device),
                "option_steps": torch.zeros_like(task.progress_buf),
            }
            self.macro_pending = (self.probe_step, assignment, record)
            self.followup_pending.append((self.probe_step, record))
        if self.macro_pending is not None:
            option_start, assignment, record = self.macro_pending
            if self.probe_step - option_start < 10:
                executed = execute_sustained_grip_lift(
                    action, assignment, finger_delta=CONFIG["delta_z"],
                    lift_delta=CONFIG["second_delta"])
                difference = executed - action
                record["option_wrist_increment_sum"] += difference[:, 2]
                record["option_finger_increment_sum"] += difference[:,
                                                               list(FINGER_SYNERGY_INDICES)]
                record["option_steps"] += (assignment != 0).long()
                if started:
                    record["executed_action"] = executed.clone()
                result = BASE_PLAYER.env_step(self, env, executed)
                if started:
                    record["next_q"] = task._dof_pos.clone()
                    record["next_object_state"] = task._target_states.clone()
                if self.probe_step - option_start == 9:
                    self.macro_pending = None
            else:
                raise RuntimeError("sustained option lifetime exceeded")
        else:
            result = BASE_PLAYER.env_step(self, env, action)
        contact = self._contact(task)
        remaining = []
        for option_start, record in self.followup_pending:
            elapsed = self.probe_step - option_start + 1
            record["followup_alive"] &= ((task.reset_buf.reshape(-1) == 0) &
                                         (task.progress_buf == record["progress"] + elapsed))
            record["followup_contact_count"] += (
                contact & record["followup_alive"]).long()
            if elapsed == 20:
                record.update(followup_object_state=task._target_states.clone(),
                              followup_contact=contact.clone(),
                              followup_progress=task.progress_buf.clone(),
                              followup_reset=task.reset_buf.reshape(-1).clone())
                if not all(torch.isfinite(value.float()).all() for value in record.values()):
                    raise FloatingPointError("non-finite sustained option record")
                self.probe_records.append({key: value.detach().cpu()
                                           for key, value in record.items()})
            else:
                remaining.append((option_start, record))
        self.followup_pending = remaining
        if started:
            print("REF2DEX_SUSTAINED_STEP " + json.dumps({
                "step": self.probe_step, "selected": int(valid.sum()),
                "grip": int((assignment == 1).sum()),
                "lift_only": int((assignment == -1).sum()),
            }, sort_keys=True), flush=True)
        if (self.probe_step >= CONFIG["stop_step"] and
                not CONFIG["record_final_outcome"]):
            raise ProbeDone
        return result

    @staticmethod
    def _contact(task):
        return ((task._contact_forces[:, task._contact_body_ids].norm(dim=-1) > .1).any(-1) &
                (task._tar_contact_forces.norm(dim=-1) > .1))

    def run(self):
        status = "COMPLETED"
        try:
            return super().run()
        except ProbeDone:
            return None
        except BaseException as error:
            status = "FAILED"
            self.probe_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            if (CONFIG["record_final_outcome"] and status == "COMPLETED" and
                    self.probe_records):
                by_env = {int(row["env_id"]): row for row in self.episode_results}
                if len(by_env) < CONFIG["expected_envs"]:
                    raise RuntimeError(
                        "complete-episode outcome requested but not every environment finished")
                for record in self.probe_records:
                    env_ids = record["env_id"].reshape(-1).tolist()
                    try:
                        outcomes = [by_env[int(env_id)] for env_id in env_ids]
                    except KeyError as error:
                        raise RuntimeError(
                            f"missing final episode outcome for env {error.args[0]}") from error
                    record["final_lift_success"] = torch.tensor(
                        [bool(row["lift_success"]) for row in outcomes], dtype=torch.bool)
                    record["final_max_contact_lift_m"] = torch.tensor(
                        [float(row["max_contact_lift_m"]) for row in outcomes],
                        dtype=torch.float32)
                    record["final_contact_fraction"] = torch.tensor(
                        [float(row["hand_object_contact_fraction"]) for row in outcomes],
                        dtype=torch.float32)
                    record["final_episode_steps"] = torch.tensor(
                        [int(row["steps"]) for row in outcomes], dtype=torch.int64)
            output = CONFIG["output"]
            if self.probe_records:
                keys = self.probe_records[0]
                records = {key: torch.cat([item[key] for item in self.probe_records])
                           for key in keys}
            else:
                records = {}
            schema = ("ref2dex.sustained_grip_lift_h20.v1"
                      if CONFIG["sustained_grip_lift"] else
                      "ref2dex.randomized_finger_primer_lift_h10.v1"
                      if CONFIG["finger_primer_lift"] else
                      "ref2dex.randomized_finger_followup.v1"
                      if CONFIG["finger_synergy"] else
                      "ref2dex.crossaxis_primer_h10.v1"
                      if CONFIG["crossaxis_primer"] else
                      "ref2dex.randomized_sequence_h10.v1"
                      if CONFIG["sequence_lengths"] else
                      "ref2dex.randomized_multiaxis_followup.v1"
                      if len(CONFIG["axes"]) > 1 else
                      "ref2dex.randomized_action_followup.v1"
                      if CONFIG["followup_horizon"] else
                      "ref2dex.randomized_action_transitions.v1")
            torch.save({"schema": schema,
                        "run_status": status, "failure": self.probe_error,
                        "assignment_seed": CONFIG["assignment_seed"],
                        "delta_z_action": CONFIG["delta_z"],
                        "second_delta_action": CONFIG["second_delta"],
                        "finger_primer_lift": CONFIG["finger_primer_lift"],
                        "sustained_grip_lift": CONFIG["sustained_grip_lift"],
                        "finger_synergy": CONFIG["finger_synergy"],
                        "finger_indices": list(FINGER_SYNERGY_INDICES)
                        if (CONFIG["finger_synergy"] or CONFIG["finger_primer_lift"] or
                            CONFIG["sustained_grip_lift"]) else None,
                        "intervention_axes": CONFIG["axes"],
                        "sequence_lengths": CONFIG["sequence_lengths"],
                        "crossaxis_primer": CONFIG["crossaxis_primer"],
                        "followup_horizon": CONFIG["followup_horizon"],
                        "record_final_outcome": CONFIG["record_final_outcome"],
                        "policy_mode": CONFIG["policy_mode"],
                        "three_arm_randomized": CONFIG["three_arm_randomized"],
                        "records": records}, output)
            selected = records.get("assignment", torch.empty(0, dtype=torch.int8))
            summary = {
                "run_status": status, "steps_collected": len(self.probe_records),
                "rows": len(selected), "plus": int((selected > 0).sum()),
                "minus": int((selected < 0).sum()),
                "final_outcome_recorded": bool(CONFIG["record_final_outcome"] and
                                                status == "COMPLETED"),
                "policy_mode": CONFIG["policy_mode"],
                "three_arm_randomized": CONFIG["three_arm_randomized"],
                "zero": int((selected == 0).sum()),
                "axis_counts": ({"sustained_grip_lift":
                                 [int((selected == 1).sum()),
                                  int((selected == -1).sum())]}
                                if CONFIG["sustained_grip_lift"] else
                                {"finger_primer_lift" if CONFIG["finger_primer_lift"]
                                 else "finger_synergy":
                                 [int((selected == 1).sum()),
                                  int((selected == -1).sum())]}
                                if CONFIG["finger_synergy"] or CONFIG["finger_primer_lift"] else
                                {str(axis if not CONFIG["sequence_lengths"] else code):
                                [int((selected == code).sum()),
                                            int((selected == -code).sum())]
                                for code, axis in enumerate(
                                    (1, 2) if CONFIG["sequence_lengths"] else CONFIG["axes"], 1)}),
                "failure": self.probe_error,
            }
            output.with_suffix(".json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print("REF2DEX_RANDOMIZED_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


def main():
    global CONFIG
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--intervention-output", type=Path, required=True)
    parser.add_argument("--intervention-first", type=int, default=50)
    parser.add_argument("--intervention-last", type=int, default=150)
    parser.add_argument("--intervention-stride", type=int, default=10)
    parser.add_argument("--intervention-delta-z", "--intervention-delta",
                        dest="intervention_delta_z", type=float, default=.3)
    parser.add_argument("--intervention-axes", type=int, nargs="+", default=[2])
    parser.add_argument("--assignment-seed", type=int, default=20260924)
    parser.add_argument("--followup-horizon", type=int, default=0)
    parser.add_argument("--sequence-lengths", type=int, nargs=2,
                        help="randomize one wrist axis for exactly 1 or 2 steps")
    parser.add_argument("--cross-axis-primer", action="store_true",
                        help="randomize x primer before common z+ action")
    parser.add_argument("--finger-synergy", action="store_true",
                        help="signed dose on five independent finger-flexion commands")
    parser.add_argument("--finger-primer-lift", action="store_true",
                        help="randomized finger primer followed by common wrist-z lift")
    parser.add_argument("--sustained-grip-lift", action="store_true",
                        help="ten-step grip+lift versus lift-only option")
    parser.add_argument("--record-final-outcome", action="store_true",
                        help="continue first episodes and attach final lift labels")
    parser.add_argument("--policy-mode", choices=("random", "cm", "off"),
                        default="random",
                        help="randomized assignment, frozen Cm policy, or no action intervention")
    parser.add_argument("--policy-joblib", type=Path,
                        help="fitted post-contact value artifact for --policy-mode cm")
    parser.add_argument("--three-arm-randomized", action="store_true",
                        help="randomize eligible rows among -z, base, and +z")
    parser.add_argument("--second-delta", type=float, default=.1)
    parser.add_argument("--source-checkpoint-sha256")
    parser.add_argument("--source-motion-manifest", type=Path)
    parser.add_argument("--source-motion-manifest-sha256")
    parser.add_argument("--source-partition", choices=("train", "heldout"))
    parser.add_argument("--source-actor-role",
                        choices=("self_trained", "official_data_collector"),
                        default="self_trained")
    args, remaining = parser.parse_known_args()
    if (args.intervention_output.exists() or
            args.intervention_output.with_suffix(".json").exists() or
            not 1 <= args.intervention_first <= args.intervention_last <= 500 or
            not 1 <= args.intervention_stride <= 100 or
            not 0 < args.intervention_delta_z <= .5 or
            len(set(args.intervention_axes)) != len(args.intervention_axes) or
            not set(args.intervention_axes).issubset({0, 1, 2}) or
            (len(args.intervention_axes) == 1 and args.intervention_axes != [2] and
             args.sequence_lengths is None) or
            (len(args.intervention_axes) > 1 and not args.followup_horizon) or
            (args.cross_axis_primer and (
                args.sequence_lengths is not None or args.intervention_axes != [0, 2] or
                args.followup_horizon != 10 or args.intervention_stride < 10)) or
            (args.finger_synergy and (
                args.cross_axis_primer or args.sequence_lengths is not None or
                args.intervention_axes != [2] or args.followup_horizon != 10 or
                args.intervention_stride < 10)) or
            (args.finger_primer_lift and (
                args.finger_synergy or args.cross_axis_primer or
                args.sequence_lengths is not None or args.intervention_axes != [2] or
                args.followup_horizon != 10 or args.intervention_stride < 10 or
                not 0 < args.second_delta <= .5)) or
            (args.sustained_grip_lift and (
                args.finger_primer_lift or args.finger_synergy or
                args.cross_axis_primer or args.sequence_lengths is not None or
                args.intervention_axes != [2] or args.followup_horizon != 20 or
                args.intervention_stride < 20 or not 0 < args.second_delta <= .5 or
                args.source_actor_role != "self_trained")) or
            args.policy_mode == "cm" and args.policy_joblib is None or
            args.policy_mode != "cm" and args.policy_joblib is not None or
            args.policy_mode == "cm" and (
                args.sustained_grip_lift or args.finger_synergy or
                args.finger_primer_lift or args.cross_axis_primer or
                args.sequence_lengths is not None or args.intervention_axes != [2] or
                args.followup_horizon == 0) or
            args.three_arm_randomized and (
                args.policy_mode != "random" or args.sustained_grip_lift or
                args.finger_synergy or args.finger_primer_lift or
                args.cross_axis_primer or args.sequence_lengths is not None or
                args.intervention_axes != [2] or args.followup_horizon == 0) or
            (args.sequence_lengths is not None and (
                args.sequence_lengths != [1, 2] or args.intervention_axes not in ([0], [2]) or
                args.followup_horizon != 10 or args.intervention_stride < 10)) or
            (args.intervention_axes == [0] and args.sequence_lengths is None) or
            not 0 <= args.followup_horizon <= (20 if args.sustained_grip_lift else 10) or
            (args.followup_horizon and args.intervention_stride < args.followup_horizon) or
            args.intervention_last + max(args.followup_horizon - 1, 0) > 500):
        raise ValueError("invalid randomized intervention design or existing output")
    if args.policy_joblib is not None and not args.policy_joblib.is_file():
        raise FileNotFoundError(args.policy_joblib)
    checkpoint = Path(pinned.argument_value(remaining, "--checkpoint")).resolve()
    motion_root = Path(pinned.argument_value(remaining, "--motion_file")).resolve()
    source_overrides = (args.source_checkpoint_sha256,
                        args.source_motion_manifest,
                        args.source_motion_manifest_sha256,
                        args.source_partition)
    if any(value is not None for value in source_overrides):
        if not all(value is not None for value in source_overrides):
            raise ValueError("cross-object source overrides must be supplied together")
        source = validate_object_split_source(
            checkpoint=checkpoint, checkpoint_sha256=args.source_checkpoint_sha256,
            motion_root=motion_root, manifest_path=args.source_motion_manifest,
            manifest_sha256=args.source_motion_manifest_sha256,
            partition=args.source_partition)
    else:
        if (checkpoint != pinned.CHECKPOINT.resolve() or
                pinned.sha256(checkpoint) != pinned.CHECKPOINT_SHA256 or
                motion_root != pinned.MOTION_ROOT.resolve() or
                pinned.sha256(pinned.MOTION_MANIFEST) != pinned.MOTION_MANIFEST_SHA256):
            raise ValueError("pinned self-trained actor or motion source drift")
        source = {"checkpoint_sha256": pinned.CHECKPOINT_SHA256,
                  "motion_manifest_sha256": pinned.MOTION_MANIFEST_SHA256,
                  "partition": None, "objects": ["airplane"]}
    if (args.source_actor_role == "official_data_collector") != (
            source["checkpoint_sha256"] == OFFICIAL_DIAGNOSTIC_SHA256):
        raise ValueError("official diagnostic actor must have its explicit role and pinned SHA256")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        raise RuntimeError("one explicit physical CUDA_VISIBLE_DEVICES index required")
    memory = int(subprocess.check_output([
        "nvidia-smi", f"--id={visible}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip())
    if memory > 512:
        raise RuntimeError(f"GPU{visible} occupied: {memory} MiB")
    args.intervention_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.intervention_output.parent / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    steps = list(range(args.intervention_first, args.intervention_last + 1,
                       args.intervention_stride))
    schema = ("ref2dex.sustained_grip_lift_h20.v1"
              if args.sustained_grip_lift else
              "ref2dex.randomized_finger_primer_lift_h10.v1"
              if args.finger_primer_lift else
              "ref2dex.randomized_finger_followup.v1"
              if args.finger_synergy else
              "ref2dex.crossaxis_primer_h10.v1"
              if args.cross_axis_primer else
              "ref2dex.randomized_sequence_h10.v1"
              if args.sequence_lengths is not None else
              "ref2dex.randomized_multiaxis_followup.v1"
              if len(args.intervention_axes) > 1 else
              "ref2dex.randomized_action_followup.v1" if args.followup_horizon
              else "ref2dex.randomized_action_transitions.v1")
    manifest = {
        "run_status": "STARTED", "schema": schema,
        "run_id": args.intervention_output.parent.name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=pinned.ROOT, text=True).strip(),
        "physical_gpu": int(visible), "max_gpu_count": 1,
        "seed": int(pinned.argument_value(remaining, "--seed")),
        "intervention_steps": steps,
        "stop_step": steps[-1] + max(args.followup_horizon - 1, 0),
        "delta_z_action": args.intervention_delta_z,
        "second_delta_action": args.second_delta,
        "finger_primer_lift": args.finger_primer_lift,
        "sustained_grip_lift": args.sustained_grip_lift,
        "finger_synergy": args.finger_synergy,
        "finger_indices": list(FINGER_SYNERGY_INDICES)
        if args.finger_synergy or args.finger_primer_lift or args.sustained_grip_lift else None,
        "intervention_axes": args.intervention_axes,
        "sequence_lengths": args.sequence_lengths,
        "crossaxis_primer": args.cross_axis_primer,
        "followup_horizon": args.followup_horizon,
        "assignment_seed": args.assignment_seed,
        "checkpoint_sha256": source["checkpoint_sha256"],
        "motion_manifest_sha256": source["motion_manifest_sha256"],
        "source_partition": source["partition"],
        "source_objects": source["objects"],
        "source_actor_role": args.source_actor_role,
        "record_final_outcome": args.record_final_outcome,
        "policy_mode": args.policy_mode,
        "three_arm_randomized": args.three_arm_randomized,
        "policy_model_sha256": (pinned.sha256(args.policy_joblib)
                                 if args.policy_joblib is not None else None),
        "expected_envs": int(pinned.argument_value(remaining, "--num_envs")),
        "wall_budget_minutes": 30, "output_budget_mb": 100,
        "stop_rule": "input drift, GPU conflict, non-finite state or wall budget",
        "command": [sys.executable, *sys.argv],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    CONFIG = {"output": args.intervention_output.resolve(), "steps": set(steps),
              "stop_step": manifest["stop_step"], "delta_z": args.intervention_delta_z,
              "axes": args.intervention_axes,
              "sequence_lengths": args.sequence_lengths,
              "crossaxis_primer": args.cross_axis_primer,
              "finger_synergy": args.finger_synergy,
              "finger_primer_lift": args.finger_primer_lift,
              "sustained_grip_lift": args.sustained_grip_lift,
              "second_delta": args.second_delta,
              "followup_horizon": args.followup_horizon,
              "assignment_seed": args.assignment_seed,
              "record_final_outcome": args.record_final_outcome,
              "policy_mode": args.policy_mode,
              "three_arm_randomized": args.three_arm_randomized,
              "policy_model": args.policy_joblib.resolve()
              if args.policy_joblib is not None else None,
              "expected_envs": int(pinned.argument_value(remaining, "--num_envs"))}
    sys.argv = [sys.argv[0], *remaining]
    original.EvalPlayer = RandomizedPlayer
    try:
        original.main()
        manifest.update(run_status="COMPLETED",
                        transitions_sha256=pinned.sha256(args.intervention_output),
                        summary_sha256=pinned.sha256(args.intervention_output.with_suffix(".json")))
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
