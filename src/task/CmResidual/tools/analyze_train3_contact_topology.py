"""Test whether a five-link contact proxy tracks held-lift within train objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


SPLIT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"
CHECKPOINT_SHA = "e442bf481ad2f02f03d0f7bc5085e4378ffbb2547baf0db25626afd6e761f390"
OBJECTS = {0: "mug", 1: "toothpaste", 2: "airplane"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    manifest = json.loads((args.input_dir / "run_manifest.json").read_text())
    path = args.input_dir / "transitions.pt"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("checkpoint_sha256") != CHECKPOINT_SHA or
            manifest.get("input_manifest_sha256") != SPLIT_SHA or
            manifest.get("seed") != 178 or
            manifest.get("contact_topology") is not True or
            manifest.get("transition_sha256") != sha256(path)):
        raise ValueError("train3 topology provenance drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    force = payload.get("configured_link_force_norm")
    object_force = payload.get("object_contact_force_norm")
    if (payload.get("schema") != "ref2dex.cmlite_transition.v1" or
            not isinstance(force, torch.Tensor) or force.shape != (len(payload["q"]), 5) or
            not isinstance(object_force, torch.Tensor) or
            object_force.shape != (len(force), 1) or len(force) % 64):
        raise ValueError("missing five-link topology")
    if not torch.isfinite(force).all() or not torch.isfinite(object_force).all():
        raise FloatingPointError("non-finite force")
    T = len(force) // 64
    active = (force.reshape(T, 64, 5) > .1).sum(-1).numpy()
    object_contact = (object_force.reshape(T, 64) > .1).numpy()
    contact = (active > 0) & object_contact
    first = first_episode_mask(payload["done"], 64).reshape(T, 64).numpy()
    z = payload["object_state"].reshape(T, 64, -1)[:, :, 2].numpy()
    if not np.isfinite(z).all():
        raise FloatingPointError("non-finite object z")
    results = json.loads((args.input_dir / "results.json").read_text())
    episodes = {int(row["env_id"]): row for row in results["per_episode"]}
    if set(episodes) != set(range(64)):
        raise ValueError("incomplete episode IDs")
    rows = []
    for env_id in range(64):
        episode = episodes[env_id]
        motion_id = int(episode["motion_id"])
        if motion_id not in OBJECTS:
            raise ValueError(f"unexpected motion_id {motion_id}")
        mask = first[:, env_id] & contact[:, env_id]
        counts = active[mask, env_id]
        early_mask = mask & (z[:, env_id] - z[0, env_id] < .01)
        early_counts = active[early_mask, env_id][:30]
        rows.append({"env_id": env_id, "object": OBJECTS[motion_id],
                     "motion_id": motion_id, "success": bool(episode["lift_success"]),
                     "contact_frames": int(mask.sum()),
                     "fraction_ge3": float((counts >= 3).mean()) if len(counts) else 0.0,
                     "mean_active_links": float(counts.mean()) if len(counts) else 0.0,
                     "early_prelift_contact_frames": len(early_counts),
                     "early_prelift_fraction_ge3": float((early_counts >= 3).mean())
                     if len(early_counts) else 0.0})
    by_object = {}
    gate_count = 0
    for object_name in OBJECTS.values():
        group = [r for r in rows if r["object"] == object_name]
        success = np.array([r["fraction_ge3"] for r in group if r["success"]])
        failure = np.array([r["fraction_ge3"] for r in group if not r["success"]])
        if not len(success) or not len(failure):
            by_object[object_name] = {
                "success_episodes": len(success), "failure_episodes": len(failure),
                "success_minus_failure": None, "episode_bootstrap_95ci": None,
                "no_contact_episodes": sum(r["contact_frames"] == 0 for r in group),
                "gate_passed": False, "reason": "one outcome class absent"}
            continue
        difference = float(success.mean() - failure.mean())
        rng = np.random.default_rng(240924 + next(k for k, v in OBJECTS.items() if v == object_name))
        draws = np.array([rng.choice(success, len(success), replace=True).mean() -
                          rng.choice(failure, len(failure), replace=True).mean()
                          for _ in range(1000)])
        by_object[object_name] = {
            "success_episodes": len(success), "failure_episodes": len(failure),
            "success_mean_fraction_ge3": float(success.mean()),
            "failure_mean_fraction_ge3": float(failure.mean()),
            "success_minus_failure": difference,
            "episode_bootstrap_95ci": np.quantile(draws, [.025, .975]).tolist(),
            "no_contact_episodes": sum(r["contact_frames"] == 0 for r in group),
            "gate_passed": difference >= .2}
        gate_count += difference >= .2
    temporal_by_object = {}
    for object_name in OBJECTS.values():
        group = [r for r in rows if r["object"] == object_name]
        success = np.array([r["early_prelift_fraction_ge3"] for r in group if r["success"]])
        failure = np.array([r["early_prelift_fraction_ge3"] for r in group if not r["success"]])
        difference = float(success.mean() - failure.mean()) if len(success) and len(failure) else None
        temporal_by_object[object_name] = {
            "success_mean_fraction_ge3": float(success.mean()) if len(success) else None,
            "failure_mean_fraction_ge3": float(failure.mean()) if len(failure) else None,
            "success_minus_failure": difference,
            "gate_passed": difference is not None and difference >= .1,
            "success_mean_early_frames": float(np.mean([r["early_prelift_contact_frames"]
                for r in group if r["success"]])) if len(success) else None,
            "failure_mean_early_frames": float(np.mean([r["early_prelift_contact_frames"]
                for r in group if not r["success"]])) if len(failure) else None}
    temporal_gate = all(temporal_by_object[name]["gate_passed"]
                        for name in ("airplane", "mug"))
    report = {"schema": "ref2dex.train3_contact_topology_audit.v2",
              "classification": "Probe", "input_transition_sha256": sha256(path),
              "split_sha256": SPLIT_SHA, "checkpoint_sha256": CHECKPOINT_SHA,
              "force_threshold": .1, "by_object": by_object,
              "gate_objects_ge20pp": gate_count,
              "multilink_target_gate_passed": gate_count >= 2,
              "early_prelift_first30_by_object": temporal_by_object,
              "early_prelift_gate_passed": temporal_gate,
              "limits": ["Configured link forces may include hand-table contact.",
                         "This is an association, not a causal effect or Cm benefit."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"early_prelift_first30_by_object": temporal_by_object,
                      "early_prelift_gate_passed": temporal_gate}, sort_keys=True))


if __name__ == "__main__":
    main()
