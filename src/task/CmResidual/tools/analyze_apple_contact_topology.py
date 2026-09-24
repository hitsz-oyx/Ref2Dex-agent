"""Compare configured hand-link force topology on matched apple episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


ROOT = Path(__file__).resolve().parents[4]
SPLIT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"
CHECKPOINT = {"self_trained": "e442bf481ad2f02f03d0f7bc5085e4378ffbb2547baf0db25626afd6e761f390",
              "official": "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"}
LINKS = ["index_intermediate", "middle_intermediate", "pinky_intermediate",
         "ring_intermediate", "thumb_distal"]


def load(directory: Path, role: str) -> dict:
    manifest = json.loads((directory / "run_manifest.json").read_text())
    path = directory / "transitions.pt"
    split_key = "input_manifest_sha256" if role == "self_trained" else "split_sha256"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("checkpoint_sha256") != CHECKPOINT[role] or
            manifest.get(split_key) != SPLIT_SHA or
            manifest.get("seed") != 174 or
            manifest.get("contact_topology") is not True or
            manifest.get("transition_sha256") != sha256(path)):
        raise ValueError(f"{role} topology export provenance drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    force = payload.get("configured_link_force_norm")
    object_force = payload.get("object_contact_force_norm")
    if (payload.get("schema") != "ref2dex.cmlite_transition.v1" or
            not isinstance(force, torch.Tensor) or force.shape != (len(payload["q"]), 5) or
            not isinstance(object_force, torch.Tensor) or
            object_force.shape != (len(force), 1)):
        raise ValueError("missing five-link force topology")
    if not torch.isfinite(force).all() or not torch.isfinite(object_force).all():
        raise FloatingPointError("non-finite contact forces")
    first = first_episode_mask(payload["done"], 64).reshape(-1, 64).numpy()
    T = len(force) // 64
    counts = (force.reshape(T, 64, 5) > .1).sum(-1).numpy()
    object_contact = (object_force.reshape(T, 64) > .1).numpy()
    contact = (counts > 0) & object_contact
    results = json.loads((directory / "results.json").read_text())
    episodes = {row["env_id"]: row for row in results["per_episode"]}
    if set(episodes) != set(range(64)):
        raise ValueError("incomplete episode IDs")
    by_env = {}
    for env_id in range(64):
        mask = first[:, env_id] & contact[:, env_id]
        if not mask.any():
            raise ValueError(f"{role} env {env_id} has no hand/object contact")
        active = counts[mask, env_id]
        by_env[env_id] = {"start_frame": episodes[env_id]["start_frame"],
                          "motion_id": episodes[env_id]["motion_id"],
                          "success": bool(episodes[env_id]["lift_success"]),
                          "contact_frames": int(mask.sum()),
                          "mean_active_links": float(active.mean()),
                          "fraction_ge2": float((active >= 2).mean()),
                          "fraction_ge3": float((active >= 3).mean())}
    return {"by_env": by_env, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-trained", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    own, expert = load(args.self_trained, "self_trained"), load(args.official, "official")
    mismatched = [i for i in range(64)
                  if (own["by_env"][i]["motion_id"], own["by_env"][i]["start_frame"]) !=
                     (expert["by_env"][i]["motion_id"], expert["by_env"][i]["start_frame"])]
    if mismatched:
        raise ValueError(f"initial-state mismatch: {mismatched}")
    metrics = {}
    for key in ("mean_active_links", "fraction_ge2", "fraction_ge3"):
        x = np.array([own["by_env"][i][key] for i in range(64)])
        y = np.array([expert["by_env"][i][key] for i in range(64)])
        difference = y - x
        generator = np.random.default_rng(240995 + len(metrics))
        draw = np.array([difference[generator.integers(0, 64, 64)].mean()
                         for _ in range(1000)])
        metrics[key] = {"self_trained": float(x.mean()),
                        "official": float(y.mean()),
                        "official_minus_self": float(difference.mean()),
                        "paired_env_bootstrap_95ci": np.quantile(draw, [.025, .975]).tolist()}
    gate = metrics["fraction_ge3"]["official_minus_self"] >= .2
    report = {"schema": "ref2dex.apple_contact_topology_audit.v1",
              "classification": "Probe", "configured_links": LINKS,
              "threshold_force_norm": .1, "initial_states_matched": True,
              "source_sha256": {"self_trained": own["sha256"],
                                "official": expert["sha256"]},
              "metrics": metrics, "multilink_gap_ge20pp": gate,
              "limits": ["Configured hand-link forces may include hand-table contact.",
                         "This is a grasp-topology proxy, not object-specific finger contact or Cm utility."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"metrics": metrics, "multilink_gap_ge20pp": gate}, sort_keys=True))


if __name__ == "__main__":
    main()
