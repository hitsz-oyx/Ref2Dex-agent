"""Compare matched-seed first-episode apple grasp phases between actors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.probe_intervention_handflow import sha256


ROOT = Path(__file__).resolve().parents[4]
SELF = ROOT / "outputs/Dexplore/agent_crossobject_train3_s179_e320/eval_s174_e320_full_apple_heldout"
SELF_SHA = "3efeca7d935542df14276d12edbe0ed462fc341b1c052b92100731e970703b51"
OFFICIAL_SHA = "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553"
SPLIT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"


def load(directory: Path, *, role: str) -> dict:
    manifest = json.loads((directory / "run_manifest.json").read_text())
    transition = directory / "transitions.pt"
    results = json.loads((directory / "results.json").read_text())
    digest = SELF_SHA if role == "self_trained" else manifest.get("transition_sha256")
    checkpoint = ("e442bf481ad2f02f03d0f7bc5085e4378ffbb2547baf0db25626afd6e761f390"
                  if role == "self_trained" else OFFICIAL_SHA)
    split_key = "input_manifest_sha256" if role == "self_trained" else "split_sha256"
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("checkpoint_sha256") != checkpoint or
            manifest.get(split_key) != SPLIT_SHA or
            manifest.get("seed") != 174 or
            manifest.get("transition_sha256") != digest or
            sha256(transition) != digest or
            results["summary"]["num_episodes"] != 64):
        raise ValueError(f"{role} apple rollout provenance drift")
    payload = torch.load(transition, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError("apple transition schema drift")
    mask = first_episode_mask(payload["done"], 64).reshape(-1, 64).numpy()
    T = len(payload["q"]) // 64
    contact = (payload["hand_contact"].reshape(T, 64).bool() &
               payload["object_contact"].reshape(T, 64).bool()).numpy()
    z = payload["object_state"].reshape(T, 64, 13)[:, :, 2].numpy()
    episodes = {entry["env_id"]: entry for entry in results["per_episode"]}
    if set(episodes) != set(range(64)):
        raise ValueError("expected one first episode per environment")
    by_env = {}
    for env_id in range(64):
        times = np.flatnonzero(mask[:, env_id])
        if not len(times):
            raise ValueError("empty first episode")
        c = contact[times, env_id]
        lift = z[times, env_id] - z[times[0], env_id]
        longest = 0
        current = 0
        for value in c:
            current = current + 1 if value else 0
            longest = max(longest, current)
        first_contact = int(np.flatnonzero(c)[0]) if c.any() else None
        episode = episodes[env_id]
        by_env[str(env_id)] = {
            "start_frame": episode["start_frame"],
            "motion_id": episode["motion_id"],
            "lift_success": bool(episode["lift_success"]),
            "first_contact_step": first_contact,
            "longest_contact_run": longest,
            "contact_fraction": float(c.mean()),
            "peak_contact_lift_mm": float(np.max(np.where(c, lift * 1000, 0))),
            "num_exported_first_steps": len(times),
        }
    return {"by_env": by_env, "summary": results["summary"],
            "transition_sha256": digest}


def summarize(source: dict) -> dict:
    values = list(source["by_env"].values())
    failure = [row for row in values if not row["lift_success"]]
    return {"successes": sum(row["lift_success"] for row in values),
            "mean_contact_fraction": float(np.mean([row["contact_fraction"] for row in values])),
            "median_first_contact_step": float(np.median([
                row["first_contact_step"] for row in values
                if row["first_contact_step"] is not None])),
            "median_longest_contact_run": float(np.median([
                row["longest_contact_run"] for row in values])),
            "failed_never_contact": sum(row["first_contact_step"] is None for row in failure),
            "failed_contact_run_lt20": sum(
                row["first_contact_step"] is not None and
                row["longest_contact_run"] < 20 for row in failure),
            "failed_contact_run_ge20_no_lift": sum(
                row["longest_contact_run"] >= 20 for row in failure),
            "mean_peak_contact_lift_mm": float(np.mean([
                row["peak_contact_lift_mm"] for row in values]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    own = load(SELF, role="self_trained")
    expert = load(args.official, role="official")
    mismatch = [env for env in range(64)
                if (own["by_env"][str(env)]["motion_id"],
                    own["by_env"][str(env)]["start_frame"]) !=
                   (expert["by_env"][str(env)]["motion_id"],
                    expert["by_env"][str(env)]["start_frame"])]
    if mismatch:
        raise ValueError(f"initial apple episode mismatch in environments: {mismatch}")
    report = {"schema": "ref2dex.apple_failure_phase_audit.v1",
              "classification": "Probe", "initial_states_matched": True,
              "self_trained": summarize(own), "official": summarize(expert),
              "source_sha256": {"self_trained": own["transition_sha256"],
                                "official": expert["transition_sha256"]},
              "per_environment": {str(env): {
                  "self_trained": own["by_env"][str(env)],
                  "official": expert["by_env"][str(env)]}
                                  for env in range(64)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"initial_states_matched": True,
                      "self_trained": report["self_trained"],
                      "official": report["official"]}, sort_keys=True))


if __name__ == "__main__":
    main()
