"""Probe action-specific future-contact Cm on unseen object identity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from src.task.CmResidual.contact_aware_cm import ContactAwareCm, RawContactAwareCm
from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask
from src.task.CmResidual.tools.probe_expert_crossobject_cm import (
    ROOT, calibrate, geometric_inputs,
)
from src.task.CmResidual.tools.probe_intervention_handflow import sha256
from src.task.CmResidual.tools.probe_randomized_geometric_cm import raw_inputs


BASE = ROOT / "outputs/Dexplore/agent_crossobject_train3_s179_e320"
TRAIN = BASE / "eval_s178_e320_full_train3"
TEST = BASE / "eval_s174_e320_full_apple_heldout"
EXPECTED = {"train": "28e3004781abfce8cb6d7af64c1561e89f5b9a57871cea7fd70428e743d490d4",
            "heldout": "3efeca7d935542df14276d12edbe0ed462fc341b1c052b92100731e970703b51"}
CHECKPOINT_SHA = "e442bf481ad2f02f03d0f7bc5085e4378ffbb2547baf0db25626afd6e761f390"
SPLIT_SHA = "35fffcb500f1f3db76fb8a59c940f5da0a113627bb0be9db0d68b0112fe7f516"
NUM_ENVS = 64
HORIZON = 20


def load_rows(directory: Path, partition: str, *, quota: int, seed: int) -> dict:
    manifest = json.loads((directory / "run_manifest.json").read_text())
    path = directory / "transitions.pt"
    expected_root = ROOT / "outputs/CmResidual/agent_crossobject_split_v1" / partition
    if (manifest.get("run_status") != "COMPLETED" or
            manifest.get("checkpoint_sha256") != CHECKPOINT_SHA or
            manifest.get("input_manifest_sha256") != SPLIT_SHA or
            Path(manifest.get("motion_root", "")).resolve() != expected_root.resolve() or
            manifest.get("transition_sha256") != EXPECTED[partition] or
            sha256(path) != EXPECTED[partition]):
        raise ValueError(f"cross-object rollout provenance drift: {directory}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError("transition schema drift")
    n = len(payload["q"])
    if n % NUM_ENVS or payload["q"].shape[1] != 18:
        raise ValueError("incomplete step-major rollout")
    first = first_episode_mask(payload["done"], NUM_ENVS).reshape(-1, NUM_ENVS)
    T = n // NUM_ENVS
    progress = payload["progress"].reshape(T, NUM_ENVS)
    valid = first[:-HORIZON] & first[HORIZON:] & (
        progress[HORIZON:] == progress[:-HORIZON] + HORIZON)
    contact = (payload["hand_contact"].reshape(T, NUM_ENVS).bool() &
               payload["object_contact"].reshape(T, NUM_ENVS).bool())
    future = torch.stack([contact[k:T-HORIZON+k] for k in range(1, HORIZON+1)])
    target = future.float().mean(0)
    eligible = torch.where(valid.reshape(-1))[0]
    data_id = payload["data_id"].reshape(T, NUM_ENVS)[:-HORIZON].reshape(-1)
    current_contact = contact[:-HORIZON].reshape(-1)
    num_objects = 3 if partition == "train" else 1
    generator = torch.Generator().manual_seed(seed)
    choices = []
    counts = {}
    for object_id in range(num_objects):
        for state in (False, True):
            candidates = eligible[(data_id[eligible] == object_id) &
                                  (current_contact[eligible] == state)]
            if len(candidates) < quota:
                raise ValueError(f"{partition} object {object_id} contact={state}: "
                                 f"only {len(candidates)} rows for quota {quota}")
            chosen = candidates[torch.randperm(len(candidates), generator=generator)[:quota]]
            choices.append(chosen)
            counts[f"{object_id}:{int(state)}"] = len(candidates)
    selected = torch.cat(choices)
    rows = {name: value[selected] for name, value in payload.items()
            if isinstance(value, torch.Tensor)}
    rows["motion_id"] = data_id[selected].long()
    rows["contact_now"] = current_contact[selected]
    rows["contact_target"] = target.reshape(-1)[selected].float()
    rows["env_id"] = selected % NUM_ENVS
    rows["target"] = torch.zeros(len(selected), 3)
    rows["category"] = torch.zeros(len(selected), dtype=torch.long)
    if not all(torch.isfinite(value.float()).all() for value in rows.values()):
        raise FloatingPointError("non-finite future-contact row")
    rows["available_counts"] = counts
    return rows


def model_input(rows: dict, names: list[str], setup: dict) -> dict:
    return {**geometric_inputs(rows, names, setup), "raw": raw_inputs(rows),
            "target": rows["contact_target"], "contact_now": rows["contact_now"]}


def fit(model, kind: str, data: dict, schedule: list[torch.Tensor]) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    value = float("nan")
    for ids in schedule:
        model.train()
        prediction = (model(data["raw"][ids])["contact_fraction"]
                      if kind == "raw" else
                      model(data["region"][ids], data["context"][ids])[
                          "contact_fraction"])
        loss = F.mse_loss(prediction, data["target"][ids])
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite future-contact Cm loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
    model.eval()
    return value


@torch.no_grad()
def evaluate(model, kind: str, data: dict) -> dict:
    prediction = (model(data["raw"])["contact_fraction"] if kind == "raw" else
                  model(data["region"], data["context"])["contact_fraction"])
    target = data["target"]
    result = {"score_std": float(prediction.std()),
              "target_mean": float(target.mean())}
    for name, mask in (("all", torch.ones_like(target, dtype=torch.bool)),
                       ("current_contact", data["contact_now"]),
                       ("no_current_contact", ~data["contact_now"])):
        result[name] = {"count": int(mask.sum()),
                        "rmse": float((prediction[mask] - target[mask]).square().mean().sqrt()),
                        "mae": float((prediction[mask] - target[mask]).abs().mean()),
                        "predicted_mean": float(prediction[mask].mean()),
                        "actual_mean": float(target[mask].mean())}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 500:
        parser.error("new output and 1..500 steps required")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = args.output / "run_manifest.json"
    manifest = {"run_status": "STARTED", "started_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip(),
                "input_sha256": EXPECTED, "checkpoint_sha256": CHECKPOINT_SHA,
                "split_sha256": SPLIT_SHA, "train_objects": ["airplane", "mug", "toothpaste"],
                "heldout_object": "apple", "steps": args.steps,
                "cpu_threads": 2, "gpu_count": 0, "wall_budget_minutes": 45,
                "output_budget_mb": 20,
                "stop_rule": "provenance drift, episode leakage, non-finite or wall budget"}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        torch.set_num_threads(2)
        torch.manual_seed(240980)
        train = load_rows(TRAIN, "train", quota=500, seed=240981)
        test = load_rows(TEST, "heldout", quota=500, seed=240982)
        setup = calibrate(train)
        train_input = model_input(train, ["airplane", "mug", "toothpaste"], setup)
        test_input = model_input(test, ["apple"], setup)
        raw_mean = train_input["raw"].mean(0)
        raw_std = train_input["raw"].std(0).clamp_min(.01)
        torch.manual_seed(240983)
        geometric = ContactAwareCm()
        initial = {key: value.detach().clone() for key, value in geometric.state_dict().items()}
        state_only = ContactAwareCm()
        state_only.load_state_dict(initial)
        raw = RawContactAwareCm(raw_mean, raw_std)
        state_train = dict(train_input)
        state_train["region"] = train_input["region"].clone()
        state_train["region"][..., 11:16] = 0
        state_test = dict(test_input)
        state_test["region"] = test_input["region"].clone()
        state_test["region"][..., 11:16] = 0
        generator = torch.Generator().manual_seed(240984)
        schedule = [torch.randint(len(train["q"]), (96,), generator=generator)
                    for _ in range(args.steps)]
        losses = {"geometric": fit(geometric, "geometric", train_input, schedule),
                  "state_only": fit(state_only, "geometric", state_train, schedule),
                  "raw_action": fit(raw, "raw", train_input, schedule)}
        metrics = {"geometric": evaluate(geometric, "geometric", test_input),
                   "state_only": evaluate(state_only, "geometric", state_test),
                   "raw_action": evaluate(raw, "raw", test_input)}
        active = {name: value["current_contact"]["rmse"]
                  for name, value in metrics.items()}
        gate = (active["geometric"] <= .9 * min(active["state_only"],
                                                active["raw_action"]) and
                metrics["geometric"]["score_std"] >= .02)
        report = {"schema": "ref2dex.crossobject_contact_future_cm_probe.v1",
                  "classification": "Probe", "train_count": len(train["q"]),
                  "heldout_count": len(test["q"]),
                  "train_available_counts": train["available_counts"],
                  "heldout_available_counts": test["available_counts"],
                  "losses": losses, "heldout": metrics,
                  "current_contact_rmse": active,
                  "continue_to_ppo_aux_probe": gate,
                  "elapsed_seconds": time.monotonic() - started}
        (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest.update(run_status="COMPLETED", report_sha256=sha256(args.output / "report.json"))
        print(json.dumps({"current_contact_rmse": active,
                          "score_std": {k: v["score_std"] for k, v in metrics.items()},
                          "continue_to_ppo_aux_probe": gate}, sort_keys=True), flush=True)
    except BaseException as error:
        manifest.update(run_status="FAILED", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
