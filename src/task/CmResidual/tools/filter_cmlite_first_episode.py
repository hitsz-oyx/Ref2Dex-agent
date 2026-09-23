#!/usr/bin/env python3
"""Materialize first-episode, nonterminal CmLite transitions with provenance."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import torch

from src.task.CmResidual.tools.analyze_cmlite_on_policy import first_episode_mask, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    args = parser.parse_args()
    manifest = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest.exists():
        raise FileExistsError("filtered output and manifest must both be new")
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    if source.get("schema") != "ref2dex.cmlite_transition.v1":
        raise ValueError("transition schema mismatch")
    mask = first_episode_mask(source["done"], args.num_envs)
    filtered = {key: value[mask] for key, value in source.items()
                if isinstance(value, torch.Tensor)}
    filtered["schema"] = source["schema"]
    filtered["selection"] = "first_episode_nonterminal_per_environment"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(filtered, args.output)
    record = {
        "run_status": "COMPLETED",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()), "output_sha256": sha256(args.output),
        "num_envs": args.num_envs, "source_samples": int(source["done"].shape[0]),
        "filtered_samples": int(mask.sum()),
        "selection": filtered["selection"],
    }
    manifest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
