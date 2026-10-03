#!/usr/bin/env python3
"""Replace only the direct-Q head in a frozen physical-value bundle."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--augmented-q", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.physical, map_location="cpu", weights_only=False)
    q = torch.load(args.augmented_q, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ref2dex.physical_value_models.v1":
        raise ValueError("physical checkpoint schema mismatch")
    if q.get("schema") != "ref2dex.cm_model_based_augmented_q.v1":
        raise ValueError("augmented Q schema mismatch")
    original = payload["direct_q"]
    if set(original) != set(q["state_dict"]):
        raise ValueError("augmented Q parameter names differ")
    payload["direct_q"] = q["state_dict"]
    payload["cm_model_based_augmentation"] = {
        "augmented_q_sha256": sha(args.augmented_q),
        "source_physical_sha256": sha(args.physical),
        "source_direct_q_state": {key: value.detach().cpu() for key, value in original.items()},
        "role": "critic-training synthetic transition target; no actor observation change",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print({"output": str(args.output.resolve()), "sha256": sha(args.output),
           "source_physical_sha256": sha(args.physical),
           "augmented_q_sha256": sha(args.augmented_q)})


if __name__ == "__main__":
    main()
