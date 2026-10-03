#!/usr/bin/env python3
"""Add auditable temporal interaction deltas to an assembled Gate 1 dataset.

The base v2 interaction already uses contemporaneous object-frame geometry,
relative velocity, forces and contact masks.  This offline transform adds only
quantities reconstructible from that tensor: relative acceleration, force
increments, and contact on/off increments.  The first future slot is zeroed
because the preceding contact/force sample is not part of the assembled
interaction sequence.  No target, split key or row order is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


# Five contact bodies are part of the collector contract.  These offsets are
# kept explicit so a malformed base tensor fails loudly rather than silently
# receiving an incorrect augmentation.
BODY_COUNT = 5
BASE_DIM = 80
REL_POS = slice(0, BODY_COUNT * 3)
REL_QUAT = slice(REL_POS.stop, REL_POS.stop + BODY_COUNT * 4)
REL_VEL = slice(REL_QUAT.stop, REL_QUAT.stop + BODY_COUNT * 3)
HAND_FORCE = slice(REL_VEL.stop, REL_VEL.stop + BODY_COUNT * 3)
OBJECT_FORCE = slice(HAND_FORCE.stop, HAND_FORCE.stop + 3)
HAND_NORM = slice(OBJECT_FORCE.stop, OBJECT_FORCE.stop + BODY_COUNT)
OBJECT_NORM = slice(HAND_NORM.stop, HAND_NORM.stop + 1)
HAND_CONTACT = slice(OBJECT_NORM.stop, OBJECT_NORM.stop + BODY_COUNT)
OBJECT_CONTACT = slice(HAND_CONTACT.stop, HAND_CONTACT.stop + 1)


def augment(dataset: dict, control_dt: float | None = None) -> dict:
    interaction = dataset.get("interaction")
    if not isinstance(interaction, torch.Tensor) or interaction.ndim != 3:
        raise ValueError("dataset.interaction must have shape [N, H, D]")
    if interaction.shape[-1] != BASE_DIM:
        raise ValueError(f"expected base interaction dim {BASE_DIM}, got {interaction.shape[-1]}")
    if interaction.shape[1] < 1:
        raise ValueError("interaction horizon must be positive")
    metadata = dataset.get("metadata", {})
    runs = metadata.get("runs", []) if isinstance(metadata, dict) else []
    run_dts = {float(run["control_dt"]) for run in runs if "control_dt" in run}
    if control_dt is None:
        if len(run_dts) != 1:
            raise ValueError("provide --control-dt when runs do not share one control_dt")
        control_dt = next(iter(run_dts))
    if not (control_dt > 0):
        raise ValueError("control_dt must be positive")

    rel_velocity = interaction[..., REL_VEL]
    hand_force = interaction[..., HAND_FORCE]
    object_force = interaction[..., OBJECT_FORCE]
    hand_contact = interaction[..., HAND_CONTACT]
    object_contact = interaction[..., OBJECT_CONTACT]

    # The first future slot has no preceding future interaction sample.  Keep
    # it zero rather than treating the current frame as an unrecorded label.
    relative_acceleration = torch.zeros_like(rel_velocity)
    hand_force_delta = torch.zeros_like(hand_force)
    object_force_delta = torch.zeros_like(object_force)
    hand_contact_delta = torch.zeros_like(hand_contact)
    object_contact_delta = torch.zeros_like(object_contact)
    if interaction.shape[1] > 1:
        relative_acceleration[:, 1:] = (
            rel_velocity[:, 1:] - rel_velocity[:, :-1]
        ) / control_dt
        hand_force_delta[:, 1:] = hand_force[:, 1:] - hand_force[:, :-1]
        object_force_delta[:, 1:] = object_force[:, 1:] - object_force[:, :-1]
        hand_contact_delta[:, 1:] = hand_contact[:, 1:] - hand_contact[:, :-1]
        object_contact_delta[:, 1:] = object_contact[:, 1:] - object_contact[:, :-1]

    augmented = torch.cat((
        interaction,
        relative_acceleration,
        hand_force_delta,
        object_force_delta,
        hand_contact_delta,
        object_contact_delta,
    ), dim=-1)
    out = dict(dataset)
    out["interaction"] = augmented
    out_metadata = dict(metadata)
    out_metadata["interaction_layout"] = (
        str(metadata.get("interaction_layout", "base interaction"))
        + " + [relative_acceleration, hand_force_delta, object_force_delta, "
          "hand_contact_delta, object_contact_delta]"
    )
    out_metadata["interaction_augmentation"] = "temporal_deltas_from_base_v2"
    out_metadata["interaction_augmentation_control_dt"] = float(control_dt)
    out_metadata["interaction_augmentation_first_slot"] = "zero (no preceding future sample)"
    out["metadata"] = out_metadata
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control-dt", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = torch.load(args.input, map_location="cpu", weights_only=False)
    output = augment(dataset, args.control_dt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    report = {
        "schema": "ref2dex.gate1_interaction_augmentation.v1",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "rows": int(output["interaction"].shape[0]),
        "horizon": int(output["interaction"].shape[1]),
        "base_interaction_dim": BASE_DIM,
        "augmented_interaction_dim": int(output["interaction"].shape[-1]),
        "control_dt": output["metadata"]["interaction_augmentation_control_dt"],
        "first_slot_zeroed": True,
        "target_unchanged": torch.equal(output["return_to_go"], dataset["return_to_go"]),
        "episode_key_unchanged": torch.equal(output["episode_id"], dataset["episode_id"]),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
