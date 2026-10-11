#!/usr/bin/env python3
"""Plot a fixed dev record's index-tip displacement against both controls."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / "src"))
from oakink_wm.pointwam_data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", default="part2/fold_unfold_paper_basic/1")
    args = parser.parse_args()
    metrics = json.loads((args.run / "final_metrics.json").read_text())
    row = next(r for r in metrics["clips"] if r["episode"] == args.episode)
    if row["partition"] != "dev_candidate":
        raise ValueError("plot expects a dev record")
    source = Path(row["pack"])
    admission = json.loads((args.run / "admission.json").read_text())["sources"]
    identity = next(r for r in admission if r["pack"] == str(source))
    if sha256(source) != identity["sha256"]:
        raise ValueError("admitted pack changed")
    with np.load(source, allow_pickle=False) as data:
        target = data["hand_displacement_m"].reshape(30, 20, 3)
        valid = data["hand_target_valid"].reshape(30, 20)
        time = data["timestamps_s"][1:] - data["timestamps_s"][0]
    prediction_path = args.run / (args.episode.replace("/", "__") + "-" + source.parent.name + "-prediction.npz")
    with np.load(prediction_path, allow_pickle=False) as data:
        prediction = data["hand"]
    contract = json.loads((args.run / "contract.json").read_text())
    mean = np.asarray(contract["hand_mean_m"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for hand, name in enumerate(("Right", "Left")):
        point = hand * 10 + 7  # author semantic index-tip slot, fixed before scores
        for coord, label in enumerate(("X", "Y", "Z")):
            axis = axes[hand, coord]
            mask = valid[:, point]
            axis.plot(time, np.where(mask, target[:, point, coord] * 1000, np.nan), color="black", label="Offline teacher")
            axis.plot(time, prediction[:, point, coord] * 1000, color="#d24d36", label="Joint model")
            axis.plot(time, mean[:, coord] * 1000, color="#3979b5", linestyle="--", label="Fit-only mean")
            axis.axhline(0, color="gray", linestyle=":", label="Static")
            axis.set_title(f"{name} index tip / {label}")
            axis.set_ylabel("Displacement (mm)"); axis.grid(alpha=.2)
            if hand == 1:
                axis.set_xlabel("Future horizon (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Fixed dev example: {args.episode}, {source.parent.name}\nSmall architecture substitute; offline teacher labels")
    fig.tight_layout()
    destination = args.run / "dev-index-trajectories.png"
    if destination.exists():
        raise FileExistsError(destination)
    fig.savefig(destination, dpi=160)
    print(destination)


if __name__ == "__main__":
    main()
