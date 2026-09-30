"""Fixed HF08 evaluation contract and gate, independent of scheduling."""
import hashlib
import json
from pathlib import Path

ARMS = ("plain_off", "direct_q", "cm_value")
TRAINING_SEEDS = (286, 287)
EPOCHS = (0, 40, 80, 160)
EVALUATION_SEEDS = (288, 289)
NN = "inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn"


def checkpoint(out, arm, training_seed, epoch, source):
    return Path(source) if epoch == 0 else Path(out) / f"train_{arm}_s{training_seed}" / NN / ("GRAB_%08d.pth" % (260 + epoch))


def summarize(out, source):
    out = Path(out)
    counts = {arm: {} for arm in ARMS}
    results = {}
    hashes = {}
    for t in TRAINING_SEEDS:
        for epoch in EPOCHS:
            for seed in EVALUATION_SEEDS:
                reference = None
                for arm in ARMS:
                    name = f"eval_{arm}_t{t}_e{epoch}_s{seed}"
                    directory = out / name
                    native = json.loads((directory / "run_manifest.json").read_text())
                    result = json.loads((directory / "results.json").read_text())
                    path = checkpoint(out, arm, t, epoch, source)
                    if path not in hashes:
                        hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                    if native["run_status"] != "COMPLETED" or native["input_sha256"] != hashes[path]:
                        raise ValueError("native status/checkpoint mismatch: " + name)
                    if result["run_status"] != "COMPLETED" or result["mode"] != "evaluate":
                        raise ValueError("native evaluation missing: " + name)
                    episodes = sorted(result["per_episode"], key=lambda r: r["env_id"])
                    if [r["env_id"] for r in episodes] != list(range(96)):
                        raise ValueError("first-episode evaluation incomplete: " + name)
                    if any(r["start_frame"] != 0 or r["steps"] < 45 or abs(r["control_dt"] - 1/30) > 1e-8 for r in episodes):
                        raise ValueError("start/time contract: " + name)
                    if [sum(r["motion_id"] == m for r in episodes) for m in range(3)] != [32] * 3:
                        raise ValueError("motion balance: " + name)
                    paired = [(r["env_id"], r["motion_id"], r["start_frame"], r["steps"], r["initial_object_height"], r["control_dt"]) for r in episodes]
                    if reference is not None and paired != reference:
                        raise ValueError("native pairing mismatch: " + name)
                    reference = paired
                    n = sum(r["stable_success"] for r in episodes)
                    drops = sum(r["drop_after_success"] for r in episodes)
                    if n != result["stable_success_count"] or drops != result["drop_after_success_count"]:
                        raise ValueError("episode/count mismatch: " + name)
                    counts[arm][f"{t}/{epoch}/{seed}"] = n
                    results[name] = result
    terminal = {arm: sum(n for k, n in values.items() if k.split("/")[1] == "160") for arm, values in counts.items()}
    effects = {arm: (terminal["cm_value"] - terminal[arm]) / 384 for arm in ARMS[:2]}
    nonnegative = all(sum(counts["cm_value"][f"{t}/160/{s}"] - counts[arm][f"{t}/160/{s}"] for s in EVALUATION_SEEDS) >= 0
                      for t in TRAINING_SEEDS for arm in ARMS[:2])
    drops = {arm: sum(r["drop_after_success_count"] for name, r in results.items() if name.startswith(f"eval_{arm}_") and "_e160_" in name) for arm in ARMS}
    drop_gate = all((drops["cm_value"] - drops[arm]) / 384 <= .05 for arm in ARMS[:2])
    curves = {arm: {str(e): sum(n for k, n in values.items() if int(k.split("/")[1]) == e) / 384 for e in EPOCHS} for arm, values in counts.items()}
    areas = {arm: sum((c[str(left)] + c[str(right)]) / 2 * (right-left) * 2048 for left, right in zip(EPOCHS[:-1], EPOCHS[1:])) for arm, c in curves.items()}
    return dict(run_status="COMPLETED", conclusion="PROMISING" if min(effects.values()) >= .05 and nonnegative and drop_gate else "UNPROMISING",
        terminal_counts=terminal, terminal_success_rates={a: terminal[a]/384 for a in ARMS}, differences=effects,
        per_checkpoint_counts=counts, learning_curve_rates=curves, learning_curve_area=areas,
        drop_counts=drops, conditional_drop_rates={a: drops[a]/terminal[a] if terminal[a] else None for a in ARMS},
        native_pairing_valid=True, evaluation_points=48)
