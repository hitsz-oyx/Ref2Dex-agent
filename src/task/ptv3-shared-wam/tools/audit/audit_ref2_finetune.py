"""Read-only ref2 checks against the running RL660 loss, curve and data.

CPU only: tiny analytical gradients and file/label arithmetic, no model fitting
or batched neural inference. Never change live trainer sources or checkpoints.
"""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK / "src"))
from ptv3_shared_wam.robot import robot_loss, native_trajectory, ACTIVE
from ptv3_shared_wam.rl_robot import RLRobotWindows
from ptv3_shared_wam.execution import ActionBuffer
from trajectory_policy.chunk_objective import control_aware_loss


def loss_check(config):
    # Exercise the real robot_loss seam with zero forecast residuals. Uniform
    # unit scales isolate objective coefficients from dataset-specific sigma.
    hand = torch.zeros(1, 24, 2, 11, 3)
    scene = torch.zeros(1, 1, 24, 32, 3)
    action = torch.full((1, 8, 12), .25, requires_grad=True)
    anchor = torch.full((1, 8, 12), .25, requires_grad=True)
    batch = dict(trajectory_target=hand, scene_target=scene,
                 trajectory_valid=torch.ones(1, 24, 2, 11, dtype=torch.bool),
                 start_hand_valid=torch.ones(1, 2, 11, dtype=torch.bool),
                 scene_valid=torch.ones(1, 1, 24, 32, dtype=torch.bool),
                 object_valid=torch.ones(1, 1, dtype=torch.bool),
                 native_action=torch.zeros_like(action))
    prediction = dict(trajectory=hand, scene=scene, action=action, action_anchor=anchor)
    scale = torch.ones(8, 12)
    _, terms = robot_loss(prediction, batch, config, SimpleNamespace(action_scale=scale))
    result = {}
    for key, value in (("action", action), ("anchor", anchor)):
        grad, = torch.autograd.grad(terms[key], value)
        result[key] = dict(
            first_step_gradient_share=float(grad[:, 0].abs().sum() / grad.abs().sum()),
            first_xyz_gradient_share=float(grad[:, 0, :3].abs().sum() / grad.abs().sum()),
            first_to_tail_per_scalar=float(grad[0, 0, 0] / grad[0, 1, 0]))
    candidate, _ = control_aware_loss(action, batch["native_action"], scale,
                                    torch.tensor(config["native_PD_scale"]))
    grad, = torch.autograd.grad(candidate, action)
    result["historical_control_aware_first_to_tail_per_scalar"] = float(grad[0, 0, 0] / grad[0, 1, 0])
    result["first_emphasis_present"] = result["anchor"]["first_to_tail_per_scalar"] > 1
    return result


def execution_default():
    tree = ast.parse((TASK / "tools/run/launch_native.py").read_text())
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and any(isinstance(a, ast.Constant) and
                a.value == "--execution-steps" for a in call.args):
            steps = next(ast.literal_eval(k.value) for k in call.keywords if k.arg == "default")
            buffer = ActionBuffer(steps)
            buffer.refill(np.zeros((8, 12), np.float32), 0)
            consumed = []
            while not buffer.needs_query:
                consumed.append(buffer.pop()[1]["chunk_index"])
            return dict(default_K=steps, consumed_indices=consumed)
    raise ValueError("cannot identify execution default")


def selection_check(run):
    curve = json.loads((run / "curve.json").read_text())
    best = min(curve, key=lambda x: x["objective"])
    control = min(curve, key=lambda x: x["metrics"]["GT/PD_xyz_mm"]["p95"])
    predicted = min(curve, key=lambda x: x["metrics"]["pred/PD_xyz_mm"]["p95"])
    def compact(row):
        return dict(step=row["step"], objective=row["objective"],
                    GT_xyz=row["metrics"]["GT/PD_xyz_mm"],
                    pred_xyz=row["metrics"]["pred/PD_xyz_mm"],
                    checkpoint_exists=(run / ("step%d.pt" % row["step"])).is_file())
    return dict(curve_rows=len(curve), total_best=compact(best),
                GT_control_p95_best=compact(control), pred_control_p95_best=compact(predicted),
                best_alias=(run / "best.pt").resolve().name,
                control_and_total_disagree=best["step"] != control["step"])


def data_check(source, config):
    # Four complete sequences per split, one startup and one later admitted
    # window each. This is a bounded wiring check, not a corpus quality claim.
    sys.path.insert(0, str(ROOT / "src/task/trajectory-policy/tools/audit"))
    from audit_competent_pd import independent_pd
    maxima = dict(action=0., tau=0., state=0., wam_to_native_tau=0., PD=0., delta_PD=0.)
    panels = []
    for split in ("train", "val"):
        dataset = RLRobotWindows(source, split)
        for seq, record in enumerate(dataset.records[:4]):
            data, _ = dataset.sequence(seq)
            ticks = dataset.ticks_by_sequence[seq]
            start = next(i for i, key in enumerate(dataset.keys) if key[1] == seq)
            for offsets in (np.flatnonzero(ticks < 16), np.flatnonzero(ticks >= 16)):
                if not len(offsets):
                    continue
                offset = int(offsets[0]); tick = int(ticks[offset]); sample = dataset[start + offset]
                rotation = data["obj"][tick, :3, :3]
                local = (data["hand"][tick] - data["obj"][tick, :3, 3]) @ rotation
                state = np.concatenate((data["q"][tick], data["dq"][tick], local.ravel(),
                                        data["obj"][tick, :3].ravel(), data["velocity"][tick]))
                tau = ((data["hand"][tick+1:tick+25] - data["hand"][tick]) @ rotation).reshape(24, 33)
                mapped = native_trajectory(torch.tensor(sample["trajectory_target"])[None],
                                          torch.tensor(sample["start_hand_valid"])[None])[0].numpy()
                actual = data["applied"][tick:tick+8][:, list(ACTIVE)]
                pd = independent_pd(data["applied"][tick], data["q"][tick], dataset.controller)
                probe = data["applied"][tick].copy()
                probe[list(ACTIVE)] = np.clip(probe[list(ACTIVE)] + .01, -1, 1)
                pd_difference = independent_pd(probe, data["q"][tick], dataset.controller) - pd
                expected_difference = ((probe - data["applied"][tick])[list(ACTIVE)] *
                                       np.asarray(config["native_PD_scale"]))
                errors = dict(action=np.max(np.abs(sample["native_action"]-actual)),
                              tau=np.max(np.abs(sample["native_tau"]-tau)),
                              state=np.max(np.abs(sample["native_state"]-state)),
                              wam_to_native_tau=np.max(np.abs(mapped-tau)),
                              PD=np.max(np.abs(pd-data["pd_targets"][tick])),
                              delta_PD=np.max(np.abs(pd_difference[list(ACTIVE)]-expected_difference)))
                maxima = {k: max(maxima[k], float(v)) for k, v in errors.items()}
                panels.append(dict(split=split, sequence=record["sequence"], tick=tick))
    return dict(windows=len(panels), panels=panels, maxima=maxima,
                pass_=max(maxima.values()) < 1e-6)


def native_input_check(run, manifest):
    provenance = manifest["provenance"]
    paths = dict(canonical=Path(provenance["canonical"]), reference=Path(provenance["reference"]),
                 source_trajectory=Path(provenance["source"]) / "trajectory.npz")
    checks = {}
    for key, path in paths.items():
        try:
            with np.load(path, allow_pickle=False):
                pass
            checks[key] = dict(path=str(path), loadable=True)
        except (OSError, ValueError) as error:
            checks[key] = dict(path=str(path), loadable=False, error=type(error).__name__)
    return dict(inputs=checks, ready=all(x["loadable"] for x in checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-control-alignment", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or ROOT / "outputs/ptv3-shared-wam" not in output.parents:
        raise ValueError("fresh report inside this Task's outputs required")
    torch.set_num_threads(2)
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    report = dict(observed_at=datetime.now(timezone.utc).isoformat(), run_id=run.name,
                  code=manifest["provenance"]["git_commit"],
                  progress=json.loads((run / "progress.json").read_text()),
                  loss=loss_check(config), execution=execution_default(), selection=selection_check(run),
                  data=data_check(manifest["provenance"]["source"], config),
                  native=native_input_check(run, manifest),
                  scope="read-only engineering audit; no causal attribution or training")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("loss", "execution", "selection", "native")}))
    print("DATA_REPLAY", report["data"]["pass_"], report["data"]["maxima"])
    if not report["data"]["pass_"]:
        raise SystemExit("FAIL: measured label/coordinate/PD replay")
    if args.require_control_alignment and (not report["loss"]["first_emphasis_present"] or
            report["selection"]["control_and_total_disagree"] or not report["native"]["ready"]):
        raise SystemExit("FAIL: first-control emphasis, selection, or native inputs are not aligned")


if __name__ == "__main__":
    main()
