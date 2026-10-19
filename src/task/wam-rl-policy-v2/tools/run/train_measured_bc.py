"""Train the Current421 BC executor on the audited same-execution corpus.

This entry point deliberately consumes ``TeacherRolloutCorpus`` directly.  The
legacy cache builder is for the old 319-dimensional contract and must not be
used for the measured force/shape/task10 corpus.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("REF2DEX_RUN_ROOT", TASK.parents[2])).resolve()
SRC = TASK / "src"


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def resource(gpu):
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu,memory.used,power.draw",
         "--format=csv,noheader"], text=True).strip()


def load_split(manifest, split):
    import numpy as np
    from wam_rl_policy_v2.teacher_dataset import TeacherRolloutCorpus

    corpus = TeacherRolloutCorpus(manifest, split)
    measured, task, action, object_ids, points = [], [], [], [], []
    sequences = []
    for index in range(len(corpus)):
        episode = corpus.episode(index)
        keep = np.asarray(episode["active"], dtype=bool)
        if not keep.any():
            continue
        object_id = len(points)
        points.append(episode["object_points"])
        measured.append(episode["measured"][keep])
        task.append(episode["task"][keep])
        action.append(episode["action"][keep])
        object_ids.append(np.full(int(keep.sum()), object_id, dtype=np.int64))
        sequences.append(dict(sequence=episode["sequence"], object_id=object_id,
                              commands=int(keep.sum())))
    if not measured:
        raise ValueError(f"empty {split} corpus")
    return dict(measured=np.concatenate(measured), task=np.concatenate(task),
                action=np.concatenate(action), object_id=np.concatenate(object_ids),
                points=np.stack(points), sequences=sequences)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=246)
    p.add_argument("--seconds", type=int, default=28800)
    a = p.parse_args()
    out = a.output.resolve()
    if ROOT / "outputs/wam-rl-policy-v2" not in out.parents:
        raise ValueError("Task output required")
    if out.exists() and any(x.name not in {"train.log", "pid"} for x in out.iterdir()):
        raise ValueError("fresh bounded Current421 run required")
    if not 1 <= a.batch <= 8192 or a.steps != 50000:
        raise ValueError("fresh bounded Current421 run required")
    if not 60 <= a.seconds <= 28800:
        raise ValueError("bounded wall-clock budget required")
    if not a.manifest.exists():
        raise FileNotFoundError(a.manifest)
    out.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "tmp" / out.name
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), TMPDIR=str(scratch),
                      OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                      PYTHONDONTWRITEBYTECODE="1")
    import sys
    sys.path.insert(0, str(SRC))
    import numpy as np
    import torch
    from wam_rl_policy_v2.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v2.task_annotations import GOAL_SOURCE

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    train = load_split(a.manifest.resolve(), "train")
    dev = load_split(a.manifest.resolve(), "dev")
    if train["measured"].shape[1] != 347 or train["task"].shape[1] != 10:
        raise ValueError("audited Current421 fields have unexpected shape")

    device = torch.device("cuda")
    def tensor(split, key):
        return torch.as_tensor(split[key], device=device)
    train_t = {k: tensor(train, k) for k in ("measured", "task", "action", "object_id", "points")}
    dev_t = {k: tensor(dev, k) for k in ("measured", "task", "action", "object_id", "points")}

    model = MeasuredInspirePolicy(with_tau=False).to(device)
    # The force channels are measured in physical units and are much wider than
    # the clipped legacy state.  Normalize only the fixed robot/validity input;
    # the shape block is normalized using the initial encoder and remains
    # differentiable during training.
    with torch.no_grad():
        measured = train_t["measured"]
        model.actor.x_mean[:345].copy_(measured[:, :345].mean(0))
        model.actor.x_scale[:345].copy_(measured[:, :345].std(0).clamp_min(.05))
        shape0 = model.shape(train_t["points"])
        model.actor.x_mean[345:409].copy_(shape0.mean(0))
        model.actor.x_scale[345:409].copy_(shape0.std(0).clamp_min(.05))
        model.actor.x_mean[409:411].zero_()
        model.actor.x_scale[409:411].fill_(1.)
        model.actor.y_mean.copy_(train_t["action"].mean(0))
        model.actor.y_scale.copy_(train_t["action"].std(0).clamp_min(.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    rng = np.random.default_rng(a.seed + 1)
    best = float("inf")
    curve = []
    started = time.monotonic()
    commit = os.environ.get("REF2DEX_CODE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = dict(schema="ref2dex.inspire-measured-current-bc.v1", status="RUNNING",
                    experiment_id="P-20261020-inspire-current421-bc", run_id=out.name,
                    git_commit=commit, source_manifest=str(a.manifest.resolve()),
                    goal_source=GOAL_SOURCE, input_dim=421, measured_dim=347, task_dim=10,
                    action_dim=12, hidden_dims=[1024, 1024, 512, 512], activation="elu",
                    optimizer="AdamW", lr=3e-4, weight_decay=1e-5, steps=a.steps,
                    batch=a.batch, seed=a.seed, train_rows=int(len(train["action"])),
                    dev_rows=int(len(dev["action"])), train_sequences=len(train["sequences"]),
                    dev_sequences=len(dev["sequences"]), normalization="train measured/std and initial shape/std",
                    stop_condition="8h wall clock or owned STOP")
    write(out / "manifest.json", manifest)

    @torch.no_grad()
    def evaluate():
        model.eval()
        pred = model(dev_t["measured"], dev_t["task"], dev_t["points"][dev_t["object_id"]])
        error = (pred - dev_t["action"]).abs()
        result = dict(command_mae=float(error.mean()), root_mae=float(error[:, :6].mean()),
                      finger_mae=float(error[:, 6:].mean()))
        model.train()
        return result

    initial = evaluate()
    write(out / "initial-dev.json", initial)
    for step in range(1, a.steps + 1):
        elapsed = time.monotonic() - started
        if elapsed > a.seconds or (out / "STOP").exists():
            raise TimeoutError("owned Current421 BC stop condition")
        ids = torch.as_tensor(rng.integers(0, len(train["action"]), size=a.batch),
                              device=device, dtype=torch.long)
        prediction = model(train_t["measured"][ids], train_t["task"][ids],
                           train_t["points"][train_t["object_id"][ids]])
        loss = ((prediction - train_t["action"][ids]) / model.actor.y_scale).abs().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Current421 BC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 1000 == 0 or step == a.steps:
            metrics = evaluate()
            elapsed = time.monotonic() - started
            row = dict(step=step, loss=float(loss), dev=metrics, elapsed_s=elapsed,
                       eta_s=elapsed / step * (a.steps - step), gpu=resource(a.gpu))
            curve.append(row)
            write(out / "curve.json", curve)
            write(out / "progress.json", row)
            packet = dict(schema=manifest["schema"], step=step, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), config=manifest,
                          torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                          numpy_rng=rng.bit_generator.state)
            torch.save(packet, out / "latest.pt.part")
            (out / "latest.pt.part").replace(out / "latest.pt")
            if metrics["command_mae"] < best:
                best = metrics["command_mae"]
                torch.save(packet, out / "best.pt.part")
                (out / "best.pt.part").replace(out / "best.pt")
            print(json.dumps(row), flush=True)
    manifest.update(status="COMPLETED", elapsed_s=time.monotonic() - started,
                    best_dev_command_mae=best)
    write(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
