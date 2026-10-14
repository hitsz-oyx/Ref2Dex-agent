"""Distinct full-model identity; no reuse of legacy WAM decoder patches."""
from pathlib import Path
import json
import subprocess

import torch

from .paths import ROOT
from .contracts import GOAL_ENCODING, TASK_CONDITIONING

SCHEMA = "ref2dex.ptv3-shared-wam.v1"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, model, optimizer, config, stats, step, stage, provenance, rank_rng=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("checkpoint is immutable: " + str(path))
    if config.get("goal_encoding") != GOAL_ENCODING:
        raise ValueError("checkpoint config must explicitly identify the delta-goal contract")
    if config.get("task_conditioning") != TASK_CONDITIONING:
        raise ValueError("checkpoint config must explicitly identify shared task conditioning")
    state = dict(schema=SCHEMA, stage=stage, step=step, config=config, stats=stats,
                 model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                 provenance=provenance, optimizer=optimizer.state_dict() if optimizer else None,
                 torch_rng=torch.get_rng_state(),
                 cuda_rng=[torch.cuda.get_rng_state()] if any(p.is_cuda for p in model.parameters()) else [],
                 rank_rng=rank_rng)
    temporary = path.with_suffix(".pt.part")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("schema") != SCHEMA or state.get("stage") not in ("pretrain", "robot"):
        raise ValueError("full shared-WAM checkpoint required; legacy patches are incompatible")
    if state.get("config", {}).get("goal_encoding") != GOAL_ENCODING:
        raise ValueError("legacy absolute-goal checkpoint requires explicit migration; cannot resume as delta-goal")
    if state.get("config", {}).get("task_conditioning") != TASK_CONDITIONING:
        raise ValueError("legacy hand-only goal checkpoint cannot resume as shared task conditioning")
    return state


def directory_bytes(*paths):
    """Ignore no files; retry transient ENOENT races in owned compiler caches."""
    for attempt in range(3):
        result = subprocess.run(["du", "-sb", *map(str, paths)], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return sum(int(line.split()[0]) for line in result.stdout.splitlines())
        errors = result.stderr.splitlines()
        if attempt == 2 or not errors or not all("No such file or directory" in line for line in errors):
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def checkpoint_alias(output, name, target):
    """Atomically select an immutable owned checkpoint without copying weights."""
    output = Path(output).resolve()
    path = output / target
    if (name not in ("best.pt", "latest.pt") or Path(target).name != target
            or not target.startswith("step") or path.suffix != ".pt"
            or path.resolve().parent != output or not path.is_file()):
        raise ValueError("owned existing stepN.pt and best/latest alias required")
    int(target[4:-3])
    link = output / name
    if link.exists() and not link.is_symlink():
        raise ValueError("cannot overwrite an existing checkpoint with an alias")
    temporary = output / (name + ".alias-part")
    temporary.symlink_to(target)
    temporary.replace(link)


def prune_recovery_checkpoints(output, files, config):
    """Keep milestones and recent recovery states from this run's registry only.

    Default behavior preserves every checkpoint. Explicit long-run configs may
    discard unselected recovery states; other files and prior runs stay intact.
    """
    keep = config.get("checkpoint_keep_recent")
    if keep is None:
        return files
    milestone = int(config.get("checkpoint_milestone_interval", 0))
    if int(keep) < 1 or milestone < 1:
        raise ValueError("positive recovery retention and milestone interval required")
    output = Path(output).resolve()
    steps = {}
    for name in files:
        path = Path(name)
        if path.name != name or not name.startswith("step") or path.suffix != ".pt":
            raise ValueError("checkpoint registry must contain owned stepN.pt basenames")
        if (output / name).resolve().parent != output:
            raise ValueError("checkpoint registry escapes owned run")
        steps[name] = int(name[4:-3])
    recent = sorted(files, key=steps.get)[-int(keep):]
    preserved = {name for name, step in steps.items()
                 if step == 1 or step == config["updates"] or step % milestone == 0}
    preserved.update(recent)
    for alias in ("best.pt", "latest.pt"):
        link = output / alias
        if link.is_symlink():
            target = link.resolve()
            if target.parent != output or target.name not in steps or not target.is_file():
                raise ValueError("checkpoint alias must identify a registered owned checkpoint")
            preserved.add(target.name)
    for name in files:
        if name not in preserved:
            (output / name).unlink()
    return [name for name in files if name in preserved]


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
