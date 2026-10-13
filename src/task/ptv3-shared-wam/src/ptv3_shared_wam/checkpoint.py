"""Distinct full-model identity; no reuse of legacy WAM decoder patches."""
from pathlib import Path
import json
import subprocess

import torch

from .paths import ROOT
from .contracts import GOAL_ENCODING

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


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
