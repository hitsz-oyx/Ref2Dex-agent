"""v2 identity: immutable complete model, normalizer, optimizer and per-rank RNG."""
from pathlib import Path
import torch
from .paths import SCHEMA
from ptv3_shared_wam.checkpoint import atomic_json, directory_bytes, git_commit, checkpoint_alias, prune_recovery_checkpoints


def save_checkpoint(path, model, optimizer, config, stats, step, provenance, rank_rng):
    path = Path(path)
    if path.exists():
        raise FileExistsError("immutable v2 checkpoint")
    packet = dict(schema=SCHEMA, stage="pretrain", config=config, stats=stats, step=step,
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), provenance=provenance, rank_rng=rank_rng)
    temporary = path.with_suffix(".pt.part")
    torch.save(packet, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA or state.get("stage") != "pretrain":
        raise ValueError("v2 full pretraining checkpoint required; v1 incompatible")
    return state
