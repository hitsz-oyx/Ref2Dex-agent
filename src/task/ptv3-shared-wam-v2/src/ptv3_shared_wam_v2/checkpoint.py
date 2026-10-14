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


def restore_optimizer(optimizer, state):
    """Restore CUDA scalar steps required by fused/capturable AdamW.

    Torch2.0 deliberately leaves state['step'] on its loaded device, even
    when a CPU-mapped checkpoint feeds a CUDA fused optimizer. Newer Torch
    already handles this conversion; moving the scalar is idempotent there.
    Unfused, non-capturable optimizers retain their normal CPU step storage.
    """
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        if not (group.get("fused", False) or group.get("capturable", False)):
            continue
        for parameter in group["params"]:
            slot = optimizer.state.get(parameter, {})
            step = slot.get("step")
            if isinstance(step, torch.Tensor):
                slot["step"] = step.to(device=parameter.device)
