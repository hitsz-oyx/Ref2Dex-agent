"""v4 identity: immutable complete model, normalizer, optimizer and per-rank RNG."""
from pathlib import Path
import torch
from .paths import SCHEMA
from ptv3_shared_wam.checkpoint import atomic_json, directory_bytes, checkpoint_alias, prune_recovery_checkpoints


def initialize_from_v3(path, model, config, stats):
    """Copy compatible v3 model weights and freshly initialize v4 C modules.

    This is deliberately a model-only transfer: the v3 optimizer, step counter
    and rank RNG are not resumed because the structured C objective changes the
    parameter groups and loss surface.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("schema") != "ref2dex.ptv3-shared-wam.v3" or state.get("stage") != "pretrain":
        raise ValueError("v3 pretraining checkpoint required for warm-start")
    old_config, old_stats = state.get("config", {}), state.get("stats", {})
    shared_fields = ("goal_encoding", "task_conditioning", "point_width", "shared_width",
                     "shared_heads", "shared_layers", "head_width", "head_heads",
                     "object_points", "history", "horizon", "position_scale_m",
                     "velocity_scale_mps", "acceleration_scale_mps2", "hand_in_ptv3",
                     "explicit_hand", "horizon_normalization", "checkpoint_interaction")
    for key in shared_fields:
        if old_config.get(key) != config.get(key):
            raise ValueError(f"v3/v4 shared architecture mismatch: {key}")
    for key in ("schema", "split", "units", "fps", "history", "horizon", "hand", "scene"):
        if old_stats.get(key) != stats.get(key):
            raise ValueError(f"v3/v4 normalization/statistics mismatch: {key}")
    source = state["model"]
    target = model.state_dict()
    compatible = {key: value for key, value in source.items()
                  if key in target and tuple(value.shape) == tuple(target[key].shape)}
    minimum = min(100, max(16, len(target) // 2))
    if len(compatible) < minimum:
        raise ValueError(f"v3 warm-start transferred too few parameters: {len(compatible)}")
    model.load_state_dict(compatible, strict=False)
    return dict(parent=str(Path(path).resolve()), parent_step=int(state["step"]),
                transferred=len(compatible), target_parameters=len(target),
                new_parameters=sorted(set(target) - set(compatible)))


def save_checkpoint(path, model, optimizer, config, stats, step, provenance, rank_rng):
    path = Path(path)
    if path.exists():
        raise FileExistsError("immutable v3 checkpoint")
    packet = dict(schema=SCHEMA, stage="pretrain", config=config, stats=stats, step=step,
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), provenance=provenance, rank_rng=rank_rng)
    temporary = path.with_suffix(".pt.part")
    torch.save(packet, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA or state.get("stage") != "pretrain":
        raise ValueError("v4 full pretraining checkpoint required")
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


def git_commit():
    import subprocess
    from .paths import ROOT
    return subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True).strip()
