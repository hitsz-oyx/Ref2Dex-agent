"""v4 identity: immutable complete model, normalizer, optimizer and per-rank RNG."""
from pathlib import Path
import hashlib
import json
import torch
from .paths import SCHEMA, INPUT_SEMANTICS, LABEL_SEMANTICS
from ptv3_shared_wam.checkpoint import atomic_json, directory_bytes, checkpoint_alias
from .objectives import OBJECTIVE_SEMANTICS, EVALUATION_SEMANTICS


def data_identity(stats):
    recipe = stats.get('recipe', {})
    return dict(recipe_sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(),
                surface_geometry_sha256=recipe.get('surface_geometry_sha256'),
                input_semantics=INPUT_SEMANTICS, interaction_semantics=LABEL_SEMANTICS)


def validate_resume_contract(parent, config, stats, world):
    if (parent.get('objective_semantics') != OBJECTIVE_SEMANTICS
            or parent.get('evaluation_semantics') != EVALUATION_SEMANTICS):
        raise ValueError('legacy objective/evaluation checkpoint cannot be silently resumed')
    if (parent.get('data_identity') != data_identity(stats) or parent['stats'] != stats
            or parent['provenance']['world_size'] != world or len(parent['rank_rng']) != world):
        raise ValueError('resume requires identical bound data/statistics/world/rank RNG')
    if not parent.get('selection'):
        raise ValueError('resume checkpoint must preserve historical best selection')


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
                     "explicit_hand", "horizon_normalization", "checkpoint_interaction",
                     "scene_tokens", "patch_size", "voxel_m", "voxel_origin_m")
    for key in shared_fields:
        if old_config.get(key) != config.get(key):
            raise ValueError(f"v3/v4 shared architecture mismatch: {key}")
    # The PTv3 backbone is loaded by shape-compatible name, so an unchecked
    # architecture difference would silently copy old weights into a new
    # network.  Check the complete trunk declaration as well.
    if old_config.get("ptv3") != config.get("ptv3"):
        raise ValueError("v3/v4 PTv3 architecture mismatch")
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


def normalization_identity(stats):
    """Normalizer arrays only; a new window filter/recipe is allowed."""
    keys = ("schema", "split", "units", "fps", "history", "horizon", "source_ids",
            "hand", "scene")
    return {key: stats.get(key) for key in keys}


def initialize_from_v4(path, model, config, stats):
    """Load a v4 checkpoint as model-only initialization for a NEW run.

    This is deliberately not a resume: the optimizer, step counter, per-rank
    RNG and historical best belong to the old input/label semantics.  The new
    run starts at step 0 with a fresh data identity and selection record.
    """
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA or state.get("stage") != "pretrain":
        raise ValueError("v4 pretraining checkpoint required for model-only warm start")
    if normalization_identity(state.get("stats", {})) != normalization_identity(stats):
        raise ValueError("v4 warm start requires identical normalization arrays")
    parent_config = state.get("config", {})
    for key, value in config.items():
        # Execution-only fields may differ; architecture and geometry may not.
        if key in ("batch_size", "updates", "max_seconds", "artifact_budget_bytes",
                   "validation_samples", "validation_seed", "validation_interval",
                   "checkpoint_interval", "checkpoint_keep_recent",
                   "checkpoint_milestone_interval", "save_first_step", "log_interval",
                   "profile_steps", "expected_world_size"):
            continue
        if parent_config.get(key) != value:
            raise ValueError(f"v4 warm start config mismatch: {key}")
    model.load_state_dict(state["model"], strict=True)
    return dict(parent=str(Path(path).resolve()), parent_step=int(state.get("step", 0)),
                parent_run_id=state.get("provenance", {}).get("run_id"),
                parent_data_identity=state.get("data_identity"),
                initialization="v4 model-only warm start; optimizer/step/RNG/selection reset")


def save_checkpoint(path, model, optimizer, config, stats, step, provenance, rank_rng, selection):
    path = Path(path)
    if path.exists():
        raise FileExistsError("immutable v3 checkpoint")
    packet = dict(schema=SCHEMA, stage="pretrain", config=config, stats=stats, step=step,
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        optimizer=optimizer.state_dict(), provenance=provenance, rank_rng=rank_rng,
        selection=dict(selection), objective_semantics=OBJECTIVE_SEMANTICS,
        evaluation_semantics=EVALUATION_SEMANTICS, data_identity=data_identity(stats))
    temporary = path.with_suffix(".pt.part")
    torch.save(packet, temporary)
    temporary.replace(path)
    atomic_json(path.with_suffix('.selection.json'), dict(selection=selection))


def prune_recovery_checkpoints(output, files, config):
    """Keep the best dependencies of every retained recovery checkpoint."""
    output = Path(output).resolve()
    if config.get('checkpoint_keep_recent') is None:
        return files
    keep, milestone = int(config['checkpoint_keep_recent']), int(config['checkpoint_milestone_interval'])
    if keep < 1 or milestone < 1:
        raise ValueError('positive checkpoint retention required')
    steps = {}
    for name in files:
        path = output / name
        if Path(name).name != name or path.resolve().parent != output or not path.is_file():
            raise ValueError('owned existing checkpoint required')
        steps[name] = int(Path(name).stem[4:])
    preserved = set(sorted(files, key=steps.get)[-keep:])
    preserved.update(name for name, step in steps.items()
                     if step == 1 or step == config['updates'] or step % milestone == 0)
    for alias in ('best.pt', 'latest.pt'):
        path = output / alias
        if path.is_symlink():
            if path.resolve().parent != output or path.resolve().name not in steps:
                raise ValueError('checkpoint alias must identify an owned registered file')
            preserved.add(path.resolve().name)
    for name in list(preserved):
        sidecar = (output / name).with_suffix('.selection.json')
        if not sidecar.exists():
            raise ValueError('selection dependency record required')
        selected = Path(json.loads(sidecar.read_text())['selection']['checkpoint'])
        if selected.parent == output:
            if selected.name not in steps:
                raise ValueError('retained checkpoint is missing its historical best')
            preserved.add(selected.name)
    for name in files:
        if name not in preserved:
            (output/name).unlink()
            (output/name).with_suffix('.selection.json').unlink()
    return [name for name in files if name in preserved]


def load_checkpoint(path):
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA or state.get("stage") != "pretrain":
        raise ValueError("v4 full pretraining checkpoint required")
    # An inherited checkpoint is a byte copy/hard link of an old run, so its
    # embedded selection still names the old directory.  The adjacent
    # portable sidecar is part of the read protocol and binds the same file to
    # the new run's owned best path.
    sidecar = path.with_suffix(".selection.json")
    if sidecar.exists():
        payload = json.loads(sidecar.read_text())
        selection = payload.get("selection")
        if selection is not None:
            if not isinstance(selection, dict) or "checkpoint" not in selection:
                raise ValueError("malformed portable selection sidecar")
            state["selection"] = selection
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
