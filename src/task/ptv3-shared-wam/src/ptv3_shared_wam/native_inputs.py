"""Resolve one explicit motion without mixing RL660 geometry or oracle rows."""
import json
from pathlib import Path

import numpy as np


def resolve_native_inputs(packet, sequence=None):
    source = Path(packet["provenance"]["source"]).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema") != "ref2dex.rl660-corpus.v1":
        if sequence is not None:
            raise ValueError("sequence selector is only valid for an RL660 corpus")
        return dict(kind="legacy", sequence=None, trajectory=str(source / "trajectory.npz"),
                    reference=packet["provenance"]["reference"],
                    canonical=packet["provenance"]["canonical"], oracle_env=24)
    if manifest.get("status") != "COMPLETED" or not manifest.get("training_allowed"):
        raise ValueError("complete admitted RL660 corpus required")
    if not sequence:
        raise ValueError("RL660 native execution requires an explicit --sequence")
    if manifest["native_controller"] != packet["config"]["native_controller"]:
        raise ValueError("checkpoint and selected corpus controller differ")
    records = [r for r in manifest["records"] if r["sequence"] == sequence]
    if len(records) != 1:
        raise ValueError("unknown or nonunique RL660 sequence")
    record = records[0]
    folder = (source / record["path"]).resolve()
    if source not in folder.parents:
        raise ValueError("record path escapes RL660 corpus")
    result = dict(kind="rl660", sequence=sequence, split=record["split"],
                  object=record.get("object"), frames=record["frames"],
                  trajectory=str(folder / "trajectory.npz"),
                  reference=str(folder / "trajectory.npz"), canonical=str(folder / "canonical.npz"))
    membership = manifest.get("source_membership")
    if membership:
        metadata = json.loads((Path(membership) / "manifest.json").read_text())
        geometric = Path(metadata["source_geometric_root"]).resolve()
        motion = (geometric / sequence).resolve()
        if geometric not in motion.parents or not (motion / "interaction_hand_inspire.pt").is_file():
            raise ValueError("selected geometric motion is missing or outside source")
        result["motion"] = str(motion)
    for name in ("trajectory", "reference", "canonical"):
        if not Path(result[name]).is_file():
            raise FileNotFoundError(result[name])
    return result


def load_native_arrays(selection):
    with np.load(selection["trajectory"], allow_pickle=False) as stream:
        if selection["kind"] == "rl660":
            oracle = stream["hand"].copy()
            reference = stream["reference_obj"].copy()
        else:
            oracle = stream["hand"][:, selection["oracle_env"]].copy()
            reference = None
    if reference is None:
        with np.load(selection["reference"], allow_pickle=False) as stream:
            reference = stream["obj"].copy()
    with np.load(selection["canonical"], allow_pickle=False) as stream:
        canonical = {k: stream[k].copy() for k in stream.files}
    if (oracle.shape != (len(reference), 11, 3) or reference.shape[1:] != (4, 4)
            or not np.isfinite(oracle).all() or not np.isfinite(reference).all()):
        raise ValueError("same-sequence finite oracle/reference and complete terminal states required")
    return dict(oracle=oracle, reference=reference, canonical=canonical)


def install_selected_motion_contract(task_class, selection):
    """Disable legacy duplication/left-contact filtering for one corpus record.

    RL660 was collected without this filter. Restore export_rl before physics
    or observation code runs; only table/motion membership is affected.
    """
    if selection["kind"] != "rl660":
        return
    original = task_class._load_table
    def selected_table(task, files):
        if not files or any(Path(p).name != selection["sequence"] for p in files):
            raise ValueError("native table loader received another sequence")
        exporting = task.export_rl
        task.export_rl = True
        try:
            original(task, [selection["motion"]])
        finally:
            task.export_rl = exporting
        if [str(Path(p).resolve()) for p in task.motion_file] != [selection["motion"]]:
            raise ValueError("native table loader changed the selected motion")
    task_class._load_table = selected_table


def load_execution_retargeter(pair, override=None, device="cpu"):
    """Load a compatible 8-action R-only candidate without copying a full WAM."""
    import torch
    from .paths import add_dependencies
    add_dependencies()
    from trajectory_policy.chunk_retargeter import ChunkRetargeter
    from .action_targets import NATIVE, GEOMETRY, R_ONLY_SCHEMA, validate_representation
    representation=validate_representation(pair['config'].get('action_representation',NATIVE))
    legacy_override=False
    state = {k[len("retargeter."):]: v for k, v in pair["model"].items() if k.startswith("retargeter.")}
    if override:
        path = Path(override).resolve()
        packet = torch.load(path, map_location="cpu", weights_only=False)
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if (manifest.get("status") != "COMPLETED"
                or packet.get("schema") not in ("ref2dex.rl660-R-only.v1",R_ONLY_SCHEMA) or packet.get("width") != 128 or not manifest.get("source")
                or Path(manifest["source"]).resolve() != Path(pair["provenance"]["source"]).resolve()):
            raise ValueError("R-only schema/width/source differs from full pair")
        candidate = packet["model"]
        legacy_override=packet['schema']!=R_ONLY_SCHEMA
        representation=validate_representation(packet.get('action_representation',NATIVE))
        if legacy_override and representation!=NATIVE:raise ValueError('legacy R has native units')
        if not legacy_override:
            if (packet.get('native_controller')!=pair['config'].get('native_controller')
                    or manifest.get('normalization')!='TRAIN-only per-horizon-coordinate.v1'
                    or manifest.get('representations',[]).count(representation)!=1):
                raise ValueError('physical R controller/representation/normalization differs')
            if representation==GEOMETRY and not manifest.get('geometry_native_roundtrip_pass'):
                raise ValueError('geometry Euler branch roundtrip has not qualified native execution')
        if candidate["action_queries"].shape != (8, 128):
            raise ValueError("native chunk execution requires an 8-query candidate; first-only C is diagnostic")
        if not torch.allclose(torch.as_tensor(packet["native_PD_scale"]).cpu(),
                              torch.tensor(pair["config"]["native_PD_scale"]), atol=1e-7, rtol=0):
            raise ValueError("R-only PD scales differ")
        for name in (("state", "tau", "action") if legacy_override else ("state", "tau")):
            for suffix in ("_mean", "_scale"):
                key = name + suffix
                if not torch.equal(candidate[key].cpu(), state[key].cpu()):
                    raise ValueError("R-only TRAIN normalizers differ: " + key)
        state = candidate
    from .target_retargeter import make_retargeter
    model = make_retargeter(representation)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("nonfinite execution R")
    if any((getattr(model,name+'_scale')<=0).any() for name in ('state','tau','action')):
        raise ValueError('positive execution normalizers required')
    model.action_representation=representation
    model.first_only=legacy_override or pair['config'].get('action_objective')=='control-aware-first.v1'
    return model.to(device).eval()
