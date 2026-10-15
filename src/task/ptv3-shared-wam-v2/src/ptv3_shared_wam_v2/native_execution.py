"""Explicit v2 robot checkpoint admission for the shared native execution harness."""
import json
from pathlib import Path
import numpy as np
from .paths import SCHEMA


def validate_robot_packet(packet):
    from ptv3_shared_wam.contracts import GOAL_ENCODING, TASK_CONDITIONING
    config = packet.get("config", {})
    if packet.get("schema") != SCHEMA or packet.get("stage") != "robot":
        raise ValueError("full v2 robot checkpoint required")
    if (config.get("schema") != SCHEMA or config.get("horizon") != 24
            or config.get("goal_encoding") != GOAL_ENCODING
            or config.get("task_conditioning") != TASK_CONDITIONING):
        raise ValueError("v2 execution input contract differs")
    if config.get("action_representation") != "wrist-geometry-continuous-lift.v1":
        raise ValueError("qualified q_PD geometry-lift output required")
    if not packet.get("provenance", {}).get("source"):
        raise ValueError("original source and native controller required")
    roots = {key.split(".")[0] for key in packet.get("model", {})}
    if roots != {"forecaster", "retargeter"}:
        raise ValueError("complete WAM and R required")
    # v2 training stored the source identity, not a duplicate controller config.
    # Bind to that exact corpus in memory; never rewrite the archived weights.
    source = Path(packet["provenance"]["source"]).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest.get("schema") != "ref2dex.rl660-corpus.v1"
            or manifest.get("status") != "COMPLETED" or not manifest.get("training_allowed")):
        raise ValueError("original admitted RL660 corpus required")
    controller = manifest["native_controller"]
    offset, scale = (np.asarray(controller[k]) for k in ("offset", "scale"))
    if (offset.shape != (18,) or scale.shape != (18,)
            or not np.isfinite(offset).all() or not np.isfinite(scale).all()
            or (scale == 0).any()):
        raise ValueError("finite 18D native controller required")
    if config.get("native_controller", controller) != controller:
        raise ValueError("checkpoint/source controller mismatch")
    return dict(packet, config=dict(config, native_controller=controller),
                execution_controller_source=str(source / "manifest.json"))


def load_robot_checkpoint(path):
    import torch
    return validate_robot_packet(torch.load(path, map_location="cpu", weights_only=False))
