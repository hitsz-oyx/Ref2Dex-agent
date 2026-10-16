"""Explicit v2 robot checkpoint admission for the shared native execution harness."""
import json
from pathlib import Path
import numpy as np
from .paths import SCHEMA


def validate_robot_packet(packet, allow_shared=False):
    from ptv3_shared_wam.contracts import GOAL_ENCODING, TASK_CONDITIONING
    config = packet.get("config", {})
    if packet.get("schema") != SCHEMA or packet.get("stage") != "robot":
        raise ValueError("full v2 robot checkpoint required")
    if (config.get("schema") != SCHEMA or config.get("horizon") != 24
            or config.get("goal_encoding") != GOAL_ENCODING
            or config.get("task_conditioning") != TASK_CONDITIONING):
        raise ValueError("v2 execution input contract differs")
    if config.get("action_interface") is not None and not (allow_shared and config["action_interface"] == "shared"):
        raise ValueError("shared-query actor execution requires its own native bridge; legacy tau harness rejected")
    if allow_shared and (config.get('action_interface') != 'shared' or config.get('anchor_weight') != 0):
        raise ValueError('explicit shared-only bridge requires shared interface and no GT tau anchor')
    admitted = ('native-command.v1','wrist-geometry-continuous-lift.v1') if allow_shared else ('wrist-geometry-continuous-lift.v1',)
    if config.get("action_representation") not in admitted:
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


def load_robot_checkpoint(path, allow_shared=False):
    import torch
    return validate_robot_packet(torch.load(path, map_location="cpu", weights_only=False), allow_shared)


def predict_shared_action(model, batch, state,current_obs=None,return_pooled=False):
    """Run the trained shared interface with measured current state; labels are absent."""
    from ptv3_shared_wam.robot import native_trajectory
    if model.action_interface != 'shared':
        raise ValueError('shared-only model required')
    prediction = model.forecaster(batch, return_shared=True)
    tau = native_trajectory(prediction['trajectory'], batch['start_hand_valid'])
    chunk = model.retargeter(state, tau, prediction['shared_tokens'], prediction['shared_valid'],current_obs)
    if return_pooled:
        from .wrist_shared_probe import pool_shared
        count=prediction['shared_tokens'].shape[1]-22-batch['object_goal_valid'].shape[1]
        pooled=pool_shared(prediction['shared_tokens'],prediction['shared_valid'],count)
        return chunk,tau,pooled
    return chunk, tau


def mix_teacher_components(student_intended, teacher_action):
    """Diagnostic oracle intervention in command units; never a deployable student policy."""
    from trajectory_policy.retargeter_data import ACTIVE
    if student_intended.shape != (3,18) or teacher_action.shape != (3,18):
        raise ValueError('Three teacher/diagnostic rows of18 commands required')
    mixed=student_intended.clone()
    mixed[1,:6]=teacher_action[1,:6]
    mixed[2,list(ACTIVE)[6:]]=teacher_action[2,list(ACTIVE)[6:]]
    return mixed
