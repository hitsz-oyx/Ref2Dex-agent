"""Actual same-row robot labels and a jointly trained forecaster/Chunk-R pair."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import point_labels
from .paths import add_dependencies
from .objectives import forecast_loss

add_dependencies()
from trajectory_policy.chunk_retargeter import ChunkRetargeter, chunk_windows
from trajectory_policy.retargeter_data import ACTIVE, features
from trajectory_policy.wam_shadow import WAM_TO_NATIVE
from trajectory_policy.wam_consequence import actual_consequence_sample
from trajectory_policy.chunk_objective import control_aware_loss


class RobotWindows(torch.utils.data.Dataset):
    """Full successful teacher demonstrations, split by complete environment row.

    No failed learner future is used as an expert hand label. Recovery collection
    can be admitted separately later; this V0 implements the specified airplane
    overfit stage from the existing audited demonstration bank.
    """
    def __init__(self, source, reference, canonical, split, point_count=32):
        source = Path(source)
        self.manifest = json.loads((source / "manifest.json").read_text())
        result = json.loads((source / "result.json").read_text())
        audit = json.loads((source / "audit.json").read_text())
        if (self.manifest.get("status") != "COMPLETED"
                or self.manifest.get("training_allowed") is not True
                or not result.get("teacher_readiness") or not audit.get("pass")):
            raise ValueError("completed, successful, audited teacher demonstrations required")
        with np.load(source / "trajectory.npz", allow_pickle=False) as stream:
            self.data = {key: stream[key].copy() for key in stream.files}
        d = self.data
        controls, envs = d["applied"].shape[:2]
        if (not np.array_equal(d["action"], d["applied"]) or envs < 28
                or any(len(d[k]) != controls + 1 for k in ("q", "dq", "hand", "obj", "velocity"))
                or any(not np.isfinite(d[k]).all() for k in ("q", "dq", "hand", "obj", "velocity", "applied"))):
            raise ValueError("actual same-row controls and complete terminal measurements required")
        with np.load(reference, allow_pickle=False) as stream:
            self.reference = stream["obj"].copy()
        with np.load(canonical, allow_pickle=False) as stream:
            self.canonical = {key: stream[key].copy() for key in stream.files}
        if len(self.reference) != controls + 1:
            raise ValueError("reference clock and actual trajectory lengths differ")
        if split not in ("train", "val"):
            raise ValueError("V0 exposes only train/dev robot rows")
        self.envs = list(range(16)) if split == "train" else list(range(16, 24))
        self.keys = [(tick, env) for tick in range(controls + 1 - 24) for env in self.envs
                     if not d["clipped"][tick:tick + 24, env].any()]
        if not self.keys:
            raise ValueError("no unclipped complete expert future windows")
        self.point_count = point_count
        self.controller = self.manifest["native_controller"]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        tick, env = self.keys[index]
        d = self.data
        sample = actual_consequence_sample(d["obj"], d["hand"], tick, env,
                    self.reference[tick + 24], self.canonical, sample_id=tick * 32 + env)
        # These futures are admitted teacher motion on THIS actual row.
        future = sample["action"].reshape(24, 2, 11, 9)[..., :3]
        sample["trajectory_target"] = future - sample["start_hand"][None]
        sample["trajectory_valid"] = sample["action_valid"].reshape(24, 2, 11)
        state, tau = features(*(d[k][tick, env] for k in ("q", "dq", "hand", "obj", "velocity")),
                              d["hand"][tick + np.arange(1, 25), env])
        sample.update(point_labels(sample, self.point_count))
        sample.update(native_state=state, native_tau=tau,
                      native_action=d["applied"][tick + np.arange(8), env][:, list(ACTIVE)].astype(np.float32),
                      has_action=np.bool_(True), source_id=np.int64(3))
        return sample

    def normalization(self):
        """Train rows only; repeat occurrences never change normalizers."""
        state, tau, action = chunk_windows(self.data, self.envs)
        controls = len(self.data["applied"])
        keep = np.asarray([not self.data["clipped"][t:t + 24, e].any()
                           for t in range(controls + 1 - 24) for e in self.envs])
        return [array[keep] for array in (state, tau, action)]


def initialize_retargeter(net, dataset):
    if (getattr(dataset, 'split', None) != 'train'
            and getattr(dataset, 'envs', None) != list(range(16))):
        raise ValueError("retargeter normalization must use TRAIN rows only")
    with torch.no_grad():
        for key, array, floor in zip(("state", "tau", "action"), dataset.normalization(), (.001, .001, .01)):
            if key=='action' and getattr(dataset,'action_representation','native-command.v1')!='native-command.v1':
                floor=1e-6
            value = torch.as_tensor(array, device=net.state_mean.device)
            getattr(net, key + "_mean").copy_(value.mean(0))
            getattr(net, key + "_scale").copy_(value.std(0, unbiased=False).clamp_min(floor))


def native_trajectory(trajectory, hand_valid):
    """Native right-hand link order, current-object frame; no coordinate refit."""
    if not hand_valid[:, 0].all():
        raise ValueError("Chunk-R requires every right-hand semantic point")
    order = torch.as_tensor(WAM_TO_NATIVE, device=trajectory.device)
    return trajectory[:, :, 0].index_select(2, order).reshape(-1, 24, 33)


class RobotPair(nn.Module):
    def __init__(self, forecaster, retargeter=None):
        super().__init__()
        self.forecaster = forecaster
        self.retargeter = retargeter if retargeter is not None else ChunkRetargeter(128)
        if retargeter is None:
            nn.init.normal_(self.retargeter.head.weight, std=.01)

    def forward(self, batch):
        prediction = self.forecaster(batch)
        if batch.get("has_action") is not None and batch["has_action"].any() and not batch["has_action"].all():
            raise ValueError("use separate robot and human batches; never drop mixed robot action labels")
        if batch.get("has_action") is not None and batch["has_action"].all():
            tau = native_trajectory(prediction["trajectory"], batch["start_hand_valid"])
            prediction["action"] = self.retargeter(batch["native_state"], tau)
            # This anchor updates R; recorded hand labels cannot update forecaster.
            prediction["action_anchor"] = self.retargeter(batch["native_state"], batch["native_tau"].detach())
        return prediction

    def parameter_groups(self, config):
        groups = self.forecaster.parameter_groups(config["scene_lr"], config["new_lr"],
            interaction_lr=config.get("interaction_lr"),new_layer_lr=config.get("new_layer_lr"),
            expansion_source_layers=int(config.get("expanded_source_layers",0)))
        groups.append(dict(params=list(self.retargeter.parameters()), lr=config["new_lr"], name="retargeter"))
        return groups


def robot_loss(prediction, batch, config, retargeter):
    total, terms = forecast_loss(prediction, batch, config)
    if "action" not in prediction:
        return total, terms
    scale = retargeter.action_scale
    from .action_targets import NATIVE, target_tensor
    representation=config.get('action_representation',NATIVE)
    target=target_tensor(batch,representation)
    objective = config.get("action_objective", "all8-standardized-l1.v1")
    if objective == "all8-standardized-l1.v1":
        action = ((prediction["action"] - target) / scale).abs().mean()
        anchor = ((prediction["action_anchor"] - target) / scale).abs().mean()
    elif objective == "control-aware-first.v1":
        if representation!=NATIVE:raise ValueError('physical chunks use all8 objective; native first-control loss has different units')
        pd_scale = scale.new_tensor(config["native_PD_scale"])
        action, action_terms = control_aware_loss(prediction["action"], batch["native_action"], scale, pd_scale)
        anchor, anchor_terms = control_aware_loss(prediction["action_anchor"], batch["native_action"], scale, pd_scale)
        terms.update({"action_" + k: v for k, v in action_terms.items()})
        terms.update({"anchor_" + k: v for k, v in anchor_terms.items()})
    else:
        raise ValueError("unknown robot action objective: " + objective)
    total = total + config.get("action_weight", 1.) * action + config.get("anchor_weight", 1.) * anchor
    return total, dict(terms, action=action, anchor=anchor)


def human_replay(sample):
    sample = dict(sample)
    sample.update(has_action=np.bool_(False), native_state=np.zeros(87, np.float32),
                  native_tau=np.zeros((24, 33), np.float32), native_action=np.zeros((8, 12), np.float32))
    return sample
