"""Offline H -> tau -> PointWorld effect -> trajectory-C2 planner adapter."""

from pathlib import Path
import sys

import numpy as np
import torch

from .data import sha
from .hand_execution import SCHEMA as BRIDGE_SCHEMA, HandExecution, compose_motion
from .old_utility import (deterministic_group_mean, future_from_prediction, pw_sample)
from .trajectory_utility import SCHEMA, TrajectoryUtility


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "src/task/cm-interaction-oracle/src"))
from oracle_y_utility import candidate_deltas  # noqa: E402


class TrajectoryPlanner:
    """Frozen prospective adapter; all future geometry remains model-generated."""

    def __init__(self, bridge, evaluator, pointworld, canonical, device="cuda:0"):
        self.device = torch.device(device)
        pw_root = ROOT / "src/task/cm-pointflow-effect-pretrain"
        sys.path.insert(0, str(pw_root / "src"))
        from oakink_wm.pointworld_temporal import VENDOR, capped_collate, model_from_config
        from oakink_wm.pointworld_performance import install_fused_hilbert
        import oakink_wm.pointworld as spatial
        import oakink_wm.pointworld_temporal as temporal

        bridge = Path(bridge).resolve(); evaluator = Path(evaluator).resolve()
        pointworld = Path(pointworld).resolve(); canonical = Path(canonical).resolve()
        g = torch.load(bridge, map_location="cpu", weights_only=False)
        if g.get("schema") != BRIDGE_SCHEMA or g.get("arm") != "HA":
            raise ValueError("action-conditioned hand bridge required")
        self.bridge = HandExecution(**g["architecture"]).to(self.device).eval()
        self.bridge.load_state_dict(g["model"], strict=True)
        self.bridge_stats = tuple(torch.as_tensor(value, device=self.device) for value in g["statistics"])
        self.bridge_unit = float(g["output_unit_m"])
        self.plan_unit = torch.as_tensor(g["plan_unit"], device=self.device)

        q = torch.load(evaluator, map_location="cpu", weights_only=False)
        if q.get("schema") != SCHEMA or q.get("arm") != "C2":
            raise ValueError("trajectory C2 checkpoint required")
        self.evaluator = TrajectoryUtility(**{
            key: q["architecture"][key] for key in ("history_dim", "width", "layers")
        }).to(self.device).eval()
        self.evaluator.load_state_dict(q["model"], strict=True)
        self.evaluator_stats = {
            key: tuple(torch.as_tensor(value, device=self.device) for value in pair)
            for key, pair in q["statistics"].items()
        }

        state = torch.load(pointworld, map_location="cpu", weights_only=False)
        self.source_hashes = {}
        for base, key in ((pw_root, "implementation_sources"), (VENDOR, "vendor_sources")):
            for name, digest in state["identity"][key].items():
                path = (base / name).resolve()
                if sha(path) != digest:
                    raise ValueError("PointWorld source drift: " + str(path))
                self.source_hashes[str(path)] = digest
        install_fused_hilbert()
        spatial.mean_groups = temporal.mean_groups = deterministic_group_mean
        self.pointworld = model_from_config(state["identity"]["stats"], state["config"]).to(self.device).eval()
        self.pointworld.load_state_dict(state["model"])
        self.collate = capped_collate
        del state
        with np.load(canonical, allow_pickle=False) as source:
            self.canonical = {key: source[key] for key in source.files}

        self.plans = np.zeros((7, 24, 18), dtype="float32")
        self.plans[:, :8] = candidate_deltas().numpy()[:, None]
        self.hashes = {str(path): sha(path) for path in (bridge, evaluator, pointworld, canonical)}
        self.hashes.update(self.source_hashes)
        self.hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
        self.first_repeat_checked = False

    @torch.inference_mode()
    def predict_hand(self, history, hand_history, plans):
        history = np.asarray(history, dtype="float32")
        hand_history = np.asarray(hand_history, dtype="float32")
        plans = np.asarray(plans, dtype="float32")
        if (history.ndim != 2 or history.shape[1] != 1442
                or hand_history.ndim != 4 or hand_history.shape[1:] != (4, 11, 3)
                or plans.shape != (len(history), 24, 18)):
            raise ValueError("trajectory planner input shape mismatch")
        state = np.concatenate((history, hand_history.reshape(len(history), -1)), -1)
        mean, scale = self.bridge_stats
        state = (torch.as_tensor(state, device=self.device) - mean) / scale
        plan = torch.as_tensor(plans, device=self.device) / self.plan_unit
        motion = self.bridge(state, plan) * self.bridge_unit
        return compose_motion(torch.as_tensor(hand_history[:, -1], device=self.device), motion).cpu().numpy()

    @torch.inference_mode()
    def predict_object(self, object_history, hand_history, hand_future):
        object_history = np.asarray(object_history)
        hand_history = np.asarray(hand_history)
        hand_future = np.asarray(hand_future)
        if (object_history.shape != (len(hand_future), 4, 4, 4)
                or hand_history.shape != (len(hand_future), 4, 11, 3)
                or hand_future.shape != (len(hand_future), 24, 11, 3)):
            raise ValueError("PointWorld planner input shape mismatch")
        values = []
        for begin in range(0, len(hand_future), 8):
            end = min(begin + 8, len(hand_future))
            samples = [pw_sample(object_history[i], hand_history[i], hand_future[i], self.canonical, i)
                       for i in range(begin, end)]
            batch = {key: value.to(self.device) for key, value in self.collate(samples).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = self.pointworld(batch, "action")
            if not self.first_repeat_checked:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    repeat = self.pointworld(batch, "action")
                if (not torch.equal(prediction["rotation"], repeat["rotation"])
                        or not torch.equal(prediction["translation"], repeat["translation"])):
                    raise ValueError("PointWorld repeatability failed")
                self.first_repeat_checked = True
            rotation = prediction["rotation"][:, 0].float().cpu().numpy()
            translation = prediction["translation"][:, 0].float().cpu().numpy()
            values.extend(future_from_prediction(rotation[j], translation[j], hand_future[i])
                          for j, i in enumerate(range(begin, end)))
        return np.stack(values)

    @torch.inference_mode()
    def score(self, history, trajectory, effect):
        history = np.asarray(history, dtype="float32")
        trajectory = np.asarray(trajectory, dtype="float32").reshape(-1, 24, 33)
        effect = np.asarray(effect, dtype="float32")[..., :12]
        if history.shape != (len(trajectory), 1442) or effect.shape != (len(trajectory), 24, 12):
            raise ValueError("trajectory score shape mismatch")
        values = {}
        for key, raw in (("history", history), ("trajectory", trajectory), ("effect", effect)):
            mean, scale = self.evaluator_stats[key]
            values[key] = (torch.as_tensor(raw, device=self.device) - mean) / scale
        result = self.evaluator(values["history"], values["trajectory"], values["effect"], True)
        result = result.cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("nonfinite trajectory score")
        return result

    def choose(self, history, object_history, hand_history):
        history = np.asarray(history, dtype="float32")
        object_history = np.asarray(object_history, dtype="float32")
        hand_history = np.asarray(hand_history, dtype="float32")
        n = len(history)
        if object_history.shape != (n, 4, 4, 4) or hand_history.shape != (n, 4, 11, 3):
            raise ValueError("candidate choice history shape mismatch")
        repeated_history = np.repeat(history, 7, axis=0)
        repeated_hands = np.repeat(hand_history, 7, axis=0)
        plans = np.tile(self.plans, (n, 1, 1))
        trajectory = self.predict_hand(repeated_history, repeated_hands, plans)
        repeated_objects = np.repeat(object_history, 7, axis=0)
        repeated_hand_history = np.repeat(hand_history, 7, axis=0)
        effect = self.predict_object(repeated_objects, repeated_hand_history, trajectory)
        values = self.score(repeated_history, trajectory, effect).reshape(n, 7)
        return values.argmax(1), values, trajectory.reshape(n, 7, 24, 11, 3), effect.reshape(n, 7, 24, 45)

    def verify(self):
        if any(sha(path) != digest for path, digest in self.hashes.items()):
            raise ValueError("trajectory planner frozen input drift")
