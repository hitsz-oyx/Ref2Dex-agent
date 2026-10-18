"""Online WAM-v4 adapter: explicit 512/32/128 inputs and output contract.

The v4 checkpoint cannot reuse the v3 ``batched_inputs`` path.  v4 is
target-only: 512 canonical target-surface points with 18-D features, 32 fixed
flow queries, 128 fixed interaction queries and their normals, plus a current
SE(3) goal delta.  This module owns that boundary and keeps the summary
(``encode``) and full ``tau``/``C`` (``forward``) paths separate and explicit.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np
import torch

TASK_ROOT = Path(__file__).resolve().parents[2]  # src/task/wam-rl-policy
_DATA_SRC = TASK_ROOT.parent / "3d-data-preparation" / "src"
_VIDEO_SRC = TASK_ROOT.parent / "video-data-preparation" / "src"
_SHADOW_SRC = TASK_ROOT.parent / "shadow-finetune" / "src"
for _path in (_DATA_SRC, _VIDEO_SRC, _SHADOW_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from video_data_preparation.unidexgrasp_data import object_poses  # noqa: E402

from pose_mesh_data.v4_interaction import (  # noqa: E402
    FEATURE_STATIC, FEATURE_TARGET, STATIC_THRESHOLD_M,
    INTERACTION_POINTS, FLOW_POINTS, SURFACE_POINTS,
    interaction_origin, scene_target_static,
)
from pose_mesh_data.v4_windows import prepare_v4_geometry  # noqa: E402

V4_SCHEMA = "ref2dex.ptv3-shared-wam.v4"
DENSE_SCHEMAS = ("ref2dex.wam-v4-dense-surface.v1", "ref2dex.wam-v4-dense-surface.v2")
HAND_SLOTS = ("right", "left")
SUMMARY_DIM = 2304


def _normal_transform(normals, rotation):
    return np.asarray(normals) @ rotation.T


def _dynamics(values):
    """H4 velocity/acceleration/displacement along axis 1."""
    return np.concatenate(((values[:, -1] - values[:, -2]) * 30,
                           (values[:, -1] - 2 * values[:, -2] + values[:, -3]) * 900,
                           values[:, -1] - values[:, 0]), axis=-1)


def _read_dense_manifest(path, expected_sha256):
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("dense surface manifest SHA-256 mismatch")
    manifest = json.loads(payload)
    if manifest.get("schema") not in DENSE_SCHEMAS or manifest.get("status") != "COMPLETED":
        raise ValueError("completed v4 dense surface manifest required")
    return manifest


def apply_dense_overlay(geometry, manifest_path, expected_sha256=None):
    """Replace only identities present in the manifest; verify every payload."""
    manifest = _read_dense_manifest(manifest_path, expected_sha256)
    root = Path(manifest_path).parent
    replaced = 0
    for row in manifest["objects"]:
        key = row["key"]
        if key not in geometry:
            continue
        payload = (root / row["path"]).read_bytes()
        if ((manifest['schema'].endswith('.v2') or row.get('sha256'))
                and hashlib.sha256(payload).hexdigest() != row.get('sha256')):
            raise ValueError("dense surface payload SHA-256 mismatch: " + key)
        with np.load(io.BytesIO(payload), allow_pickle=False) as values:
            surface = values["surface"].copy()
            normals = values["normals"].copy()
            query = values["query"].copy() if "query" in values.files else None
        if (surface.shape != (SURFACE_POINTS, 3) or normals.shape != surface.shape
                or len(np.unique(surface, axis=0)) != SURFACE_POINTS
                or not np.isfinite(surface).all() or not np.isfinite(normals).all()
                or not np.allclose(np.linalg.norm(normals, axis=-1), 1., atol=1e-5)):
            raise ValueError("dense surface/normal identity mismatch: " + key)
        original_query = geometry[key].get('query')
        if (query is None or original_query is None
                or query.shape != (FLOW_POINTS, 3) or not np.isfinite(query).all()
                or not np.array_equal(query, original_query)):
            raise ValueError("dense flow query must preserve the original 32 identities: " + key)
        replacement = dict(surface=surface.astype(np.float32),
                           normals=normals.astype(np.float32), query=query.astype(np.float32))
        geometry[key] = replacement
        replaced += 1
    return replaced


def load_geometry(cache, env_objects, dense_manifest=None, dense_sha256=None):
    """Load one prepared 512/32/128 geometry per environment object."""
    meta = json.loads((Path(cache) / "manifest.json").read_text())
    records = meta.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("geometry cache manifest must contain records")
    by_code = {}
    for row in records:
        code = row.get("object_code")
        if code is None or "geometry" not in row:
            raise ValueError("geometry cache record requires object_code and geometry")
        by_code.setdefault(code, []).append(row)
    geometry, keys = {}, []
    for obj in env_objects:
        code = obj["object_code"] if isinstance(obj, dict) else str(obj)
        scale = float(obj.get("scale", 0.)) if isinstance(obj, dict) else None
        rows = by_code.get(code)
        if not rows:
            raise ValueError("no cached geometry for object: " + code)
        row = rows[0]
        if scale is not None:
            matches = [r for r in rows if np.isclose(float(r.get("scale", scale)), scale)]
            if matches:
                row = matches[0]
        key = code if "@" in code else f"{code}@{scale:g}" if scale is not None else code
        keys.append(key)
        if key in geometry:
            continue
        with np.load(row["geometry"], allow_pickle=False) as values:
            surface = values["surface"].copy()
            normals = values["normals"].copy()
            query = values["query"].copy() if "query" in values.files else None
        if (query is None or query.shape != (FLOW_POINTS, 3)
                or not np.isfinite(query).all() or len(np.unique(query, axis=0)) != FLOW_POINTS):
            raise ValueError('online cache requires the original 32 flow queries: ' + key)
        length = np.linalg.norm(normals, axis=-1)
        keep = length > 1e-6
        if not keep.all():
            surface, normals = surface[keep], normals[keep] / length[keep, None]
        entry = dict(surface=surface.astype(np.float32), normals=normals.astype(np.float32))
        if query is not None:
            entry["query"] = query.astype(np.float32)
        geometry[key] = entry
    if dense_manifest is not None:
        apply_dense_overlay(geometry, dense_manifest, dense_sha256)
    prepared = []
    for key in keys:
        try:
            prepared.append(prepare_v4_geometry(geometry[key]))
        except ValueError as error:
            raise ValueError(f"online geometry {key} does not satisfy v4 512/32/128: {error}") from error
    return prepared


def v4_online_samples(objects, hands, geometries, *, goal_delta=None, goal_world=None):
    """Build target-only v4 samples from the observed 30 Hz H4 snapshot.

    ``objects`` is ``B x 4 x 7`` and ``hands`` is ``B x 4 x 11 x 3`` in the
    current action yaw history.  ``geometries`` is one prepared
    ``(surface, normals, flow, interaction, interaction_normals)`` tuple per
    environment.  ``goal_delta`` is the scheduler's same-ID point displacement
    in the current object frame; ``goal_world`` is the position-only task goal
    used by the existing online path.
    """
    objects = np.asarray(objects, dtype=np.float64)
    hands = np.asarray(hands, dtype=np.float64)
    if objects.ndim != 3 or objects.shape[-1] != 7 or objects.shape[1] != 4:
        raise ValueError("objects must be Bx4x7 at the 30 Hz history contract")
    if hands.ndim != 4 or hands.shape[1:] != (4, 11, 3):
        raise ValueError("hands must be Bx4x11x3")
    if len(geometries) != len(objects) or len(hands) != len(objects):
        raise ValueError("one geometry and hand history per environment")
    poses = object_poses(objects)
    now = poses[:, -1]
    rotation, center = now[:, :3, :3], now[:, :3, 3]
    local_hand = np.einsum("bhki,bij->bhkj", hands - center[:, None, None], rotation)
    relative = np.einsum("bij,bhjk->bhik", np.linalg.inv(now), poses)
    hand_dynamics = _dynamics(local_hand)
    if goal_delta is None:
        if goal_world is None:
            raise ValueError("v4 online input requires goal_delta or goal_world")
        goal_world = np.asarray(goal_world, dtype=np.float64)
        if goal_world.shape != (len(objects), 3) or not np.isfinite(goal_world).all():
            raise ValueError("goal_world must be Bx3 and finite")
        delta = np.einsum("bi,bij->bj", goal_world - center, rotation)
    else:
        delta = np.asarray(goal_delta, dtype=np.float64)
        if delta.shape != (len(objects), FLOW_POINTS, 3) or not np.isfinite(delta).all():
            raise ValueError("goal_delta must be Bx32x3 and finite")
    samples = []
    for env, (surface, normals, flow, interaction, interaction_normals) in enumerate(geometries):
        if surface.shape != (SURFACE_POINTS, 3) or normals.shape != surface.shape:
            raise ValueError("online v4 surface must be 512x3 with aligned normals")
        moved = np.einsum("hij,pj->hpi", relative[env, :, :3, :3], surface)
        moved += relative[env, :, None, :3, 3]
        features = np.zeros((SURFACE_POINTS, 18), np.float32)
        features[:, :3] = _normal_transform(normals, relative[env, -1, :3, :3])
        features[:, 3:12] = _dynamics(moved[None])[0]
        features[:, FEATURE_TARGET] = 1.
        features[:, FEATURE_STATIC] = scene_target_static(features[:, 9:12])
        start_hand = np.zeros((2, 11, 3), np.float32)
        start_hand[0] = local_hand[env, -1]
        start_history = np.zeros((2, 11, 9), np.float32)
        start_history[0] = hand_dynamics[env]
        start_valid = np.zeros((2, 11), bool)
        start_valid[0] = True
        goal = np.broadcast_to(delta[env], flow.shape).astype(np.float32)
        sample = dict(
            xyz=moved[-1].astype(np.float32), features=features,
            scene_object=np.zeros(SURFACE_POINTS, np.int64),
            point_valid=np.ones(SURFACE_POINTS, bool), object_valid=np.ones(1, bool),
            scene_points=flow[None].astype(np.float32),
            start_object_points=flow.astype(np.float32),
            goal_object_delta=goal.copy(), object_goal_valid=np.ones(FLOW_POINTS, bool),
            start_hand=start_hand, start_hand_history=start_history,
            start_hand_valid=start_valid,
            interaction_points=interaction[None].astype(np.float32),
            interaction_normals=interaction_normals[None].astype(np.float32),
            interaction_query_valid=np.ones((1, INTERACTION_POINTS), bool),
        )
        for key, value in sample.items():
            if not np.isfinite(value).all():
                raise ValueError("nonfinite online v4 field: " + key)
        samples.append(sample)
    return samples


def collate_v4(samples):
    if not samples:
        raise ValueError("empty v4 online batch")
    out = {}
    for key, value in samples[0].items():
        values = [torch.as_tensor(sample[key]) for sample in samples]
        out[key] = torch.stack(values)
    return out


def summary_from_prediction(prediction):
    """Explicit 2304-D shared summary; it does not consume tau or C."""
    from .summary_helpers import type_summaries
    if "shared_tokens" not in prediction or "shared_valid" not in prediction:
        raise ValueError("prediction must be built with return_shared=True")
    return type_summaries(prediction["shared_tokens"], prediction["shared_valid"],
                          int(prediction["scene_token_count"]), 22)


class V4Runtime:
    """Load one v4 checkpoint and expose explicit online inference modes."""

    def __init__(self, checkpoint, env_objects, cache, *, dense_manifest=None,
                 dense_sha256=None, device="cuda:0", model_factory=None):
        packet = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if packet.get("schema") != V4_SCHEMA:
            raise ValueError("v4 runtime requires ref2dex.ptv3-shared-wam.v4")
        self.config, self.stats = packet["config"], packet["stats"]
        self.step = int(packet.get("step", 0))
        recipe = self.stats.get('recipe', {})
        bound_digest = recipe.get('surface_geometry_sha256')
        if bound_digest is not None:
            if dense_sha256 is not None and dense_sha256 != bound_digest:
                raise ValueError('checkpoint and online dense geometry identities differ')
            dense_sha256 = bound_digest
            if dense_manifest is None:
                dense_manifest = recipe.get('surface_geometry_manifest')
            if dense_manifest is None:
                raise ValueError('checkpoint-bound dense geometry requires its surface manifest')
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if model_factory is None:
            v4_src = TASK_ROOT.parent / "ptv3-shared-wam-v4" / "src"
            if str(v4_src) not in sys.path:
                sys.path.insert(0, str(v4_src))
            from ptv3_shared_wam_v4.model import SharedWAMV4
            model_factory = SharedWAMV4
        model = model_factory(self.stats, self.config)
        model.load_state_dict(packet["model"], strict=True)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.geometry = load_geometry(cache, env_objects, dense_manifest, dense_sha256)
        if len(self.geometry) != len(env_objects):
            raise ValueError("one prepared v4 geometry per environment required")

    def batch(self, objects, hands, *, goal_delta=None, goal_world=None):
        samples = v4_online_samples(np.asarray(objects, dtype=np.float64),
                                    np.asarray(hands, dtype=np.float64),
                                    self.geometry, goal_delta=goal_delta, goal_world=goal_world)
        return {key: value.to(self.device) for key, value in collate_v4(samples).items()}

    @torch.inference_mode()
    def predict(self, batch):
        return self.model(batch, return_shared=True)

    @torch.inference_mode()
    def summary(self, batch):
        shared, valid, scene = self.model.encode(batch)
        from .summary_helpers import type_summaries
        return type_summaries(shared, valid, scene["tokens"].shape[1], 22)

    @torch.inference_mode()
    def predict_with_summary(self, batch):
        return self.predict(batch)

    @torch.inference_mode()
    def tau_for_student(self, prediction, object_pose, *, hand_slot="right",
                        student_horizon=8, frame="world"):
        """Explicitly bridge v4 tau to the existing 60 Hz student contract."""
        from .tau_contract import v4_trajectory_to_student
        trajectory = prediction["trajectory"]
        if trajectory.ndim != 5 or trajectory.shape[0] != 1:
            raise ValueError("one v4 sample trajectory required")
        rotation = None
        if frame == "world":
            rotation = np.asarray(object_pose, dtype=np.float64)[:3, :3]
        return v4_trajectory_to_student(
            trajectory[0].detach().cpu().numpy(), object_rotation=rotation,
            hand_slot=hand_slot, student_horizon=student_horizon, frame=frame)

    @torch.inference_mode()
    def interaction_for_executor(self, prediction):
        """Return the first sample's ``C[objects,24,128,7]`` and validity mask."""
        if "interaction" not in prediction or "interaction_prediction_valid" not in prediction:
            raise ValueError("full v4 forward output required")
        return (prediction["interaction"][0].detach().cpu().numpy(),
                prediction["interaction_prediction_valid"][0].detach().cpu().numpy())
