"""Goal-Point-WAM model with a goal-blind executed-action world branch."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn


CM_SRC = Path(__file__).resolve().parents[2].parent / "cm-pointflow-effect-pretrain" / "src"
if str(CM_SRC) not in sys.path:
    sys.path.insert(0, str(CM_SRC))

from oakink_wm.model import geodesic, mean_groups, rigid_points, rotation6d  # noqa: E402
from oakink_wm.pointworld import (  # noqa: E402
    DEFAULT_VOXEL_ORIGIN_M,
    DEFAULT_VOXEL_SIZE_M,
    VENDOR,
    fixed_workspace_grid,
)
from ptv3.ptv3 import PointTransformerV3  # noqa: E402


HORIZON = 24
HANDS = 2
KEYPOINTS = 11
HAND_POINTS = HANDS * KEYPOINTS
HIDDEN = 128


def _sparse_context(tensor: torch.Tensor):
    if tensor.is_cuda:
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def _masked_mean(values: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weight = valid.to(values.dtype)
    return (values * weight.unsqueeze(-1)).sum(dim) / weight.sum(dim, keepdim=True).clamp_min(1.0)


class SceneHistoryEncoder(nn.Module):
    """Encode only observed current/history scene points into fixed spatial tokens."""

    def __init__(self, stats: dict, patch_size: int = 128, token_count: int = 256,
                 voxel_origin_m=DEFAULT_VOXEL_ORIGIN_M, voxel_size_m=DEFAULT_VOXEL_SIZE_M):
        super().__init__()
        if token_count < 1 or patch_size < 1:
            raise ValueError("token_count and patch_size must be positive")
        self.token_count = int(token_count)
        self.voxel_size_m = float(voxel_size_m)
        self.register_buffer("voxel_origin_m", torch.as_tensor(voxel_origin_m, dtype=torch.float32),
                             persistent=False)
        cfg = yaml.safe_load((VENDOR / "ptv3/ptv3_arch.yaml").read_text())["sizes"]["small"]
        channels = lambda key: tuple(HIDDEN if x == "channels" else x for x in cfg[key])
        self.backbone = PointTransformerV3(
            in_channels=HIDDEN, order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2,) * 4, enc_depths=tuple(cfg["enc_depths"]),
            enc_channels=channels("enc_channels"), enc_num_head=tuple(cfg["enc_num_head"]),
            enc_patch_size=(patch_size,) * 5, dec_depths=tuple(cfg["dec_depths"]),
            dec_channels=channels("dec_channels"), dec_num_head=tuple(cfg["dec_num_head"]),
            dec_patch_size=(patch_size,) * 4, drop_path=.3, shuffle_orders=True,
            enable_rpe=False, traceable=True, mask_token=False, enc_mode=False,
        )
        self.scene_proj = nn.Sequential(nn.Linear(18, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.object_proj = nn.Sequential(nn.LayerNorm(15), nn.Linear(15, HIDDEN), nn.GELU(),
                                         nn.Linear(HIDDEN, HIDDEN))
        self.register_buffer("scene_mean", torch.as_tensor(stats["scene_mean"], dtype=torch.float32))
        self.register_buffer("scene_std", torch.as_tensor(stats["scene_std"], dtype=torch.float32).clamp_min(1e-6))

    def train(self, mode: bool = True):
        super().train(mode)
        # Evaluation must use the same serialization order for batch-invariance
        # checks; training retains the upstream order randomization.
        for module in self.backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = mode
        return self

    def _sample_tokens(self, packed: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = packed.shape
        tokens = packed.new_zeros((batch, self.token_count, dim))
        mask = torch.zeros((batch, self.token_count), dtype=torch.bool, device=packed.device)
        for row in range(batch):
            ids = torch.nonzero(valid[row], as_tuple=False).flatten()
            if not len(ids):
                raise ValueError("scene encoder received no valid point")
            if len(ids) > self.token_count:
                choice = torch.linspace(0, len(ids) - 1, self.token_count, device=ids.device)
                ids = ids[choice.round().long()]
            count = min(len(ids), self.token_count)
            tokens[row, :count] = packed[row, ids[:count]]
            mask[row, :count] = True
        return tokens, mask

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        xyz = batch["xyz"]
        valid = batch["point_valid"].bool()
        scene_object = batch["scene_object"].long()
        object_valid = batch["object_valid"].bool()
        object_count = object_valid.shape[1]
        features = (batch["features"].float() - self.scene_mean) / self.scene_std
        projected = self.scene_proj(features)
        ids = torch.arange(len(xyz), device=xyz.device)[:, None].expand_as(valid)
        with _sparse_context(xyz):
            coords = xyz.float()
            inputs = projected.float()
            grid_all = fixed_workspace_grid(coords, valid, self.voxel_origin_m, self.voxel_size_m)
            keys = torch.cat((ids[valid][:, None], grid_all[valid]), dim=-1)
            unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
            point = self.backbone(dict(
                coord=mean_groups(coords[valid], inverse, len(unique)),
                feat=mean_groups(inputs[valid], inverse, len(unique)),
                grid_coord=unique[:, 1:].int(), batch=unique[:, 0], grid_size=self.voxel_size_m,
            ))
        packed = point.feat.new_zeros((*valid.shape, HIDDEN))
        packed[valid] = point.feat[inverse]
        scene_tokens, scene_mask = self._sample_tokens(packed, valid)
        object_points = valid & (scene_object >= 0)
        group = (ids * object_count + scene_object)[object_points]
        pooled = mean_groups(packed[object_points], group, len(xyz) * object_count)
        pooled = pooled.reshape(len(xyz), object_count, HIDDEN)
        object_tokens = pooled + self.object_proj(batch["object_features"].float())
        return dict(scene_tokens=scene_tokens, scene_mask=scene_mask,
                    object_tokens=object_tokens, object_mask=object_valid)


class GoalEncoder(nn.Module):
    """Encode start/end point sets for the trajectory branch only."""

    def __init__(self, hidden: int = HIDDEN, layers: int = 2):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.hand = nn.Embedding(HANDS, hidden)
        self.keypoint = nn.Embedding(KEYPOINTS, hidden)
        self.goal_type = nn.Embedding(2, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.null = nn.Parameter(torch.zeros(1, 1, hidden))

    def forward(self, start: torch.Tensor, goal: torch.Tensor, valid: torch.Tensor):
        batch = start.shape[0]
        points = torch.stack((start, goal), dim=1).reshape(batch, 2, HANDS, KEYPOINTS, 3)
        mask = valid.reshape(batch, 2, HANDS, KEYPOINTS)
        hand = torch.arange(HANDS, device=start.device)[None, None, :, None]
        keypoint = torch.arange(KEYPOINTS, device=start.device)[None, None, None, :]
        kind = torch.arange(2, device=start.device)[None, :, None, None]
        tokens = self.point(points)
        tokens = (tokens + self.hand(hand) + self.keypoint(keypoint) + self.goal_type(kind))
        tokens = tokens.reshape(batch, 2 * HAND_POINTS, -1)
        mask = mask.reshape(batch, 2 * HAND_POINTS)
        tokens = torch.cat((tokens, self.null.expand(batch, -1, -1)), dim=1)
        mask = torch.cat((mask, torch.ones((batch, 1), dtype=torch.bool, device=start.device)), dim=1)
        tokens = self.encoder(tokens, src_key_padding_mask=~mask)
        return tokens, mask


class TrajectoryDecoder(nn.Module):
    def __init__(self, hidden: int = HIDDEN, layers: int = 3):
        super().__init__()
        self.time = nn.Embedding(HORIZON, hidden)
        self.hand = nn.Embedding(HANDS, hidden)
        self.keypoint = nn.Embedding(KEYPOINTS, hidden)
        layer = nn.TransformerDecoderLayer(hidden, 4, hidden * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, scene: torch.Tensor, scene_mask: torch.Tensor,
                goal: torch.Tensor, goal_mask: torch.Tensor) -> torch.Tensor:
        batch = scene.shape[0]
        memory = torch.cat((scene, goal), dim=1)
        memory_mask = torch.cat((scene_mask, goal_mask), dim=1)
        time = torch.arange(HORIZON, device=scene.device)[:, None, None]
        hand = torch.arange(HANDS, device=scene.device)[None, :, None]
        keypoint = torch.arange(KEYPOINTS, device=scene.device)[None, None, :]
        query = (self.time(time) + self.hand(hand) + self.keypoint(keypoint)).reshape(1, -1, HIDDEN)
        query = query.expand(batch, -1, -1)
        decoded = self.decoder(query, memory, memory_key_padding_mask=~memory_mask)
        # Physical/masked losses run outside autocast; keep their prediction
        # tensors in float32 so CUDA has a stable backward path.
        return self.head(decoded).float().reshape(batch, HORIZON, HANDS, KEYPOINTS, 3)


class ExecutedActionAdapter(nn.Module):
    """Encode the observed/teacher-executed trajectory for the world branch."""

    def __init__(self, stats: dict, hidden: int = HIDDEN):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.time = nn.Embedding(HORIZON, hidden)
        self.hand = nn.Embedding(HANDS, hidden)
        self.keypoint = nn.Embedding(KEYPOINTS, hidden)
        self.register_buffer("action_mean", torch.as_tensor(stats["action_mean"], dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(stats["action_std"], dtype=torch.float32).clamp_min(1e-6))

    def forward(self, action: torch.Tensor, valid: torch.Tensor):
        normalized = (action.float() - self.action_mean) / self.action_std
        normalized = torch.where(valid[..., None], normalized, torch.zeros_like(normalized))
        time = torch.arange(HORIZON, device=action.device)[:, None, None]
        hand = torch.arange(HANDS, device=action.device)[None, :, None]
        keypoint = torch.arange(KEYPOINTS, device=action.device)[None, None, :]
        encoded = (self.proj(normalized) + self.time(time) + self.hand(hand) + self.keypoint(keypoint))
        encoded = torch.where(valid[..., None], encoded, torch.zeros_like(encoded))
        return (encoded.reshape(len(action), HORIZON * HAND_POINTS, HIDDEN),
                valid.reshape(len(action), HORIZON * HAND_POINTS))


class WorldDecoder(nn.Module):
    """Decode rigid object effects without any goal tokens."""

    def __init__(self, stats: dict, hidden: int = HIDDEN, layers: int = 2):
        super().__init__()
        self.time = nn.Embedding(HORIZON, hidden)
        layer = nn.TransformerDecoderLayer(hidden, 4, hidden * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, 9))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        for key in ("flow_mean", "flow_std", "translation_mean", "translation_std", "rotation_scale"):
            self.register_buffer(key, torch.as_tensor(stats[key], dtype=torch.float32))

    def forward(self, scene: torch.Tensor, scene_mask: torch.Tensor,
                objects: torch.Tensor, object_mask: torch.Tensor,
                action_tokens: torch.Tensor, action_mask: torch.Tensor):
        batch, objects_count = objects.shape[:2]
        memory = torch.cat((scene, action_tokens), dim=1)
        memory_mask = torch.cat((scene_mask, action_mask), dim=1)
        time = self.time.weight[None, None].expand(batch, objects_count, -1, -1)
        query = objects[:, :, None] + time
        query = query.reshape(batch, objects_count * HORIZON, HIDDEN)
        query_mask = (~object_mask)[:, :, None].expand(batch, objects_count, HORIZON).reshape(batch, -1)
        decoded = self.decoder(query, memory, tgt_key_padding_mask=query_mask,
                               memory_key_padding_mask=~memory_mask)
        raw = self.head(decoded).reshape(batch, objects_count, HORIZON, 9).float()
        translation = raw[..., :3] * self.translation_std + self.translation_mean
        identity = raw.new_tensor([1., 0., 0., 0., 1., 0.])
        rotation = rotation6d(identity + raw[..., 3:])
        return dict(translation=translation, rotation=rotation)


class GoalPointWAM(nn.Module):
    """Three-domain scene/trajectory/world learner for the no-video phase."""

    def __init__(self, stats: dict, config: dict):
        super().__init__()
        hidden = int(config.get("hidden", HIDDEN))
        if hidden != HIDDEN:
            raise ValueError("the PTv3-small contract currently requires hidden=128")
        self.scene = SceneHistoryEncoder(
            stats, patch_size=int(config.get("patch_size", 128)),
            token_count=int(config.get("scene_tokens", 256)),
            voxel_origin_m=config.get("voxel_origin_m", DEFAULT_VOXEL_ORIGIN_M),
            voxel_size_m=float(config.get("voxel_m", DEFAULT_VOXEL_SIZE_M)),
        )
        self.goal = GoalEncoder(hidden, int(config.get("goal_layers", 2)))
        self.trajectory = TrajectoryDecoder(hidden, int(config.get("trajectory_layers", 3)))
        self.action = ExecutedActionAdapter(stats, hidden)
        self.world = WorldDecoder(stats, hidden, int(config.get("world_layers", 2)))
        self.trajectory_scale_m = float(config.get("trajectory_scale_m", .05))
        self.world_weight = float(config.get("world_weight", 1.0))
        self.motion_tau_m = float(config.get("motion_tau_m", .002))
        self.rotation_tau_rad = float(config.get("rotation_tau_rad", .02))
        self.motion_temperature = float(config.get("motion_temperature", 5.0))
        self.motion_floor = float(config.get("motion_floor", .1))
        if self.trajectory_scale_m <= 0 or not 0 < self.motion_floor <= 1:
            raise ValueError("invalid loss scales")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        scene = self.scene(batch)
        goal_input = torch.cat((batch["start_hand_valid"], batch["goal_hand_valid"]), dim=1)
        goal_tokens, goal_mask = self.goal(batch["start_hand"], batch["goal_hand"], goal_input)
        trajectory = self.trajectory(scene["scene_tokens"], scene["scene_mask"], goal_tokens, goal_mask)
        action = batch["action"].reshape(-1, HORIZON, HANDS, KEYPOINTS, 9)
        action_valid = batch["action_valid"].reshape(-1, HORIZON, HANDS, KEYPOINTS)
        action_tokens, action_mask = self.action(action, action_valid)
        world = self.world(scene["scene_tokens"], scene["scene_mask"], scene["object_tokens"],
                           scene["object_mask"], action_tokens, action_mask)
        return dict(trajectory=trajectory, translation=world["translation"], rotation=world["rotation"])

    def trajectory_loss(self, pred: torch.Tensor, batch: dict[str, torch.Tensor]):
        valid = batch["trajectory_valid"].bool()
        scale = self.trajectory_scale_m
        error = F.huber_loss(pred / scale, batch["trajectory_target"].float() / scale,
                             reduction="none").mean(-1)
        denom = valid.sum().clamp_min(1).to(error.dtype)
        return (error * valid).sum() / denom

    def motion_weights(self, batch: dict[str, torch.Tensor], actual: torch.Tensor):
        gt = batch["effect"].float()
        displacement = torch.linalg.vector_norm(actual - batch["points"][:, :, None], dim=-1)
        identity = torch.eye(3, device=gt.device, dtype=gt.dtype).expand_as(gt[..., :3, :3])
        angle = geodesic(identity, gt[..., :3, :3])
        magnitude = torch.maximum(displacement / self.motion_tau_m,
                                  angle[..., None] / self.rotation_tau_rad)
        weight = self.motion_floor + (1 - self.motion_floor) * torch.sigmoid(
            self.motion_temperature * (magnitude - 1))
        return weight * batch["object_valid"][:, :, None, None].to(weight.dtype)

    def world_loss(self, pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        gt = batch["effect"].float()
        actual = rigid_points(gt[..., :3, :3], gt[..., :3, 3], batch["points"].float())
        predicted = rigid_points(pred["rotation"], pred["translation"], batch["points"].float())
        weight = self.motion_weights(batch, actual)
        target = (actual - batch["points"][:, :, None] - self.world.flow_mean[None, None, :, None])
        target = target / self.world.flow_std[None, None, :, None]
        estimate = (predicted - batch["points"][:, :, None] - self.world.flow_mean[None, None, :, None])
        estimate = estimate / self.world.flow_std[None, None, :, None]
        point = (F.huber_loss(estimate, target, reduction="none").mean(-1) * weight).sum()
        point = point / weight.sum().clamp_min(1e-6)
        object_weight = weight.mean(-1)
        object_norm = object_weight.sum().clamp_min(1e-6)
        trans = F.huber_loss(
            (pred["translation"] - self.world.translation_mean) / self.world.translation_std,
            (gt[..., :3, 3] - self.world.translation_mean) / self.world.translation_std,
            reduction="none",
        ).mean(-1)
        angle = geodesic(pred["rotation"], gt[..., :3, :3]) / self.world.rotation_scale
        rot = F.huber_loss(angle, torch.zeros_like(angle), reduction="none")
        terms = dict(point=point, translation=(trans * object_weight).sum() / object_norm,
                     rotation=(rot * object_weight).sum() / object_norm)
        return sum(terms.values()), terms

    def loss(self, prediction: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]):
        trajectory = self.trajectory_loss(prediction["trajectory"], batch)
        world, world_terms = self.world_loss(prediction, batch)
        total = trajectory + self.world_weight * world
        terms = {"trajectory": trajectory.detach(), "world": world.detach()}
        terms.update({"world_" + key: value.detach() for key, value in world_terms.items()})
        return total, terms


__all__ = ["GoalPointWAM", "SceneHistoryEncoder", "GoalEncoder"]
