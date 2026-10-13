"""U=D_U(Z), V=D_V(Z,G); Z sees only observed scene and semantic hands."""
import torch
from torch import nn
from .contracts import GOAL_ENCODING

HORIZON, HANDS, KEYPOINTS, HIDDEN = 24, 2, 11, 128


def clean(values, mask):
    values = torch.where(mask[..., None], values.float(), 0.)
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite valid observation")
    return values


class HandTokenEncoder(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(12, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.hand_id = nn.Embedding(HANDS, HIDDEN)
        self.keypoint_id = nn.Embedding(KEYPOINTS, HIDDEN)
        self.register_buffer("history_mean", torch.tensor(stats["scene_mean"][3:12]))
        self.register_buffer("history_std", torch.tensor(stats["scene_std"][3:12]).clamp_min(1e-6))

    def forward(self, batch):
        valid = batch["start_hand_valid"].bool()
        xyz = clean(batch["start_hand"], valid)
        history = clean(batch["start_hand_history"], valid)
        x = torch.cat((xyz / .05, (history - self.history_mean) / self.history_std), -1)
        identity = self.hand_id.weight[:, None] + self.keypoint_id.weight[None]
        token = self.mlp(x) + identity[None]
        return torch.where(valid[..., None], token, 0.).reshape(len(xyz), 22, HIDDEN), valid.flatten(1)


class GoalConditioner(nn.Module):
    """32 [current point, t+24 displacement] tokens condition only the hands."""
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(6, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.target = nn.Parameter(torch.zeros(HIDDEN))
        self.null = nn.Parameter(torch.zeros(1, 1, HIDDEN))
        self.attention = nn.MultiheadAttention(HIDDEN, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, hand, batch):
        valid = batch["object_goal_valid"].bool()
        start = clean(batch["start_object_points"], valid)
        delta = clean(batch["goal_object_delta"], valid)
        if start.shape != delta.shape or start.shape[:-1] != valid.shape:
            raise ValueError("current points and goal delta must be aligned")
        token = self.point(torch.cat((start, delta), -1) / .05) + self.target
        token = torch.where(valid[..., None], token, 0.)
        token = torch.cat((token, self.null.expand(len(hand), -1, -1)), 1)
        mask = torch.cat((valid, torch.ones(len(hand), 1, dtype=torch.bool, device=hand.device)), 1)
        update, _ = self.attention(hand, token, token, key_padding_mask=~mask, need_weights=False)
        # Null-goal human/video support has no goal-conditioned residual.
        update = torch.where(valid.any(1)[:, None, None], update, 0.)
        return self.norm(hand + update)


class FutureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = nn.Embedding(HORIZON, HIDDEN)
        self.mlp = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, HIDDEN),
                                 nn.GELU(), nn.Linear(HIDDEN, 3))
        # Small nonzero output allows BOTH tasks to reach the backbone at step1.
        nn.init.normal_(self.mlp[-1].weight, std=.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens):
        return self.mlp(tokens[:, None] + self.time.weight[None, :, None]).float() * .05


class SceneFutureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(3, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.readout = nn.MultiheadAttention(HIDDEN, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(HIDDEN)
        self.future = FutureHead()

    def forward(self, shared, shared_valid, points, object_valid):
        count, objects, point_count, _ = points.shape
        valid = object_valid[..., None].expand(-1, -1, point_count)
        # Current point coordinates identify readout locations; all learned
        # interaction context comes from Z, with no pre-trunk feature bypass.
        query = self.position(clean(points, valid) / .05)
        query = torch.where(valid[..., None], query, 0.).reshape(count, -1, HIDDEN)
        update, _ = self.readout(query, shared, shared, key_padding_mask=~shared_valid,
                                 need_weights=False)
        output = self.future(self.norm(query + update))
        output = output.reshape(count, HORIZON, objects, point_count, 3).transpose(1, 2)
        return torch.where(valid[:, :, None, :, None], output, 0.)


class SharedWAM(nn.Module):
    def __init__(self, stats, config, scene_encoder=None):
        super().__init__()
        if config.get("hidden", HIDDEN) != HIDDEN or config.get("horizon", 24) != 24:
            raise ValueError("PTv3-small128 and future24 required")
        if config.get("goal_encoding", GOAL_ENCODING) != GOAL_ENCODING:
            raise ValueError("current-point/horizon-delta goal contract required")
        if scene_encoder is None:
            from .scene import SceneEncoder
            scene_encoder = SceneEncoder(stats, config)
        self.scene = scene_encoder
        self.hand = HandTokenEncoder(stats)
        self.position = nn.Sequential(nn.Linear(3, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.token_type = nn.Embedding(2, HIDDEN)
        layer = nn.TransformerEncoderLayer(HIDDEN, 4, 512, dropout=0., activation="gelu",
                                           batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, config.get("interaction_layers", 4),
                                                 enable_nested_tensor=False)
        self.goal = GoalConditioner()
        self.hand_future = FutureHead()
        self.scene_future = SceneFutureHead()

    def encode(self, batch):
        # No future labels, goals, q/dq or native commands enter either encoder.
        scene_keys = ("xyz", "features", "point_valid", "scene_object", "object_valid", "scene_points")
        scene_inputs = {key: batch[key] for key in scene_keys}
        point_valid = batch["object_valid"][..., None].expand(*batch["scene_points"].shape[:-1])
        scene_inputs["scene_points"] = clean(batch["scene_points"], point_valid)
        scene = self.scene(scene_inputs)
        hand, hand_valid = self.hand(batch)
        scene_tokens = scene["tokens"] + self.position(clean(scene["xyz"], scene["mask"]) / .05)
        scene_tokens = scene_tokens + self.token_type.weight[0]
        hand = hand + self.token_type.weight[1]
        valid = torch.cat((scene["mask"], hand_valid), 1)
        tokens = torch.cat((scene_tokens, hand), 1)
        tokens = torch.where(valid[..., None], tokens, 0.)
        shared = self.interaction(tokens, src_key_padding_mask=~valid)
        shared = torch.where(valid[..., None], shared, 0.)
        return shared, valid, scene

    def forward(self, batch):
        shared, valid, scene = self.encode(batch)
        scene_prediction = self.scene_future(shared, valid, batch["scene_points"], batch["object_valid"])
        hand = self.goal(shared[:, -22:], batch)
        prediction = self.hand_future(hand).reshape(len(hand), 24, 2, 11, 3)
        prediction = torch.where(batch["start_hand_valid"][:, None, ..., None], prediction, 0.)
        return dict(scene=scene_prediction, trajectory=prediction)

    def parameter_groups(self, scene_lr=1e-5, new_lr=1e-4):
        pretrained = list(self.scene.parameters())
        ids = {id(p) for p in pretrained}
        return [dict(params=pretrained, lr=scene_lr, name="ptv3_scene"),
                dict(params=[p for p in self.parameters() if id(p) not in ids],
                     lr=new_lr, name="shared_and_heads")]
