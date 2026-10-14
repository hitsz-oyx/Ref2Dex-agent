"""Z=F(S,H,G), U=D_U(Z), V=D_V(Z): shared task-conditioned forecasting."""
import torch
from torch import nn
from .contracts import GOAL_ENCODING, TASK_CONDITIONING

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


class GoalTokenEncoder(nn.Module):
    """Paired [current point, t+24 displacement] tokens for the shared trunk."""
    def __init__(self):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(6, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.target = nn.Parameter(torch.zeros(HIDDEN))

    def forward(self, batch):
        valid = batch["object_goal_valid"].bool()
        start = clean(batch["start_object_points"], valid)
        delta = clean(batch["goal_object_delta"], valid)
        if start.shape != delta.shape or start.shape[:-1] != valid.shape:
            raise ValueError("current points and goal delta must be aligned")
        token = self.point(torch.cat((start, delta), -1) / .05) + self.target
        token = torch.where(valid[..., None], token, 0.)
        return token, valid


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
        if config.get("task_conditioning", TASK_CONDITIONING) != TASK_CONDITIONING:
            raise ValueError("shared scene/hand/goal conditioning required")
        if scene_encoder is None:
            from .scene import SceneEncoder
            scene_encoder = SceneEncoder(stats, config)
        self.scene = scene_encoder
        self.hand = HandTokenEncoder(stats)
        self.position = nn.Sequential(nn.Linear(3, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN))
        self.token_type = nn.Embedding(3, HIDDEN)
        self.interaction_width = int(config.get("interaction_width", HIDDEN))
        self.checkpoint_interaction = bool(config.get("checkpoint_interaction",False))
        if self.interaction_width < HIDDEN or self.interaction_width % HIDDEN:
            raise ValueError("interaction width must be an integer multiple of128")
        width = self.interaction_width
        layer = nn.TransformerEncoderLayer(width, width // 32, width * 4, dropout=0., activation="gelu",
                                           batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, config.get("interaction_layers", 4),
                                                 enable_nested_tensor=False)
        self.goal = GoalTokenEncoder()
        self.hand_future = FutureHead()
        self.scene_future = SceneFutureHead()

    def encode(self, batch):
        # PTv3 sees observations only; endpoint goals enter as separate tokens.
        scene_keys = ("xyz", "features", "point_valid", "scene_object", "object_valid", "scene_points")
        scene_inputs = {key: batch[key] for key in scene_keys}
        point_valid = batch["object_valid"][..., None].expand(*batch["scene_points"].shape[:-1])
        scene_inputs["scene_points"] = clean(batch["scene_points"], point_valid)
        scene = self.scene(scene_inputs)
        hand, hand_valid = self.hand(batch)
        goal, goal_valid = self.goal(batch)
        scene_tokens = scene["tokens"] + self.position(clean(scene["xyz"], scene["mask"]) / .05)
        scene_tokens = scene_tokens + self.token_type.weight[0]
        hand = hand + self.token_type.weight[1]
        goal = goal + self.token_type.weight[2]
        valid = torch.cat((scene["mask"], hand_valid, goal_valid), 1)
        tokens = torch.cat((scene_tokens, hand, goal), 1)
        tokens = torch.where(valid[..., None], tokens, 0.)
        repeat = self.interaction_width // HIDDEN
        shared = tokens if repeat == 1 else tokens.repeat(1, 1, repeat)
        if self.checkpoint_interaction and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            for layer in self.interaction.layers:
                shared = checkpoint(layer,shared,src_key_padding_mask=~valid,use_reentrant=False,
                                    preserve_rng_state=False)  # these layers have dropout=0
        else:
            shared = self.interaction(shared, src_key_padding_mask=~valid)
        # Integer replication/mean bridges retain the trained128D heads while
        # permitting a wider shared representation without new input features.
        shared = shared.reshape(*shared.shape[:-1], repeat, HIDDEN).mean(-2)
        shared = torch.where(valid[..., None], shared, 0.)
        return shared, valid, scene

    def forward(self, batch):
        shared, valid, scene = self.encode(batch)
        scene_prediction = self.scene_future(shared, valid, batch["scene_points"], batch["object_valid"])
        scene_count = scene["tokens"].shape[1]
        hand = shared[:, scene_count:scene_count + HANDS * KEYPOINTS]
        prediction = self.hand_future(hand).reshape(len(hand), 24, 2, 11, 3)
        prediction = torch.where(batch["start_hand_valid"][:, None, ..., None], prediction, 0.)
        return dict(scene=scene_prediction, trajectory=prediction)

    def parameter_groups(self, scene_lr=1e-5, new_lr=1e-4, interaction_lr=None,
                         new_layer_lr=None, expansion_source_layers=0):
        pretrained = list(self.scene.parameters())
        ids = {id(p) for p in pretrained}
        interaction = []
        if interaction_lr is not None:
            ids.update(id(p) for p in self.interaction.parameters())
            for name,layers,lr in (
                    ("transferred_interaction",list(self.interaction.layers)[:expansion_source_layers],interaction_lr),
                    ("new_interaction",list(self.interaction.layers)[expansion_source_layers:],new_layer_lr or interaction_lr)):
                params = [p for layer in layers for p in layer.parameters()]
                if params:interaction.append(dict(params=params,lr=lr,name=name))
        return [dict(params=pretrained, lr=scene_lr, name="ptv3_scene"),
                dict(params=[p for p in self.parameters() if id(p) not in ids],
                     lr=new_lr, name="shared_and_heads")] + interaction
