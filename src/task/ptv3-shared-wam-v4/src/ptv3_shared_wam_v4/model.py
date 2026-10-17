"""PTv3 shared WAM-v4: v3 hand/scene forecasts plus an interaction C head."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .paths import SCHEMA, HORIZON, INTERACTION_ENCODING
from .normalization import HorizonNormalizer
from .interaction import C_HORIZON, C_DIM, interaction_scales
from ptv3_shared_wam.model import clean
from ptv3_shared_wam.contracts import GOAL_ENCODING, TASK_CONDITIONING


def mlp(input_width, output_width):
    return nn.Sequential(nn.Linear(input_width, output_width), nn.GELU(), nn.Linear(output_width, output_width))


class TemporalHead(nn.Module):
    def __init__(self, input_width, width, output_dim, horizon=HORIZON):
        super().__init__()
        self.project = nn.Linear(input_width, width)
        self.time = nn.Embedding(horizon, width)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim))
        nn.init.normal_(self.mlp[-1].weight, std=.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens):
        return self.mlp(self.project(tokens)[:, None] + self.time.weight[None, :, None]).float()


class SharedWAMV4(nn.Module):
    """Goal-conditioned world/action model with ``(tau, C)`` forecasts."""
    def __init__(self, stats, config, geometry=None):
        super().__init__()
        if (config.get("schema") != SCHEMA or config.get("horizon") != HORIZON
                or config.get("goal_encoding") != GOAL_ENCODING
                or config.get("task_conditioning") != TASK_CONDITIONING):
            raise ValueError("independent v4 shared-goal future24 config required")
        if config.get("interaction_encoding", INTERACTION_ENCODING) != INTERACTION_ENCODING:
            raise ValueError("unsupported dynamic interaction graph encoding")
        if int(config.get("interaction_dim", C_DIM)) != C_DIM:
            raise ValueError("interaction_dim must be nine for the structured C contract")
        if int(config.get("interaction_horizon", C_HORIZON)) != C_HORIZON:
            raise ValueError("dynamic interaction graph must cover the 24-step horizon")
        self.config = config
        point_width, width, head = config["point_width"], config["shared_width"], config["head_width"]
        self.explicit_hand = config.get("explicit_hand", True)
        self.position_scale = config["position_scale_m"]
        scales = [self.position_scale]*3 + [config["velocity_scale_mps"]]*3
        scales += [config["acceleration_scale_mps2"]]*3 + [self.position_scale]*3
        if min(scales) <= 0:
            raise ValueError("fixed physical scales must be positive")
        self.register_buffer("hand_scale", torch.tensor(scales))
        self.hand_encoder = mlp(12, point_width)
        self.hand_id = nn.Embedding(2, point_width)
        self.keypoint_id = nn.Embedding(11, point_width)
        if geometry is None:
            from ptv3_shared_wam_v2.geometry import JointGeometry
            geometry = JointGeometry(config)
        self.geometry = geometry
        self.scene_project = nn.Linear(point_width, width)
        self.hand_project = nn.Linear(2 * point_width, width)
        self.position = mlp(3, width)
        self.goal = mlp(6, width)
        self.token_type = nn.Embedding(3, width)
        layer = nn.TransformerEncoderLayer(width, config["shared_heads"], 4 * width, dropout=0.,
            activation="gelu", batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, config["shared_layers"], enable_nested_tensor=False)
        self.hand_future = TemporalHead(width, head, 3)
        self.scene_query = mlp(3, head)
        self.scene_context = nn.Linear(width, head)
        self.scene_readout = nn.MultiheadAttention(head, config["head_heads"], dropout=0., batch_first=True)
        self.scene_norm = nn.LayerNorm(head)
        self.scene_future = TemporalHead(head, head, 3)
        # The new relational modules are separate from the v3 trunk. Common v3
        # parameters can therefore be warm-started while C is initialized fresh.
        edge_width = int(config.get("interaction_edge_width", 64))
        self.interaction_edge_width = edge_width
        self.interaction_hand_project = nn.Linear(width, edge_width)
        self.interaction_object_project = nn.Linear(head, edge_width)
        self.interaction_relation = mlp(3, edge_width)
        self.interaction_time = nn.Embedding(C_HORIZON, edge_width)
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(edge_width), nn.Linear(edge_width, edge_width), nn.GELU(),
            nn.Linear(edge_width, C_DIM)
        )
        nn.init.normal_(self.interaction_head[-1].weight, std=.01)
        nn.init.zeros_(self.interaction_head[-1].bias)
        self.register_buffer("interaction_scale", interaction_scales(config))
        self.normalizer = HorizonNormalizer(stats, config.get("horizon_normalization", True))

    def encode_hand(self, batch):
        valid = batch["start_hand_valid"].bool()
        x = torch.cat((clean(batch["start_hand"], valid), clean(batch["start_hand_history"], valid)), -1)
        identity = self.hand_id.weight[:, None] + self.keypoint_id.weight[None]
        hand = self.hand_encoder(x / self.hand_scale) + identity[None]
        # Geometry and the shared transformer both consume one token per
        # canonical point.  Keep the 2x11 token axis; flattening it here made
        # the PTv3 hand feature contract inconsistent with JointGeometry.
        return torch.where(valid[..., None], hand, 0.).reshape(len(hand), 22, -1), valid.flatten(1)

    def encode(self, batch):
        hand, hand_valid = self.encode_hand(batch)
        geometry_inputs = {key: batch[key] for key in
            ("xyz", "features", "point_valid", "start_hand", "start_hand_valid")}
        scene = self.geometry(geometry_inputs, hand, hand_valid)
        scene_tokens = self.scene_project(scene["tokens"]) + self.position(clean(scene["xyz"], scene["mask"]) / self.position_scale)
        explicit = hand if self.explicit_hand else torch.zeros_like(hand)
        hand_tokens = self.hand_project(torch.cat((scene["hand"], explicit), -1))
        goal_valid = batch["object_goal_valid"].bool()
        goal = torch.cat((clean(batch["start_object_points"], goal_valid),
                          clean(batch["goal_object_delta"], goal_valid)), -1) / self.position_scale
        tokens = torch.cat((scene_tokens + self.token_type.weight[0], hand_tokens + self.token_type.weight[1],
                            self.goal(goal) + self.token_type.weight[2]), 1)
        valid = torch.cat((scene["mask"], hand_valid, goal_valid), 1)
        tokens = torch.where(valid[..., None], tokens, 0.)
        if self.config.get("checkpoint_interaction", True) and self.training and torch.is_grad_enabled():
            for layer in self.interaction.layers:
                tokens = checkpoint(layer, tokens, src_key_padding_mask=~valid,
                    use_reentrant=False, preserve_rng_state=False)
        else:
            tokens = self.interaction(tokens, src_key_padding_mask=~valid)
        return torch.where(valid[..., None], tokens, 0.), valid, scene

    def forward(self, batch, return_shared=False):
        shared, shared_valid, geometry = self.encode(batch)
        offset = geometry["tokens"].shape[1]
        hand_z = self.hand_future(shared[:, offset:offset + 22]).reshape(len(shared), HORIZON, 2, 11, 3)
        points = batch["scene_points"]
        object_valid = batch["object_valid"].bool()
        point_valid = object_valid[..., None].expand(*points.shape[:-1])
        query = self.scene_query(clean(points, point_valid) / self.position_scale).flatten(1, 2)
        context = self.scene_context(shared)
        update, _ = self.scene_readout(query, context, context, key_padding_mask=~shared_valid, need_weights=False)
        object_latent = self.scene_norm(query + update).reshape(len(points), points.shape[1], points.shape[2], -1)
        scene_z = self.scene_future(object_latent.flatten(1, 2))
        scene_z = scene_z.reshape(len(points), HORIZON, *points.shape[1:3], 3).transpose(1, 2)
        point_mask = point_valid.bool()
        pooled = (object_latent * point_mask[..., None]).sum(2) / point_mask.sum(2).clamp_min(1)[..., None]
        # A factorized edge head keeps the graph structured: hand keypoints and
        # object patches remain separate until the relation MLP.  It avoids the
        # lossy global mean pooling used by the old v4 C head.
        hand_shared = shared[:, offset:offset + 22].reshape(len(shared), 2, 11, -1)
        hand_relation = self.interaction_hand_project(hand_shared)
        object_relation = self.interaction_object_project(object_latent)
        hand_future = batch["start_hand"][:, None] + self.normalizer.denormalize("hand", hand_z)
        object_future = points[:, :, None] + self.normalizer.denormalize("scene", scene_z)
        relative = object_future[:, :, :, None, None, :, :] - hand_future[:, None, :, :, :, None, :]
        relation = (object_relation[:, :, None, None, None, :, :]
                    + hand_relation[:, None, None, :, :, None, :]
                    + self.interaction_relation(relative / self.position_scale)
                    + self.interaction_time.weight[None, None, :, None, None, None, :])
        interaction_logits = self.interaction_head(relation)
        interaction_normalized = torch.cat((interaction_logits[..., :1].sigmoid(),
                                            interaction_logits[..., 1:]), dim=-1)
        interaction = interaction_normalized * self.interaction_scale.view(1, 1, 1, 1, 1, 1, -1)
        hand_mask = batch["start_hand_valid"][:, None, ..., None]
        scene_mask = point_valid[:, :, None, :, None]
        graph_valid = batch.get("scene_valid")
        if graph_valid is None:
            graph_valid = point_valid[:, :, None].expand(-1, -1, HORIZON, -1)
        graph_valid = (graph_valid.bool()[:, :, :, None, None, :] &
                       batch["trajectory_valid"].bool()[:, None, :, :, :, None])
        graph_valid = graph_valid & batch["start_hand_valid"].bool()[:, None, None, :, :, None]
        graph_valid = graph_valid & object_valid[:, :, None, None, None, None]
        c_mask = graph_valid
        prediction = dict(
            trajectory=torch.where(hand_mask, self.normalizer.denormalize("hand", hand_z), 0.),
            scene=torch.where(scene_mask, self.normalizer.denormalize("scene", scene_z), 0.),
            interaction=torch.where(c_mask[..., None], interaction, 0.),
            interaction_logits=torch.where(c_mask[..., None], interaction_logits, 0.),
            trajectory_normalized=torch.where(hand_mask, hand_z, 0.),
            scene_normalized=torch.where(scene_mask, scene_z, 0.),
            interaction_normalized=torch.where(c_mask[..., None], interaction_normalized, 0.),
            interaction_valid=c_mask,
        )
        if return_shared:
            prediction.update(shared_tokens=shared, shared_valid=shared_valid,
                              interaction_tokens=relation)
        return prediction

    def parameter_groups(self, geometry_lr, shared_lr):
        geometry = list(self.geometry.parameters())
        ids = {id(p) for p in geometry}
        return [dict(name="joint_geometry", params=geometry, lr=geometry_lr),
                dict(name="shared_and_heads", params=[p for p in self.parameters() if id(p) not in ids], lr=shared_lr)]

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    geometry=sum(p.numel() for p in self.geometry.parameters()),
                    shared=sum(p.numel() for p in self.interaction.parameters()),
                    interaction_head=sum(p.numel() for p in self.interaction_head.parameters()))
