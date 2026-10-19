"""PTv3 shared WAM-v6: dense observed surfaces, joint21 motion, surface contact."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from . import paths
from .contracts import SCHEMA, HORIZON, C_DIM
INTERACTION_ENCODING = "object-centric-surface-contact16-local-id.v5"
from ptv3_shared_wam_v5.normalization import HorizonNormalizer
C_HORIZON = HORIZON

def interaction_scales(config):
    return torch.tensor(config.get("interaction_scales", [1.] * C_DIM))
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


class SharedWAMV6(nn.Module):
    """Goal-conditioned WAM with ``tau`` forecasts and ``C[t, object_query]``."""

    def __init__(self, stats, config, geometry=None):
        super().__init__()
        if (config.get("schema") != SCHEMA or config.get("horizon") != HORIZON
                or config.get("goal_encoding") != GOAL_ENCODING
                or config.get("task_conditioning") != TASK_CONDITIONING):
            raise ValueError("independent v6 shared-goal future24 config required")
        if config.get("interaction_encoding", INTERACTION_ENCODING) != INTERACTION_ENCODING:
            raise ValueError("unsupported object-centric interaction field encoding")
        if int(config.get("interaction_dim", C_DIM)) != C_DIM:
            raise ValueError("interaction_dim must be seventeen for the v5 field contract")
        if int(config.get("interaction_horizon", C_HORIZON)) != C_HORIZON:
            raise ValueError("interaction field must cover the 24-step horizon")
        if int(config.get("surface_points", 2048)) != 2048 or int(config.get("interaction_points", 2048)) != 2048 or int(config.get("hand_points", 21)) != 21:
            raise ValueError("v6 requires 2048 object surface/contact queries and joint21")
        self.config = config
        point_width, width, head = config["point_width"], config["shared_width"], config["head_width"]
        self.explicit_hand = config.get("explicit_hand", True)
        self.position_scale = config["position_scale_m"]
        scales = [self.position_scale] * 3 + [config["velocity_scale_mps"]] * 3
        scales += [config["acceleration_scale_mps2"]] * 3 + [self.position_scale] * 3
        if min(scales) <= 0:
            raise ValueError("fixed physical scales must be positive")
        self.register_buffer("hand_scale", torch.tensor(scales))
        self.hand_encoder = mlp(12, point_width)
        self.hand_id = nn.Embedding(2, point_width)
        self.keypoint_id = nn.Embedding(21, point_width)
        if geometry is None:
            from ptv3_shared_wam_v5.geometry import DenseJointGeometry
            geometry = DenseJointGeometry(config)
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

        edge_width = int(config.get("interaction_edge_width", 64))
        self.interaction_edge_width = edge_width
        self.interaction_query = mlp(6, edge_width)
        self.interaction_local_project = nn.Linear(point_width, edge_width)
        self.interaction_object_project = nn.Linear(head, edge_width)
        self.interaction_hand_project = nn.Linear(width, edge_width)
        self.interaction_context_project = nn.Linear(width, edge_width)
        self.interaction_time = nn.Embedding(C_HORIZON, edge_width)
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(edge_width), nn.Linear(edge_width, edge_width), nn.GELU(), nn.Linear(edge_width, C_DIM))
        nn.init.normal_(self.interaction_head[-1].weight, std=.01)
        nn.init.zeros_(self.interaction_head[-1].bias)
        scale = interaction_scales(config)
        # C is a probability head: multiplying sigmoid/softmax outputs by a
        # non-unit scale can produce values outside [0,1] while the loss still
        # treats the pre-scale logits as probabilities.  Reject that contract
        # instead of silently reinterpreting it.
        if not torch.allclose(scale, torch.ones_like(scale)):
            raise ValueError("v6 interaction probability head requires unit interaction_scales")
        self.register_buffer("interaction_scale", scale)
        self.normalizer = HorizonNormalizer(stats, config.get("horizon_normalization", True))
        self.event_conditioning = config.get('event_conditioning',False)
        if not self.event_conditioning:
            raise ValueError('v6 requires explicit event conditioning; use v5 for baseline')
        if self.event_conditioning:
            from .events import EVENT_ENCODING,EventGoalResidual
            if config.get('event_encoding')!=EVENT_ENCODING:
                raise ValueError('explicit planned contact/release event encoding required')
            self.event_goal=EventGoalResidual(width)

    def encode_hand(self, batch):
        valid = batch["start_hand_valid"].bool()
        x = torch.cat((clean(batch["start_hand"], valid), clean(batch["start_hand_history"], valid)), -1)
        identity = self.hand_id.weight[:, None] + self.keypoint_id.weight[None]
        hand = self.hand_encoder(x / self.hand_scale) + identity[None]
        return torch.where(valid[..., None], hand, 0.).reshape(len(hand), 42, -1), valid.flatten(1)

    def encode(self, batch):
        hand, hand_valid = self.encode_hand(batch)
        geometry_inputs = {key: batch[key] for key in
            ("xyz", "features", "point_valid", "start_hand", "start_hand_valid",
             "dense_hand", "dense_history", "dense_hand_valid", "dense_hand_part")}
        scene = self.geometry(geometry_inputs, hand, hand_valid)
        scene_tokens = self.scene_project(scene["tokens"]) + self.position(clean(scene["xyz"], scene["mask"]) / self.position_scale)
        explicit = hand if self.explicit_hand else torch.zeros_like(hand)
        hand_tokens = self.hand_project(torch.cat((scene["hand"], explicit), -1))
        goal_valid = batch["object_goal_valid"].bool()
        goal = torch.cat((clean(batch["start_object_points"], goal_valid),
                          clean(batch["goal_object_delta"], goal_valid)), -1) / self.position_scale
        goal_tokens=self.goal(goal)
        if self.event_conditioning:
            goal_tokens=goal_tokens+self.event_goal(batch)[:,None]
        tokens = torch.cat((scene_tokens + self.token_type.weight[0], hand_tokens + self.token_type.weight[1],
                            goal_tokens + self.token_type.weight[2]), 1)
        valid = torch.cat((scene["mask"], hand_valid, goal_valid), 1)
        tokens = torch.where(valid[..., None], tokens, 0.)
        if self.config.get("checkpoint_interaction", True) and self.training and torch.is_grad_enabled():
            for layer in self.interaction.layers:
                tokens = checkpoint(layer, tokens, src_key_padding_mask=~valid,
                    use_reentrant=False, preserve_rng_state=False)
        else:
            tokens = self.interaction(tokens, src_key_padding_mask=~valid)
        return torch.where(valid[..., None], tokens, 0.), valid, scene

    @staticmethod
    def _masked_mean(values, mask, dim=1):
        weights = mask.float()[..., None]
        return (values * weights).sum(dim) / weights.sum(dim).clamp_min(1.)

    def forward(self, batch, return_shared=False):
        shared, shared_valid, geometry = self.encode(batch)
        offset = geometry["tokens"].shape[1]
        hand_z = self.hand_future(shared[:, offset:offset + 42]).reshape(len(shared), HORIZON, 2, 21, 3)

        points = batch["scene_points"]
        object_valid = batch["object_valid"].bool()
        point_valid = object_valid[..., None].expand(*points.shape[:-1])
        query = self.scene_query(clean(points, point_valid) / self.position_scale).flatten(1, 2)
        context = self.scene_context(shared)
        update, _ = self.scene_readout(query, context, context, key_padding_mask=~shared_valid, need_weights=False)
        object_latent = self.scene_norm(query + update).reshape(len(points), points.shape[1], points.shape[2], -1)
        scene_z = self.scene_future(object_latent.flatten(1, 2))
        scene_z = scene_z.reshape(len(points), HORIZON, *points.shape[1:3], 3).transpose(1, 2)

        interaction_points = batch["interaction_points"]
        interaction_normals = batch["interaction_normals"]
        # Prediction validity is a current-observation mask.  Clean invalid
        # padding before it reaches Linear/LayerNorm; a NaN in a valid query
        # is a hard error, not something the trailing output mask can repair.
        query_valid = batch.get("interaction_query_valid",
                                torch.ones_like(interaction_points[..., 0], dtype=torch.bool))
        if query_valid.shape != interaction_points.shape[:-1]:
            raise ValueError("current interaction query validity must have shape B,O,Q")
        interaction_points = clean(interaction_points, query_valid)
        interaction_normals = clean(interaction_normals, query_valid)
        q = self.interaction_query(torch.cat((interaction_points, interaction_normals), -1))
        # Single target object; reader binds C query IDs to the same CAD points
        # as geometry.xyz. No search/interpolation or future labels enter here.
        if interaction_points.shape[1]!=1 or interaction_points.shape[2]!=batch['xyz'].shape[1]:
            raise ValueError('C local readout requires one target and identical surface query IDs')
        if not torch.equal(interaction_points[:,0],clean(batch['xyz'],query_valid[:,0])):
            raise ValueError('C surface point order differs from geometry point IDs')
        local_valid=geometry['point_valid'].bool()&query_valid[:,0]
        local=self.interaction_local_project(clean(geometry['point_features'],local_valid))
        q=q+torch.where(local_valid[...,None],local,0.)[:,None]
        object_context = self._masked_mean(object_latent, point_valid, dim=2).unsqueeze(2).unsqueeze(3)
        hand_tokens = shared[:, offset:offset + 42]
        hand_context = self._masked_mean(hand_tokens, batch["start_hand_valid"].flatten(1)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        global_context = self._masked_mean(shared, shared_valid).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        def contact_chunk(query_features, oc, hc, gc):
            relation = (query_features.unsqueeze(2) + self.interaction_object_project(oc)
                        + self.interaction_hand_project(hc) + self.interaction_context_project(gc)
                        + self.interaction_time.weight[None, None, :, None])
            return self.interaction_head(relation)
        chunks = []
        chunk_size = int(self.config.get("contact_chunk_size", 256))
        for piece in q.split(chunk_size, dim=2):
            if self.training and torch.is_grad_enabled() and self.config.get("checkpoint_contact", True):
                logits = checkpoint(contact_chunk, piece, object_context, hand_context, global_context,
                                    use_reentrant=False, preserve_rng_state=False)
            else:
                logits = contact_chunk(piece, object_context, hand_context, global_context)
            chunks.append(logits)
        interaction_logits = torch.cat(chunks, dim=3)
        interaction_normalized = torch.cat((interaction_logits[..., :1].sigmoid(),
                                            interaction_logits[..., 1:].softmax(-1)), -1)
        interaction = interaction_normalized * self.interaction_scale.view(1, 1, 1, 1, -1)
        # Prediction validity comes only from current observations. Future
        # label availability belongs exclusively to loss/metric computation.
        c_mask = (query_valid.bool()[:, :, None, :] & object_valid[:, :, None, None]).expand(
            -1, -1, C_HORIZON, -1)
        hand_mask = batch["start_hand_valid"][:, None, ..., None]
        scene_mask = point_valid[:, :, None, :, None]
        prediction = dict(
            trajectory=torch.where(hand_mask, self.normalizer.denormalize("hand", hand_z), 0.),
            scene=torch.where(scene_mask, self.normalizer.denormalize("scene", scene_z), 0.),
            interaction=torch.where(c_mask[..., None], interaction, 0.),
            interaction_logits=torch.where(c_mask[..., None], interaction_logits, 0.),
            trajectory_normalized=torch.where(hand_mask, hand_z, 0.),
            scene_normalized=torch.where(scene_mask, scene_z, 0.),
            interaction_normalized=torch.where(c_mask[..., None], interaction_normalized, 0.),
            interaction_prediction_valid=c_mask)
        if return_shared:
            prediction.update(shared_tokens=shared, shared_valid=shared_valid,
                              
                              scene_token_count=geometry["tokens"].shape[1])
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
