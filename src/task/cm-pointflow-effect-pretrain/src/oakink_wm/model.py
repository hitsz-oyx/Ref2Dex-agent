"""Sparse scene encoder, semantic action tokens and multi-object rigid dynamics."""
import torch
from torch import nn
import torch.nn.functional as F
import spconv.pytorch as spconv


def rotation6d(x):
    a = F.normalize(x[..., :3], dim=-1, eps=1e-6)
    b = F.normalize(x[..., 3:] - (a * x[..., 3:]).sum(-1, keepdim=True) * a, dim=-1, eps=1e-6)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), dim=-1)


def geodesic(R, truth):
    relative = R.transpose(-1, -2) @ truth
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), -1) * .5
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5).clamp(-1, 1)
    return torch.atan2(sine, cosine)


def mean_groups(values, inverse, count):
    out = values.new_zeros((count, values.shape[-1]))
    out.index_add_(0, inverse, values)
    denom = torch.bincount(inverse, minlength=count).to(values.dtype).clamp_min(1)
    return out / denom[:, None]


class SparseScene(nn.Module):
    def __init__(self, d=384, layers=8):
        super().__init__()
        modules = []
        inputs = 21
        for i, width in enumerate((64, 128, 256)):
            modules += [spconv.SubMConv3d(inputs, width, 3, padding=1, bias=False, indice_key='stem'),
                        nn.LayerNorm(width), nn.GELU()]
            inputs = width
        self.stem = spconv.SparseSequential(*modules)
        self.projection = nn.Linear(256, d)
        self.position = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 6, 1536, dropout=0,
                                                activation='gelu', batch_first=True, norm_first=True), layers,
                                                enable_nested_tensor=False)
        self.world_type = nn.Parameter(torch.zeros(d))

    def forward(self, batch):
        xyz, features, valid = batch['xyz'], batch['features'], batch['point_valid']
        B = len(xyz)
        # Sparse CUDA stem remains float32; dense transformers can use bf16.
        with torch.autocast(device_type='cuda', enabled=False):
            xyz, features = xyz.float(), features.float()
            vox = torch.floor(xyz / .01).long()
            residual = xyz - (vox.float() + .5) * .01
            bids = torch.arange(B, device=xyz.device)[:, None].expand_as(valid)
            coords = torch.cat((bids[valid][:, None], vox[valid]), dim=-1)
            unique, inverse = torch.unique(coords, dim=0, return_inverse=True)
            feat = mean_groups(torch.cat((residual, features), -1)[valid], inverse, len(unique))
            origin = unique[:, 1:].amin(0)
            sparse_coords = unique.clone()
            sparse_coords[:, 1:] -= origin
            spatial_shape = (sparse_coords[:, 1:].amax(0) + 3).tolist()
            tensor = spconv.SparseConvTensor(feat, sparse_coords.int(), spatial_shape, B)
            stem = self.stem(tensor).features
            patch = unique.clone()
            patch[:, 1:] = torch.div(patch[:, 1:], 5, rounding_mode='floor')
            pcoords, pinverse = torch.unique(patch, dim=0, return_inverse=True)
            pooled = mean_groups(stem, pinverse, len(pcoords))
            positions = mean_groups((unique[:, 1:].float() + .5) * .01, pinverse, len(pcoords))
        tokens = self.projection(pooled) + self.position(positions) + self.world_type
        counts = torch.bincount(pcoords[:, 0], minlength=B)
        maximum = int(counts.max())
        packed = tokens.new_zeros((B, maximum, tokens.shape[-1]))
        mask = torch.ones((B, maximum), device=xyz.device, dtype=torch.bool)
        # Per-example patch counts vary; packing preserves all local patches.
        for b in range(B):
            selected = pcoords[:, 0] == b
            packed[b, :int(counts[b])] = tokens[selected]
            mask[b, :int(counts[b])] = False
        return self.encoder(packed, src_key_padding_mask=mask), mask


class WorldModel(nn.Module):
    def __init__(self, d=384, scene_layers=8, action_layers=4, dynamics_layers=6):
        super().__init__()
        self.scene = SparseScene(d, scene_layers)
        self.action_mlp = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.time = nn.Embedding(24, d)
        self.hand = nn.Embedding(2, d)
        self.keypoint = nn.Embedding(11, d)
        self.action_type = nn.Parameter(torch.zeros(d))
        self.action_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 6, 1536, dropout=0,
                                         activation='gelu', batch_first=True, norm_first=True), action_layers,
                                         enable_nested_tensor=False)
        self.object_mlp = nn.Sequential(nn.Linear(15, d), nn.GELU(), nn.Linear(d, d))
        self.effect_type = nn.Parameter(torch.zeros(d))
        self.dynamics = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 6, 1536, dropout=0,
                                           activation='gelu', batch_first=True, norm_first=True), dynamics_layers)
        self.head = nn.Linear(d, 9)
        nn.init.normal_(self.head.weight, std=1e-4)
        with torch.no_grad(): self.head.bias.copy_(torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0.]))

    def forward(self, batch, arm='action'):
        memory, mask = self.scene(batch)
        B, M = batch['object_valid'].shape
        if arm != 'history':
            a = batch['action'].reshape(B, 24, 2, 11, 9)
            tokens = (self.action_mlp(a) + self.time.weight[None, :, None, None] +
                      self.hand.weight[None, None, :, None] + self.keypoint.weight[None, None, None] + self.action_type)
            tokens = tokens.reshape(B, 528, -1)
            amask = ~batch['action_valid'].reshape(B, 528)
            encoded = self.action_encoder(tokens, src_key_padding_mask=amask)
            memory = torch.cat((memory, encoded), 1)
            mask = torch.cat((mask, amask), 1)
        query = self.object_mlp(batch['object_features'])[:, :, None] + self.time.weight[None, None] + self.effect_type
        qmask = (~batch['object_valid'])[:, :, None].expand(B, M, 24).reshape(B, M*24)
        result = self.dynamics(query.reshape(B, M*24, -1), memory,
                               tgt_key_padding_mask=qmask, memory_key_padding_mask=mask)
        raw = self.head(result).reshape(B, M, 24, 9).float()
        return dict(translation=raw[..., :3], rotation=rotation6d(raw[..., 3:]))


def rigid_points(rotation, translation, points):
    return torch.einsum('bmtij,bmnj->bmtni', rotation, points) + translation[..., None, :]


def losses(pred, batch):
    gt = batch['effect'].float()
    truthR, truthT = gt[..., :3, :3], gt[..., :3, 3]
    mask = batch['object_valid'][:, :, None].expand_as(truthT[..., 0]).float()
    denom = mask.sum().clamp_min(1)
    trans = (pred['translation'] - truthT).abs().mean(-1)
    angle = geodesic(pred['rotation'], truthR)
    rotation = angle * batch['radius'][:, :, None]
    predicted = rigid_points(pred['rotation'], pred['translation'], batch['points'])
    actual = rigid_points(truthR, truthT, batch['points'])
    point = (predicted - actual).abs().mean((-1, -2))
    components = {k: (v * mask).sum() / denom for k, v in
                  [('translation', trans), ('rotation_metric', rotation), ('point', point)]}
    return sum(components.values()), components


def metrics(pred, batch):
    gt = batch['effect'].float()
    p = batch['points'].float()
    predicted = rigid_points(pred['rotation'], pred['translation'], p)
    actual = rigid_points(gt[..., :3, :3], gt[..., :3, 3], p)
    point = torch.linalg.vector_norm(predicted - actual, dim=-1).mean(-1)
    center = p.mean(-2)
    cp = torch.einsum('bmtij,bmj->bmti', pred['rotation'], center) + pred['translation']
    cg = torch.einsum('bmtij,bmj->bmti', gt[..., :3, :3], center) + gt[..., :3, 3]
    return dict(point_epe=point, translation=torch.linalg.vector_norm(pred['translation']-gt[..., :3, 3], dim=-1),
                center_error=torch.linalg.vector_norm(cp-cg, dim=-1), rotation=geodesic(pred['rotation'], gt[..., :3, :3]))
