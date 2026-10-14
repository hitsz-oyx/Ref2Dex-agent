"""Joint PTv3 with separate scene/hand voxel means and semantic hand readback."""
from types import MethodType
import torch
from torch import nn
from . import paths
from ptv3_shared_wam.model import clean
from ptv3_shared_wam.scene import SceneEncoder
from ptv3_shared_wam.padding import vectorized_padding
from oakink_wm.model import mean_groups
from oakink_wm.pointworld import fixed_workspace_grid
from ptv3.ptv3 import PointTransformerV3, SerializedAttention


class JointGeometry(nn.Module):
    def __init__(self, config, backbone=None):
        super().__init__()
        width = config["point_width"]
        self.width, self.token_count = width, config["scene_tokens"]
        self.joint = config.get("hand_in_ptv3", True)
        self.voxel_size_m = config["voxel_m"]
        self.register_buffer("voxel_origin_m", torch.tensor(config["voxel_origin_m"]))
        # Normal, velocity(m/s), acceleration(m/s²), H4 displacement(m), metadata.
        self.register_buffer("feature_scale", torch.tensor([1.]*3 + [config["velocity_scale_mps"]]*3
            + [config["acceleration_scale_mps2"]]*3 + [config["position_scale_m"]]*3 + [1.]*6))
        self.scene_proj = nn.Sequential(nn.Linear(18, width), nn.GELU(), nn.Linear(width, width))
        self.voxel_fusion = nn.Linear(2 * width, width)
        if backbone is None:
            pc = config["ptv3"]
            backbone = PointTransformerV3(in_channels=width,
                enc_channels=tuple(pc["enc_channels"]), enc_depths=tuple(pc["enc_depths"]),
                enc_num_head=tuple(pc["enc_heads"]), dec_channels=tuple(pc["dec_channels"]),
                dec_depths=tuple(pc["dec_depths"]), dec_num_head=tuple(pc["dec_heads"]),
                enc_patch_size=(config["patch_size"],)*5, dec_patch_size=(config["patch_size"],)*4,
                stride=(2,)*4, drop_path=pc["drop_path"], shuffle_orders=True,
                enable_rpe=False, traceable=True, mask_token=False, enc_mode=False)
            for module in backbone.modules():
                if isinstance(module, SerializedAttention):
                    module.get_padding_and_inverse = MethodType(vectorized_padding, module)
        self.backbone = backbone

    def train(self, mode=True):
        super().train(mode)
        for module in self.backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = mode
        return self

    _sample_tokens = SceneEncoder._sample_tokens

    def forward(self, batch, hand_encoding, hand_valid):
        scene_valid = batch["point_valid"].bool() & (batch["features"][..., 14] < .5)
        if not scene_valid.any(1).all():
            raise ValueError("each scene requires observed non-hand geometry")
        scene_xyz = clean(batch["xyz"], scene_valid)
        scene_features = self.scene_proj(clean(batch["features"], scene_valid) / self.feature_scale)
        hand_xyz = clean(batch["start_hand"], batch["start_hand_valid"].bool()).flatten(1, 2)
        xyz = torch.cat((scene_xyz, hand_xyz), 1)
        valid = torch.cat((scene_valid, hand_valid & self.joint), 1)
        is_hand = torch.zeros_like(valid)
        is_hand[:, scene_valid.shape[1]:] = True
        features = torch.cat((scene_features, hand_encoding), 1)
        ids = torch.arange(len(xyz), device=xyz.device)[:, None].expand_as(valid)
        with torch.autocast("cuda", enabled=False) if xyz.is_cuda else torch.autocast("cpu", enabled=False):
            grid = fixed_workspace_grid(xyz, valid, self.voxel_origin_m, self.voxel_size_m)
            unique, inverse = torch.unique(torch.cat((ids[valid][:, None], grid[valid]), -1),
                                           dim=0, return_inverse=True)
            hand_rows = is_hand[valid]
            # Separate means prevent dense surfaces from washing out hand features.
            scene_mean = mean_groups(features.float()[valid][~hand_rows], inverse[~hand_rows], len(unique))
            hand_mean = mean_groups(features.float()[valid][hand_rows], inverse[hand_rows], len(unique))
            fused = self.voxel_fusion(torch.cat((scene_mean, hand_mean), -1))
            encoded = self.backbone(dict(coord=mean_groups(xyz[valid], inverse, len(unique)),
                feat=fused, grid_coord=unique[:, 1:].int(), batch=unique[:, 0], grid_size=self.voxel_size_m))
            packed = encoded.feat.new_zeros((*valid.shape, self.width))
            packed[valid] = encoded.feat[inverse]
        count = scene_valid.shape[1]
        scene, mask = self._sample_tokens(torch.cat((packed[:, :count], scene_xyz), -1), scene_valid)
        return dict(tokens=scene[..., :self.width], xyz=scene[..., self.width:], mask=mask,
                    hand=packed[:, count:], voxel_count=len(unique))
