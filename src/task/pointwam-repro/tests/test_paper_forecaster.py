import torch

from pointwam_repro.encoders import (
    MOSAIC_CONDITION,
    SCENE_FEATURE_DIM,
    TEXT_TOKEN_DIM,
    normalize_color,
)
from pointwam_repro.loss import standardized_joint_loss
from pointwam_repro.pointwam import PointWAMForecaster, _pack_stem
from pointwam_repro.rope import apply_axial_rope, rope_channel_splits
from pointwam_repro.sparse import SparseConv3d
from pointwam_repro.stats import fit_horizon_stats
from pointwam_repro.volt import VoltEncoder


def _tiny() -> PointWAMForecaster:
    return PointWAMForecaster(
        scene_dim=8, text_dim=6, width=32, layers=2, heads=4,
        stem_dim=8, horizon=4, drop_path_max=0.0,
    )


def _observation(points: int = 3):
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.08, 0.0, 0.0]])[:points]
    features = torch.randn(points, 8)
    valid = torch.ones(points, dtype=torch.bool)
    hand = torch.zeros(2, 10, 3)
    hand_valid = torch.zeros(2, 10, dtype=torch.bool)
    hand_valid[0, 0] = True
    text = torch.randn(2, 6)
    return xyz, features, valid, hand, hand_valid, text


def test_rope_split_and_zero_position_is_identity() -> None:
    assert rope_channel_splits(64) == (20, 22, 22)
    query = torch.randn(1, 2, 1, 8)
    key = torch.randn(1, 2, 1, 8)
    rotated_query, _ = apply_axial_rope(query, key, torch.zeros(1, 3, dtype=torch.long))
    assert torch.allclose(rotated_query, query, atol=1e-6)
    moved, _ = apply_axial_rope(query, key, torch.tensor([[3, -2, 1]]))
    assert not torch.allclose(moved, query)


def test_instruction_prefix_is_not_rotated() -> None:
    encoder = VoltEncoder(width=32, layers=1, heads=4, drop_path_max=0.0)
    tokens = torch.randn(1, 3, 32)
    origin = encoder(tokens, torch.zeros(1, 3, dtype=torch.long), n_text=2)
    shifted = encoder(tokens, torch.tensor([[5, 1, -3]]), n_text=2)
    assert not torch.allclose(origin, shifted)
    assert encoder.drop_rates[0] == 0.0
    deep = VoltEncoder(width=32, layers=4, heads=4, drop_path_max=0.3)
    assert deep.drop_rates[0] == 0.0
    assert abs(deep.drop_rates[-1] - 0.3) < 1e-8


def test_sparse_conv_reads_an_occupied_neighbor_and_skips_empty() -> None:
    conv = SparseConv3d(1, 1)
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.zero_()
        conv.weight[4, 0, 0] = 1
    coords = torch.tensor([[0, 0, 0], [1, 0, 0]])
    output = conv(torch.tensor([[3.0], [4.0]]), coords)
    assert torch.allclose(output, torch.tensor([[0.0], [3.0]]))


def test_empty_patch_slot_stays_zero() -> None:
    features = torch.tensor([[1.0], [2.0]])
    coords = torch.tensor([[0, 0, 0], [1, 0, 0]])
    packed = _pack_stem(features, coords, 5)
    assert packed["tokens"].shape[1] == 125
    filled = packed["tokens"].abs().sum(dim=-1) > 0
    assert int(filled.sum()) == 2
    assert torch.count_nonzero(packed["tokens"][~filled]) == 0


def test_points_in_one_voxel_share_a_scene_forecast() -> None:
    torch.manual_seed(2)
    model = _tiny().eval()
    xyz, features, valid, hand, hand_valid, text = _observation()
    features[0] = 0
    features[1] = 4
    output = model.forward_observation(xyz, features, valid, hand, hand_valid, text)
    assert torch.allclose(output["scene_displacement"][:, 0], output["scene_displacement"][:, 1])
    assert output["scene_displacement"].shape == (4, 3, 3)
    assert output["hand_displacement"].shape == (4, 2, 10, 3)
    assert torch.count_nonzero(output["hand_displacement"][:, 1]) == 0


def test_zero_fusion_starts_from_detokenized_features() -> None:
    model = _tiny()
    decoded = torch.randn(5, model.stem_dim)
    stem = torch.randn(5, model.stem_dim)
    fused = model.fuse(decoded, stem)
    expected = torch.nn.functional.gelu(model.out_norm(decoded))
    assert torch.allclose(fused, expected, atol=1e-6)
    assert torch.count_nonzero(model.fuse_out.weight) == 0


def test_scene_clip_and_standardized_loss() -> None:
    example = {
        "scene_displacement": torch.tensor([[[0.8, 0.0, 0.0]]]),
        "scene_target_valid": torch.ones(1, 1, dtype=torch.bool),
        "hand_displacement": torch.zeros(1, 2, 10, 3),
        "hand_target_valid": torch.zeros(1, 2, 10, dtype=torch.bool),
    }
    example["hand_displacement"][0, 0, 0, 0] = 0.2
    example["hand_target_valid"][0, 0, 0] = True
    stats = fit_horizon_stats([example])
    assert torch.allclose(stats.scene_mean[0, 0], torch.tensor(0.5))
    assert float(stats.scene_std[0, 0]) == 1.0
    assert torch.allclose(stats.standardize_hand(example["hand_displacement"])[0, 0, 0, 0], torch.tensor(0.0))
    prediction_scene = torch.zeros(1, 1, 3)
    prediction_hand = torch.zeros(1, 2, 10, 3)
    positions_scene = torch.zeros(2, 1, 3)
    positions_hand = torch.zeros(2, 2, 10, 3)
    loss = standardized_joint_loss(
        prediction_scene, prediction_hand,
        example["scene_displacement"], example["hand_displacement"],
        example["scene_target_valid"], example["hand_target_valid"],
        positions_scene, positions_hand, stats)
    assert torch.isfinite(loss)


def test_tiny_forecaster_backpropagates_into_the_stem() -> None:
    torch.manual_seed(0)
    model = _tiny()
    xyz, features, valid, hand, hand_valid, text = _observation()
    output = model.forward_observation(xyz, features, valid, hand, hand_valid, text)
    output["scene_displacement"].sum().backward()
    assert model.stem.conv.weight.grad is not None
    assert torch.isfinite(model.stem.conv.weight.grad).all()
    assert model.text_proj.weight.grad is not None


def test_mosaic_color_matches_the_published_normalization() -> None:
    rgb = torch.tensor([[0.0, 127.5, 255.0]])
    color = normalize_color(rgb)
    assert torch.allclose(color, torch.tensor([[-1.0, 0.0, 1.0]]))
    assert MOSAIC_CONDITION == "ScanNet"


def test_volt_h_budget_stays_near_the_published_count() -> None:
    with torch.device("meta"):
        model = PointWAMForecaster()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert model.backbone.drop_rates[0] == 0.0
    assert abs(model.backbone.drop_rates[-1] - 0.3) < 1e-8
    assert model.scene_dim == SCENE_FEATURE_DIM
    assert model.text_dim == TEXT_TOKEN_DIM
    assert 600_000_000 < count < 720_000_000, count
