"""Minimal samples reach both heads; padding/context have no invented labels."""
import numpy as np
import pytest
import torch
from torch import nn

from ptv3_shared_wam.egodex_data import forecasting_collate
from ptv3_shared_wam.model import SharedWAM
from ptv3_shared_wam.objectives import forecast_loss, forecast_metrics
from test_pipeline import built


class SceneFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 128)

    def forward(self, batch):
        xyz = batch["xyz"]
        return dict(tokens=self.backbone(xyz), xyz=xyz, mask=batch["point_valid"])


def test_minimal_dataset_batch_reaches_shared_dual_heads():
    torch.set_num_threads(1)
    sample, _, _ = built()
    batch = forecasting_collate([sample])
    assert not any(key in batch for key in ("action", "effect", "native_state", "points"))
    assert (batch["scene_object"] == -2).any()
    assert batch["scene_points"].shape == (1, 1, 32, 3)
    assert batch["scene_target"].shape == (1, 1, 24, 32, 3)
    net = SharedWAM(dict(scene_mean=[0.]*18, scene_std=[1.]*18), dict(interaction_layers=1), SceneFixture())
    prediction = net(batch)
    loss, terms = forecast_loss(prediction, batch, {})
    assert torch.isfinite(loss)
    for term in ("hand", "scene"):
        grads = torch.autograd.grad(terms[term], list(net.interaction.parameters()), retain_graph=True)
        assert sum(g.abs().sum() for g in grads) > 0
    loss.backward()
    metrics = forecast_metrics(prediction, batch, {})
    assert metrics["scene_non_target_all24"]["count"] == 0
    assert metrics["scene_target_all24"]["count"] == 24*32


def test_padding_and_variable_lengths_keep_current_points():
    a, _, _ = built()
    b = dict(a)
    for key in ("xyz", "features", "scene_object"):
        b[key] = b[key][:80].copy()
    batch = forecasting_collate([a, b])
    assert batch["point_valid"][1].sum() == 80
    assert not batch["point_valid"][1, 80:].any()
    np.testing.assert_array_equal(batch["scene_points"][1].numpy(), a["scene_points"])


def test_robot_action_batches_cannot_be_silently_mixed():
    a, _, _ = built()
    a["native_state"] = np.zeros(87, np.float32)
    with pytest.raises(ValueError, match="cannot fabricate"):
        forecasting_collate([a])


def test_existing_three_domain_collate_semantics_are_preserved():
    from ptv3_shared_wam.paths import ROOT
    from ptv3_shared_wam.data import SharedWindows, shared_collate
    path = ROOT/"outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree"
    if not path.exists():
        pytest.skip("three-domain read-only corpus unavailable")
    sample = SharedWindows(path, "train")[0]
    expected = shared_collate([sample])
    actual = forecasting_collate([sample])
    for key, value in actual.items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_engineering_snapshot_requires_explicit_reader(tmp_path):
    from video_data_preparation.shards import ShardDataset, freeze_snapshot
    from test_storage import write_shard
    write_shard(tmp_path, "train", "train", "train.h5")
    freeze_snapshot(tmp_path, tmp_path/"snapshot.json", require_both=False, engineering=True)
    with pytest.raises(ValueError, match="engineering corpus"):
        ShardDataset(tmp_path/"snapshot.json", "train")
    dataset = ShardDataset(tmp_path/"snapshot.json", "train", engineering=True)
    assert len(dataset) == 1
    assert (dataset.sample_indices(10, np.random.default_rng(9)) == 0).all()
    dataset.close()
