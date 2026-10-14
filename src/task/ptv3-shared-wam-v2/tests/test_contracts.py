import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from ptv3_shared_wam_v2.paths import STATS_SCHEMA
from ptv3_shared_wam_v2.normalization import Moments, HorizonNormalizer
from ptv3_shared_wam_v2.model import SharedWAMV2
from ptv3_shared_wam_v2.objectives import forecast_loss
from ptv3_shared_wam_v2.checkpoint import load_checkpoint


def stats():
    means = np.arange(24)[:, None] * np.ones((1, 3)) * .001
    return dict(schema=STATS_SCHEMA, split="train", sources=["oakink2", "grab", "arctic"],
        hand=dict(mean=means.tolist(), std=np.full((24,3), .02).tolist()),
        scene=dict(mean=(-means).tolist(), std=np.full((24,3), .03).tolist()))


def config():
    c = json.loads((Path(__file__).resolve().parents[1]/"configs/pretrain_probe.json").read_text())
    c.update(point_width=8, shared_width=16, shared_heads=4, shared_layers=2, head_width=8,
             head_heads=2, scene_tokens=4, checkpoint_interaction=False)
    return c


class Geometry(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, width)
        self.seen = None

    def forward(self, batch, hand, valid):
        self.seen = set(batch)
        tokens = self.project(hand.mean(1, keepdim=True).expand(-1, 4, -1))
        return dict(tokens=tokens, xyz=torch.zeros(len(hand),4,3),
            mask=torch.ones(len(hand),4,dtype=torch.bool), hand=self.project(hand))


def batch():
    torch.manual_seed(3)
    return dict(start_hand=torch.randn(2,2,11,3)*.05,
        start_hand_history=torch.randn(2,2,11,9)*.01,
        start_hand_valid=torch.ones(2,2,11,dtype=torch.bool), xyz=torch.rand(2,8,3),
        features=torch.zeros(2,8,18), point_valid=torch.ones(2,8,dtype=torch.bool),
        scene_points=torch.randn(2,1,3,3)*.05, object_valid=torch.ones(2,1,dtype=torch.bool),
        start_object_points=torch.rand(2,3,3)*.05, goal_object_delta=torch.rand(2,3,3)*.1,
        object_goal_valid=torch.ones(2,3,dtype=torch.bool),
        trajectory_target=torch.rand(2,24,2,11,3)*.1, trajectory_valid=torch.ones(2,24,2,11,dtype=torch.bool),
        scene_target=torch.rand(2,1,24,3,3)*.1, scene_valid=torch.ones(2,1,24,3,dtype=torch.bool))


def model():
    c = config()
    return SharedWAMV2(stats(), c, Geometry(c["point_width"])).eval()


def test_moments_mask_nan_and_std_floor():
    m = Moments()
    x = np.broadcast_to(np.arange(24)[:,None,None], (24,3,3)).copy().astype(float)
    x[:,2] = np.nan
    valid = np.ones((24,3),dtype=bool); valid[:,2]=False
    m.update(x, valid)
    r=m.result(.001)
    np.testing.assert_allclose(r["mean"], np.repeat(np.arange(24)[:,None],3,1))
    np.testing.assert_allclose(r["std"], .001)
    assert r["count"] == [2]*24


@pytest.mark.parametrize("name,shape", [("hand",(2,24,2,11,3)),("scene",(2,3,24,4,3))])
def test_horizon_roundtrip(name, shape):
    n=HorizonNormalizer(stats()); x=torch.randn(*shape)
    torch.testing.assert_close(n.denormalize(name,n.normalize(name,x)), x)


@pytest.mark.parametrize("mutation", ["val", "v1", "zero_std"])
def test_stats_fail_closed(mutation):
    s=stats()
    if mutation=="val":s["split"]="val"
    elif mutation=="v1":s["schema"]="legacy"
    else:s["scene"]["std"][0][0]=0
    with pytest.raises(ValueError):HorizonNormalizer(s)


def test_masks_and_metric_outputs():
    m=model(); b=batch(); b["start_hand_valid"][0,1]=False
    b["start_hand"][0,1]=float("nan"); b["start_hand_history"][0,1]=float("nan")
    p=m(b)
    assert p["trajectory"].shape==(2,24,2,11,3)
    assert not p["trajectory"][0,:,1].any()
    assert torch.isfinite(p["trajectory"]).all()
    torch.testing.assert_close(p["scene"],m.normalizer.denormalize("scene",p["scene_normalized"]))
    assert m.geometry.seen == {"xyz","features","point_valid","start_hand","start_hand_valid"}


def test_goal_reaches_both_heads():
    m=model(); b=batch(); first=m(b)
    b["goal_object_delta"]+=.2
    second=m(b)
    assert not torch.allclose(first["scene"],second["scene"])
    assert not torch.allclose(first["trajectory"],second["trajectory"])


def test_losses_reach_shared_hand_and_goal_and_isolate_heads():
    m=model(); b=batch(); p=m(b)
    _,terms=forecast_loss(p,b,config(),m.normalizer)
    for term in terms.values():
        for group in (m.hand_encoder,m.interaction,m.goal):
            gs=torch.autograd.grad(term,list(group.parameters()),retain_graph=True)
            assert sum(float(g.abs().sum()) for g in gs)>0
    gs=torch.autograd.grad(terms["scene"],list(m.hand_future.parameters()),allow_unused=True)
    assert all(g is None for g in gs)


def test_normalized_objective_matches_known_error():
    m=model(); b=batch()
    p={"trajectory_normalized":m.normalizer.normalize("hand",b["trajectory_target"])+2,
       "scene_normalized":m.normalizer.normalize("scene",b["scene_target"])+3}
    total,terms=forecast_loss(p,b,config(),m.normalizer)
    torch.testing.assert_close(terms["hand"],torch.tensor(4.))
    torch.testing.assert_close(terms["scene"],torch.tensor(9.))
    torch.testing.assert_close(total,torch.tensor(13.))


def test_v1_checkpoint_rejected(tmp_path):
    path=tmp_path/"old.pt";torch.save(dict(schema="ref2dex.ptv3-shared-wam.v1",stage="pretrain"),path)
    with pytest.raises(ValueError):load_checkpoint(path)


class VoxelBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.last=None
    def forward(self,b):
        self.last=b
        return SimpleNamespace(feat=b["feat"])


def test_joint_voxel_keeps_semantic_readback_and_balances_density():
    from ptv3_shared_wam_v2.geometry import JointGeometry
    c=config(); backbone=VoxelBackbone(); g=JointGeometry(c,backbone)
    b=batch(); b={k:v[:1] for k,v in b.items()}
    # Every valid point collides in one voxel; hand semantics still have22 slots.
    b["xyz"].zero_();b["start_hand"].zero_()
    hand=torch.arange(22*8).reshape(1,22,8).float().requires_grad_()
    valid=torch.ones(1,22,dtype=torch.bool)
    first=g(b,hand,valid)
    assert first["hand"].shape==(1,22,8)
    assert first["voxel_count"]==1
    first_feature=backbone.last["feat"].clone()
    b["xyz"]=b["xyz"].repeat(1,4,1);b["features"]=b["features"].repeat(1,4,1)
    b["point_valid"]=b["point_valid"].repeat(1,4)
    second=g(b,hand,valid)
    torch.testing.assert_close(backbone.last["feat"],first_feature)
    second["hand"].sum().backward()
    assert hand.grad.abs().sum()>0


def test_explicit_semantic_hand_survives_colliding_geometry():
    m=model();b=batch()
    h,_=m.encode_hand(b)
    assert not torch.allclose(h[:,0],h[:,1])
    assert not torch.allclose(h[:,0],h[:,11])
