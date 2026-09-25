"""Coverage and mesh identity invariants for the full GRAB pool."""
import importlib.util
from pathlib import Path

import pytest
import torch

MODULE = Path(__file__).resolve().parents[4] / "third_party/DExplore/dexplore/utils/object_motion_sampler.py"
spec = importlib.util.spec_from_file_location("object_motion_sampler", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_pool_larger_than_environments_is_covered():
    objects = ["apple"] * 101 + ["mug"] * 33 + ["duck"] * 7
    sampler = module.ObjectMotionSampler(objects, 6, "cpu")
    for _ in range(51):
        env_ids = torch.arange(6)
        motions = sampler.sample(env_ids)
        assert torch.equal(sampler.motion_objects[motions], sampler.env_objects)
    assert (sampler.visits > 0).all()


def test_asynchronous_resets_keep_mesh_identity_and_exhaust_each_object():
    sampler = module.ObjectMotionSampler(["a", "b", "a", "c", "b", "a"], 7, "cpu")
    for _ in range(5):
        for env_ids in (torch.tensor([4, 0, 2]), torch.tensor([6, 1, 3, 5])):
            motions = sampler.sample(env_ids)
            assert torch.equal(sampler.motion_objects[motions], sampler.env_objects[env_ids])
    assert (sampler.visits > 0).all()
    assert sampler.sample(torch.empty(0, dtype=torch.long)).numel() == 0


def test_reject_unrepresented_objects():
    with pytest.raises(ValueError, match="at least one environment"):
        module.ObjectMotionSampler(["a", "b", "c"], 2, "cpu")
