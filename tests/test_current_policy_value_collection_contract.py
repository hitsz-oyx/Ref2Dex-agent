from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/collect_current_policy_value.py"
MAIN_ROOT = Path("/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent").resolve()
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(MAIN_ROOT))
import collect_current_policy_value as contract


def test_preflight_is_source_only_and_matches_native_panel(tmp_path):
    output = subprocess.check_output(
        [sys.executable, str(SCRIPT), "--preflight", "--run-root", str(tmp_path / "new")],
        cwd=ROOT, text=True,
    )
    report = json.loads(output)
    assert report["status"] == "READY_FOR_ROOT_RUNTIME_TASK"
    assert report["source_commit"] == contract.SOURCE_COMMIT
    assert report["isaac_imported"] is False
    assert report["collection_started"] is False
    assert [p["seed"] for p in report["panels"]] == [286, 287]
    assert all(p["checkpoint_sha256"] == contract.CHECKPOINT_SHA256[p["seed"]] for p in report["panels"])


def test_terminal_reset_boundary_and_complete_episode_fixture():
    fixture = contract.synthetic_contract_fixture()
    assert fixture["validation"]["episodes"] == 2
    assert fixture["reset_boundary_preserved"] is True


def test_gaussian_sample_is_clipped_without_extra_noise():
    generator = torch.Generator().manual_seed(290)
    mean = torch.full((4, contract.ACTION_DIM), 2.0)
    sigma = torch.full_like(mean, 0.1)
    action = contract.sample_gaussian_action(mean, sigma, generator)
    assert torch.all(action <= 1) and torch.all(action >= -1)
    assert contract._source_contract(contract.SOURCE_RUNNER.read_text())["new_entry_sampling"]["extra_noise"] is False


def test_episode_rejects_reset_inside_episode():
    rows = contract.synthetic_contract_fixture()
    # Rebuild the small fixture then make an intermediate row terminal.
    generator = torch.Generator().manual_seed(290)
    action = contract.sample_gaussian_action(torch.zeros(3, 18), torch.ones(3, 18), generator)
    rewards = torch.zeros(3, 5)
    data = {
        "state": torch.zeros(3, 55), "next_state": torch.zeros(3, 55),
        "context": torch.zeros(3, 605), "next_context": torch.zeros(3, 605),
        "action": action, "previous_action": torch.zeros(3, 18),
        "reward": rewards.sum(-1), "reward_components": rewards,
        "done": torch.tensor([True, False, True]), "terminate": torch.tensor([True, False, True]),
        "timeout": torch.zeros(3, dtype=torch.bool), "episode_id": torch.tensor([1, 1, 1]),
        "env_id": torch.zeros(3, dtype=torch.long), "step": torch.tensor([0, 1, 2]),
        "motion_id": torch.zeros(3, dtype=torch.long), "start_frame": torch.zeros(3, dtype=torch.long),
        "checkpoint_seed": torch.full((3,), 286, dtype=torch.long),
        "value_at_state": torch.zeros(3), "next_value": torch.zeros(3),
        "global_tick": torch.arange(3, dtype=torch.long),
        "horizon_phase": torch.arange(3, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="episode"):
        contract.validate_episode_rows(data)


def test_collect_entry_delegates_to_mock_runtime_and_is_episodes_compatible(tmp_path):
    output_root = tmp_path / "mock-runtime"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--collect", "--mock",
         "--checkpoint-seed", "286", "--run-root", str(output_root)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["episodes_compatible"] is True
    assert result["rows"] == 96 * 33
    assert result["phase_max"] == 32
    run_dir = Path(result["run_dir"])
    assert run_dir.is_dir()
    from src.task.CmResidual.physical_value_data import Episodes
    episodes = Episodes([run_dir], gamma=.99)
    assert episodes.episode_ids.numel() == 96
    payload = torch.load(run_dir / "transitions_000.pt", map_location="cpu", weights_only=False)
    assert {"value_at_state", "next_value", "global_tick", "horizon_phase"} <= set(payload)
    assert payload["episode_id"].dtype == torch.int64
    assert payload["global_tick"].unique().numel() == 33
    assert payload["horizon_phase"].max().item() == 31
    assert payload["next_value"][payload["done"]].eq(0).all()
    assert {"stable_success", "drop_after_success", "max_hold_seconds"} <= set(
        json.loads((run_dir / "results.json").read_text())["per_episode"][0]
    )


def test_collect_entry_stays_lazy_and_requires_explicit_gpu_for_real_runtime():
    text = SCRIPT.read_text()
    assert "from isaacgym" not in text
    wrapper = (ROOT / "scripts/run_current_policy_value_environment.py").read_text()
    assert wrapper.index("from isaacgym import gymapi") < wrapper.index("import torch  # noqa: F401")
    assert "source.original.main()" in wrapper
    assert "source.SOURCE_SHA =" not in wrapper
    assert contract.main(["--collect", "--checkpoint-seed", "286", "--run-root", "/tmp/unused"]) == 2
