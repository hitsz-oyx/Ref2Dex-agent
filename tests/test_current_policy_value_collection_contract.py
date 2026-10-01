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
        "context": torch.zeros(3, contract.CONTEXT_DIM), "next_context": torch.zeros(3, contract.CONTEXT_DIM),
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


def test_context_contract_matches_saved_physical_model():
    checkpoint = contract.R7_ROOT / "models/tier_1000000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert contract.CONTEXT_DIM == payload["context_dim"]


def test_long_complete_panel_crosses_old_partial_flush_threshold(tmp_path):
    runtime = _native_runtime_module()
    result = runtime._mock_collect(tmp_path / "long-panel", 286, steps=400)
    assert result["rows"] == 38400
    from src.task.CmResidual.physical_value_data import Episodes
    episodes = Episodes([Path(result["run_dir"])], gamma=.99)
    assert episodes.episode_ids.numel() == 96
    assert episodes.excluded_rows == 0


def _native_runtime_module():
    import importlib.util
    path = ROOT / "scripts/run_current_policy_value_environment.py"
    spec = importlib.util.spec_from_file_location("current_policy_runtime_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _native_player_fixture():
    from types import SimpleNamespace
    from rl_games.algos_torch.models import ModelA2CContinuousLogStd

    class RawNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.last_obs = None

        def forward(self, input_dict):
            self.calls += 1
            obs = input_dict["obs"]
            self.last_obs = obs.detach().clone()
            mu = torch.full((obs.shape[0], 18), 3.0, dtype=obs.dtype)
            logstd = torch.full_like(mu, -1.0)  # negative logstd, not sigma
            value = obs[:, :1] * 2.0
            return mu, logstd, value, input_dict.get("rnn_states")

    raw = RawNetwork()
    model = ModelA2CContinuousLogStd.Network(
        raw, obs_shape=(4,), normalize_value=False, normalize_input=True, value_size=1
    )
    model.eval()
    with torch.no_grad():
        model.running_mean_std.running_mean.copy_(torch.tensor([1.0, -1.0, 2.0, -2.0], dtype=torch.float64))
        model.running_mean_std.running_var.copy_(torch.tensor([4.0, 9.0, 16.0, 25.0], dtype=torch.float64))
        model.running_mean_std.count.copy_(torch.tensor(100.0, dtype=torch.float64))
    player = SimpleNamespace(
        model=model, states=None, clip_actions=True,
        actions_low=torch.full((18,), -2.0), actions_high=torch.full((18,), 2.0),
        normalize_value=False,
        _preproc_obs=lambda obs: obs,
    )
    return player, raw


def test_native_model_player_equivalence_normalizes_exp_logstd_and_clips():
    runtime = _native_runtime_module()
    player, raw = _native_player_fixture()
    observation = {"obs": torch.tensor([[5.0, 2.0, 6.0, 3.0]])}
    initial_state = torch.tensor([[7.0]])
    player.states = initial_state
    torch.manual_seed(1234)
    expected = player.model({"is_train": False, "prev_actions": None,
                             "obs": observation["obs"], "rnn_states": initial_state})
    torch.manual_seed(1234)
    player.states = initial_state
    action, value, result = runtime._native_actor_forward(player, observation)
    assert raw.calls == 2  # one direct native reference call and one helper call
    assert torch.equal(player.states, result["rnn_states"]) if isinstance(result["rnn_states"], torch.Tensor) else result["rnn_states"] is None
    expected_action = torch.clamp(expected["actions"], -1.0, 1.0) * 2.0
    assert torch.allclose(action, expected_action)
    assert torch.allclose(value, expected["values"].reshape(-1))
    assert torch.all(action <= 2.0) and torch.all(action >= -2.0)
    normalized = player.model.norm_obs(observation["obs"])
    assert torch.allclose(raw.last_obs, normalized)
    assert torch.all(expected["sigmas"] > 0.0)


def test_native_critic_is_normalized_and_consumes_no_rng_for_dict_or_tensor():
    runtime = _native_runtime_module()
    player, raw = _native_player_fixture()
    tensor_obs = torch.tensor([[5.0, 2.0, 6.0, 3.0]])
    state_before = torch.random.get_rng_state()
    value_tensor = runtime._native_critic_value(player, tensor_obs)
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)
    assert raw.calls == 1
    assert torch.allclose(raw.last_obs, player.model.norm_obs(tensor_obs))
    value_dict = runtime._native_critic_value(player, {"obs": tensor_obs})
    assert torch.allclose(value_tensor, value_dict)


def test_native_guard_rejects_unknown_reward_shaper_and_accepts_installed_object():
    runtime = _native_runtime_module()
    from rl_games.common.tr_helpers import DefaultRewardsShaper
    runtime._validate_value_config({"normalize_value": False,
                                    "reward_shaper": DefaultRewardsShaper(scale_value=1)})
    with pytest.raises(ValueError, match="reward scale"):
        runtime._validate_value_config({"normalize_value": False,
                                        "reward_shaper": DefaultRewardsShaper(scale_value=.5)})
    with pytest.raises(ValueError, match="reward scale"):
        runtime._validate_value_config({"normalize_value": False, "reward_shaper": object()})
    with pytest.raises(TypeError, match="Mapping"):
        runtime._validate_value_config(object())


@pytest.mark.parametrize("seed", [286, 287])
def test_saved_dexplore_player_actions_critic_and_external_normalizer(seed):
    """Exercise the actual saved architecture and CommonPlayer preprocessing."""
    import importlib.util
    import yaml
    from rl_games.algos_torch.running_mean_std import RunningMeanStd

    def load_module(name, relative):
        spec = importlib.util.spec_from_file_location(name, MAIN_ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    builder_module = load_module("diagnostic_dexplore_builder",
        "third_party/DExplore/dexplore/learning/dexplore_network_builder.py")
    model_module = load_module("diagnostic_dexplore_model",
        "third_party/DExplore/dexplore/learning/dexplore_models.py")
    player_module = load_module("diagnostic_common_player",
        "third_party/DExplore/dexplore/learning/common_player.py")
    checkpoint = torch.load(contract.CHECKPOINTS[seed], map_location="cpu", weights_only=False)
    network_params = yaml.safe_load(contract.TRAIN_CONFIG.read_text())["params"]["network"]
    builder = builder_module.DexploreBuilder()
    builder.load(network_params)
    prefix = "_orig_mod."
    raw_state = {(k[len(prefix):] if k.startswith(prefix) else k): v
                 for k, v in checkpoint["model"].items()}
    obs_dim = raw_state["a2c_network.actor_mlp.0.weight"].shape[1]
    model = model_module.ModelDexploreContinuous(builder).build({
        "actions_num": 18, "input_shape": (obs_dim,), "num_seqs": 1,
        "value_size": 1, "normalize_value": False, "normalize_input": False,
    })
    model.load_state_dict(raw_state)
    model.eval()
    player = player_module.CommonPlayer.__new__(player_module.CommonPlayer)
    player.model = model
    player.normalize_input = True
    player.normalize_value = False
    player.running_mean_std = RunningMeanStd((obs_dim,))
    player.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    player.running_mean_std.eval()
    player.states = None
    player.has_batch_dimension = True
    player.clip_actions = True
    player.actions_low = torch.full((18,), -1.0)
    player.actions_high = torch.full((18,), 1.0)
    obs = player.running_mean_std.running_mean.float()[None].repeat(2, 1)
    obs = obs + .2 * player.running_mean_std.running_var.float().sqrt()[None]
    runtime = _native_runtime_module()
    normalized = player._preproc_obs(obs)
    assert not torch.allclose(obs, normalized)
    with torch.no_grad():
        _, logstd, expected_value, _ = model.a2c_network({"obs": normalized, "rnn_states": None})
        assert logstd.lt(0).all()
        with pytest.raises(RuntimeError, match="std >= 0"):
            torch.normal(torch.zeros_like(logstd), logstd)
    torch.manual_seed(12345)
    expected_action = player.get_action({"obs": obs}, False)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(12345)
    action, value, _ = runtime._native_actor_forward(player, {"obs": obs})
    assert torch.equal(action, expected_action)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.allclose(value, expected_value.reshape(-1))
    critic = runtime._native_critic_value(player, obs)
    assert torch.allclose(critic, expected_value.reshape(-1))
    assert not critic.requires_grad
    assert torch.equal(torch.get_rng_state(), expected_rng)
