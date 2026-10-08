"""Regression checks for the actual legacy runner/RMS integration seam."""
import importlib.util
import pytest
from pathlib import Path
from types import SimpleNamespace


def entry():
    path = Path(__file__).parents[1]/'tools/audit/probe_official_generator.py'
    spec = importlib.util.spec_from_file_location('official_screen_entry',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_factory_and_observation_normalizer():
    calls = []
    class Factory:
        def create(self, name, *, params):
            calls.append((name,params))
            return params
    class Runner:
        def create_player(self):
            # This is the old runner's keyword that broke the new entry.
            return self.player_factory.create(self.algo_name,config=self.config)
    class Base:
        def _preproc_obs(self, obs):
            return self.running_mean_std(obs)
    class Common(Base):
        def _preproc_obs(self, obs):
            # Newer rl_games needs this override; old rl_games does not.
            return self.running_mean_std(super()._preproc_obs(obs))
    native = SimpleNamespace(Runner=Runner)
    entry().install_legacy_player_compat(native,SimpleNamespace(CommonPlayer=Common),
                                       SimpleNamespace(BasePlayer=Base))
    runner = Runner()
    runner.player_factory,runner.algo_name = Factory(),'dexplore'
    runner.config = {'network':object(),'env_name':'rlgpu'}
    assert runner.create_player() is runner.config
    assert calls == [('dexplore',runner.config)]
    normalizations = []
    player = Common()
    def rms(obs):
        normalizations.append(obs)
        return obs+1
    player.running_mean_std = rms
    assert player._preproc_obs(3) == 4
    assert normalizations == [3]


def test_compiled_checkpoint_translation_is_strict():
    tensor = object()
    state = {'_orig_mod.a2c_network.mu.weight':tensor}
    assert entry().model_state(state) == {'a2c_network.mu.weight':tensor}
    assert next(iter(entry().model_state(state).values())) is tensor
    ordinary = {'a2c_network.mu.weight':tensor}
    assert entry().model_state(ordinary) is ordinary
    with pytest.raises(ValueError,match='mixed'):
        entry().model_state(dict(state,another=tensor))
