import pytest
from wam_rl_policy_v3.official_bridge import optimization_contract,validate_optimization


def test_legacy_shadow_resume_requires_explicit_kl_migration_but_evaluation_does_not():
    actual=optimization_contract('actual');legacy=optimization_contract('legacy')
    with pytest.raises(ValueError,match='adaptive KL contract changed'):validate_optimization(None,actual)
    assert validate_optimization(None,legacy)['changed'] is False
    assert validate_optimization(None,actual,evaluating=True)['changed']
    assert validate_optimization(None,actual,migration_reason='matched KL ablation')['changed']
    assert not validate_optimization(actual,actual)['changed']


def test_corrected_shadow_factory_keeps_original_collector_and_sampling_distribution(tmp_path):
    import ast
    from pathlib import Path
    from wam_rl_policy_v3.official_bridge import prepare_official
    root=Path(__file__).resolve().parents[4]
    prepared=prepare_official(root,tmp_path,correct_distribution_kl=True)
    source=(prepared/'algorithms/rl/ppo/ppo.py').read_text()
    assert 'actual_policy_kl' in source
    original=(root/'third_party/UniDexGrasp2-official-ppo/dexgrasp/algorithms/rl/ppo/ppo.py').read_text()
    def collector(text):
        cls=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='PPO')
        return ast.dump(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run'))
    assert collector(source)==collector(original)
    module=(prepared/'algorithms/rl/ppo/module.py').read_text()
    assert 'scale_tril=covariance' in module
