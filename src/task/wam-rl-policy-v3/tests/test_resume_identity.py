import pytest
from wam_rl_policy_v3.resume_identity import validate_resume_identity


def test_backend_change_requires_explicit_migration_and_keeps_other_identity_strict():
    saved = dict(physics_backend='cpu', wam='wam80000', envs=512, world_size=3)
    target = dict(saved, physics_backend='gpu')
    assert validate_resume_identity(saved, saved) is None
    with pytest.raises(ValueError):
        validate_resume_identity(saved, target)
    assert validate_resume_identity(saved, target, migrate_physics=True) == dict(
        kind='explicit_physics_backend', source='cpu', target='gpu')
    for field, value in [('wam', 'other'), ('envs', 256), ('world_size', 1)]:
        with pytest.raises(ValueError):
            validate_resume_identity(saved, dict(target, **{field: value}), migrate_physics=True)
    with pytest.raises(ValueError):
        validate_resume_identity(saved, dict(target, physics_backend='unknown'), migrate_physics=True)
