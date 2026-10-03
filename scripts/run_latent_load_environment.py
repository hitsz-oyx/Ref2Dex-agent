"""Corrected native physics with pre-prepare hidden density contrast."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build_source():
    from scripts.run_inspire_filter_environment import build_source as original
    source = original()
    changes = {
        '        import evaluate as original':
        '''        import evaluate as original
        from src.task.CmResidual.latent_load import install_preparation_hook,EVALUATION_SEED
        if args.eval_seed!=EVALUATION_SEED:raise ValueError('fixed prospective seed')
        install_preparation_hook(args.eval_seed)''',
        'physical_metadata=dict(':
        'physical_metadata=dict(latent_load=task._latent_load,force_normalization_mass_kg=[p[\'mass\'] for p in task._latent_load[\'before\']],actual_mass_not_a_policy_feature=True,',
        "weight_np=np.array([p['mass'] for p in body_properties])*":
        "weight_np=np.array([p['mass'] for p in task._latent_load['before']])*",
        '            _,option_cluster,option_raw12=matched_inputs':
        '''            if motion.cpu().tolist()!=task._latent_load['creation_motion'] or assignment.cpu().tolist()!=task._latent_load['creation_groups']:raise ValueError('creation/reset assignments disagree')
            _,option_cluster,option_raw12=matched_inputs''',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    exec(compile(build_source(), str(ROOT / 'scripts/run_continuous_critic_environment_v2.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT / 'scripts/run_continuous_critic_environment_v2.py')})
