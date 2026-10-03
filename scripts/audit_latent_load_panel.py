"""Raw physics/assignment/feature/actor/full105 replay, independent mass draw."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_inspire_filter_panel import build_source as original
    source = original()
    changes = {
        'a.panel!=655': 'a.panel!=701',
        "m['experiment_id']!='P-20261003-inspire-filter-impact'":
        "m['experiment_id']!='P-20261003-latent-load-feasibility'",
        "or any(x['flags'] or abs(x['mass']-.0025936129968613386)>1e-9 for x in physical['object_body_properties'])":
        "or any(x['flags'] for x in physical['object_body_properties'])",
        'current,extra=independent_points(initial,data,physical,base_checkpoint,steps)':
        '''feature_metadata=dict(physical,object_body_properties=[dict(p,mass=mass) for p,mass in zip(physical['object_body_properties'],physical['force_normalization_mass_kg'])])
        current,extra=independent_points(initial,data,feature_metadata,base_checkpoint,steps)''',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    marker = "        ownership=physical['native_shape_ownership']"
    block = '''        contract=physical['latent_load'];heavy=np.zeros(768,dtype=np.int64)
        generator=torch.Generator(device='cpu').manual_seed(seed+31000)
        for mo in range(3):
            for group in range(4):
                indices=np.flatnonzero((motion==mo)&(arm==group))
                chosen=indices[torch.randperm(64,generator=generator).numpy()[:32]]
                heavy[chosen]=1
        if contract['creation_motion']!=motion.tolist() or contract['creation_groups']!=arm.tolist() or not np.array_equal(contract['load'],heavy) or not contract['changed_before_prepare']:raise ValueError('prospective hidden-load assignment replay')
        nominal=np.array([p['mass'] for p in contract['before']])
        actual=np.array([p['mass'] for p in physical['object_body_properties']])
        factor=np.where(heavy,50.,1.)
        if np.max(np.abs(nominal-.0025936129968613386))>1e-9 or not np.array_equal(nominal,physical['force_normalization_mass_kg']) or not physical['actual_mass_not_a_policy_feature']:raise ValueError('nominal normalization versus actual physics')
        if not np.allclose(actual,nominal*factor,rtol=2e-6,atol=1e-10):raise ValueError('actual SDK physical mass')
        for env_index in range(768):
            pre=np.array(contract['before'][env_index]['inertia']);post=np.array(physical['object_body_properties'][env_index]['inertia'])
            if not np.allclose(pre,contract['before'][0]['inertia'],rtol=0,atol=1e-12) or np.min(np.linalg.eigvalsh(pre))<=0 or not np.allclose(post,pre*factor[env_index],rtol=2e-6,atol=1e-12) or not np.allclose(post,contract['after'][env_index]['inertia'],rtol=0,atol=1e-12):raise ValueError('all physical inertia tensors')
'''
    assert source.count(marker) == 1
    source = source.replace(marker, block + marker)
    marker = "motion=int(motion[env]),arm=int(arm[env]),"
    assert source.count(marker) == 1
    source = source.replace(marker, marker + 'load=int(heavy[env]),')
    marker = 'report=dict('
    assert source.count(marker) == 1
    source = source.replace(marker, 'report=dict(all768_load_mass_inertia_and_nominal_feature_weights_verified=True,')
    return source


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    exec(compile(build_source(), str(ROOT / 'scripts/audit_truth_successor_native_panel.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT / 'scripts/audit_truth_successor_native_panel.py')})
