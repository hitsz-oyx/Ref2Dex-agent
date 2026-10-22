"""Coverage selection respects exposure and n40 without using policy outcomes."""
import importlib.util
from pathlib import Path
from collections import Counter


def module():
    p=Path(__file__).resolve().parents[1]/'tools/run/prepare_grasp_coverage40.py';spec=importlib.util.spec_from_file_location('coverage40_registration',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_forty_distinct_initials_twenty_unexposed_objects_and_probe_seeds():
    m=module();records=[dict(object='o%02d'%i,sequence='s%d_o%02d'%(s,i),engineering='PASS',reference='r',split='train') for i in range(21) for s in (1,2,3)]
    chosen=m.select(records,{'dexplore/'+r['object'] for r in records},{'o00'})
    assert len(chosen)==40 and 'o00' not in {r['object'] for r in chosen}
    assert set(Counter(r['object'] for r in chosen).values())=={2}
    jobs=m.assignments(chosen)
    assert len({j['sequence'] for j in jobs})==40
    assert all(100<=j['native_seed']<300 and 100<=j['model_seed']<300 for j in jobs)
    assert all(set(j['order'])=={'A1','A2','A0'} for j in jobs)
    # Every arm occupies each order position 13 or14 times, not one GPU/order.
    for arm in ('A1','A2','A0'):assert set(Counter(j['order'].index(arm) for j in jobs).values())<={13,14}


def test_cached_geometry_requires_requested_unit_scale():
    m=module()
    assert m.codes({'objects':[{'key':'dexplore/bowl@1'},{'key':'dexplore/camera@1.0'},{'key':'dexplore/knife@0.5'}]})=={'dexplore/bowl','dexplore/camera'}
