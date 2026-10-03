"""Independent actor/native replay and declared shape ownership filter check."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_trained_option_panel import build_source as original
    source=original().replace('a.panel not in (611,612)','a.panel!=655').replace("m['experiment_id']!='P-20261002-option-model-policy'","m['experiment_id']!='P-20261003-inspire-filter-impact'")
    marker="        physical=json.loads((directory/'physical_metadata.json').read_text())"
    assert source.count(marker)==1
    source=source.replace(marker,marker+'''
        ownership=physical['native_shape_ownership']
        expected=[3 if ('thumb' in owner and 'distal' in owner) or ('thumb' not in owner and 'intermediate' in owner) else 2 for owner in ownership]
        if len(ownership)!=13 or len(physical['native_body_names'])!=25 or len(physical['hand_shape_filters'])!=768 or len(physical['table_shape_filters'])!=768:raise ValueError('actual shape metadata coverage')
        if any(values!=expected for values in physical['hand_shape_filters']) or any(not values or any(value!=1 for value in values) for values in physical['table_shape_filters']):raise ValueError('actual native collision filters')
''')
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    exec(compile(build_source(),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})
