"""Final actual actor evaluation with declared cold or equal-budget controls."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_trained_option_panel import build_source as original
    source=original();changes={
        'a.panel not in (611,612)':'a.panel not in (655,656,657,658)',
        "m['experiment_id']!='P-20261002-option-model-policy'":"m['experiment_id']!='P-20261003-budgeted-physical-critic'",
        "actor_file=root/'fit/actors.pt'":"actor_file=root/('fit/actors_budget.pt' if a.panel in (657,658) else 'fit/actors.pt')",
    }
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    assert source.count("m['trained_actor_sha256']")==3;source=source.replace("m['trained_actor_sha256']","m['panel_actor_sha256'][str(a.panel)]")
    marker="        decision=torch.load(directory/'option_decisions.pt'";assert source.count(marker)==1
    source=source.replace(marker,"        if actor_bundle['control_name']!=('budget_q' if a.panel in (657,658) else 'cold_q'):raise ValueError('actual declared budget control')\n"+marker)
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT));exec(compile(build_source(),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})
