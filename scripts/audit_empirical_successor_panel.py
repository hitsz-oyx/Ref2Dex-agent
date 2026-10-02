"""Actual actor/native replay for empirical physical-successor training."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_trained_option_panel import build_source as original
    source=original();changes={'a.panel not in (611,612)':'a.panel not in (623,624)',"m['experiment_id']!='P-20261002-option-model-policy'":"m['experiment_id']!='P-20261002-empirical-successor-policy'"}
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    exec(compile(build_source(),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})
