"""Full native/private Gaussian replay; all three stochastic arms share behavior."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.audit_statistical_option_panel import build_source as original
    source=original()
    changes={
        'a.panel not in (603,604)':'a.panel!=618',
        "m['experiment_id']!='P-20261002-option-model-policy'":"m['experiment_id']!='P-20261002-physical-gradient-control'",
        'options=torch.randn((192,12),generator=g).numpy()':'options=torch.randn((768,12),generator=g).numpy()',
        'if group in (2,3):option_raw[envs]=options[slots]*(1 if group==2 else -1)':'if group in (1,2,3):option_raw[envs]=options[envs]',
    }
    for old,new in changes.items():assert source.count(old)==1,old;source=source.replace(old,new)
    marker="        if not r['statistical_option_collection']"
    assert source.count(marker)==1
    source=source.replace(marker,"        if not r['independent_gaussian_options'] or r['gaussian_option_sigma']!=1.:raise ValueError('same Gaussian behavior')\n"+marker)
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    exec(compile(build_source(),str(ROOT/'scripts/audit_truth_successor_native_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_truth_successor_native_panel.py')})
