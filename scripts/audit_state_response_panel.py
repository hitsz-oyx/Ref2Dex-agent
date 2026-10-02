"""Reuse unchanged native checks, with exactly one new stochastic u00 cohort."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/audit_continuous_critic_panel.py').read_text()
    changes={
        "a.panel not in list(range(547,567))+[568,569]":"a.panel != 570",
        "m['experiment_id']!='P-20261002-continuous-critic-policy'":"m['experiment_id']!='P-20261002-state-response-field'",
        "r['deterministic']!=(a.panel in (568,569))":"r['deterministic'] or panel_record['update']!=0",
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('unexpected audit source; no broad mutation')
        source=source.replace(old,new)
    scope={'__name__':'__main__','__file__':str(ROOT/'scripts/audit_continuous_critic_panel.py')}
    exec(compile(source,str(ROOT/'scripts/audit_continuous_critic_panel.py'),'exec'),scope)

if __name__=='__main__':main()
