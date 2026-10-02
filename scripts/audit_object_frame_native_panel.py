"""Only576/u20 stochastic engineering capture; unchanged original physics checks."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/audit_continuous_critic_panel.py').read_text()
    changes={"a.panel not in list(range(547,567))+[568,569]":"a.panel != 576",
        "m['experiment_id']!='P-20261002-continuous-critic-policy'":"m['experiment_id']!='ENG-20261002-object-frame-kinematics'",
        "r['deterministic']!=(a.panel in (568,569))":"r['deterministic'] or panel_record['update']!=20"}
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('original audit marker drift')
        source=source.replace(old,new)
    exec(compile(source,str(ROOT/'scripts/audit_continuous_critic_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_continuous_critic_panel.py')})

if __name__=='__main__':main()
