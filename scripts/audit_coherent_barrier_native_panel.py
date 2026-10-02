"""Exactly three new scientific cohorts; same validated coherent physics audit."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/audit_coherent_geometry_native_panel.py').read_text()
    for old,new in [('a.panel != 583','a.panel not in (584,585,586)'),('ENG-20261002-coherent-native-contract','P-20261002-coherent-barrier-information')]:
        if source.count(old)!=1:raise ValueError('engineering audit wrapper drift '+old)
        source=source.replace(old,new)
    exec(compile(source,str(ROOT/'scripts/audit_coherent_geometry_native_panel.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_coherent_geometry_native_panel.py')})

if __name__=='__main__':main()
