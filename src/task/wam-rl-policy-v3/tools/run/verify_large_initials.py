"""Read-only nine-field initial parity across complete v9 four-arm pairs."""
import argparse,json
from pathlib import Path
import numpy as np
FIELDS=('q','dq','hand','objects','physical','tick','table_pose','contact_force_valid','dof_force_valid')


def compare(packets,tolerance=2e-6):
    errors={key:max(float(np.abs(p[key].astype(float)-packets[0][key].astype(float)).max()) for p in packets) for key in FIELDS}
    return dict(match=max(errors.values())<=tolerance,errors=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();groups={}
    for path in a.manifest:
        m=json.loads(path.read_text())
        for r in m['runs']:
            byarm=groups.setdefault(r['trial'],{})
            if r['arm'] in byarm:raise ValueError('duplicate arm, do not count carried results twice')
            with np.load(path.parent/r['id']/'initial.npz') as z:byarm[r['arm']]={k:z[k].copy() for k in FIELDS}
    rows=[];mutant=False
    for trial,byarm in sorted(groups.items()):
        if set(byarm)!={'A1','A2','A3','A4'}:continue
        packets=[byarm[a] for a in sorted(byarm)];rows.append(dict(trial=trial,**compare(packets)))
        if not mutant:
            altered={k:v.copy() for k,v in packets[1].items()};altered['hand'][0,0,0]+=.01
            mutant=not compare([packets[0],altered])['match']
    report=dict(status='PASS' if all(r['match'] for r in rows) and mutant else 'FAIL',complete_pairs=len(rows),matched=sum(r['match'] for r in rows),initial_translation_mutant_killed=mutant,records=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}));return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
