"""Counts for a frozen coverage panel; no n40 statistics on single-seed families."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from verify_large_initials import compare,FIELDS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();reg=json.loads((a.run/'registration.json').read_text());rows=[];controls=0
    for g in (4,5,6):
        m=json.loads((a.run/('g%d'%g)/'manifest.json').read_text())
        if m['protocol']!=reg['protocol'] or m['status']!='COMPLETED':raise ValueError('incomplete or mixed workers')
        controls+=m['environment_controls'];rows+=[dict(r,run=str(a.run/('g%d'%g))) for r in m['runs']]
    planned={j['id']+'-'+arm for j in reg['jobs'] for arm in j['order']};ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=planned:raise ValueError('missing/duplicate/extra episode')
    initial_parity=[]
    for job in reg['jobs']:
        packets=[]
        for row in rows:
            if row['trial']!=job['id']:continue
            with np.load(Path(row['run'])/row['id']/'initial.npz') as z:packets.append({k:z[k].copy() for k in FIELDS})
        initial_parity.append(dict(trial=job['id'],**compare(packets)))
    if not all(r['match'] for r in initial_parity):raise ValueError('coverage arms have different measured initial states')
    groups={}
    for subset in reg['subsets']:
        groups[subset]={}
        for arm in reg['arms']:
            records=[r for r in rows if r['subset']==subset and r['arm']==arm]
            groups[subset][arm]=dict(n=len(records),success=sum(r['final_success'] for r in records),hold=sum(r['final_hold'] for r in records),
                aborts=dict(collections.Counter(r['physical_abort'] for r in records if r['physical_abort'])),source_admission_failures=sum(r.get('source_admission_failures',0) for r in records),
                errors_mm=[r['final_goal_error_m']*1000 for r in records],tolerances_mm=[r['geometry']['position_tolerance_m']*1000 for r in records])
    result=dict(status='COMPLETED',n=len(rows),environment_controls=controls,groups=groups,initial_parity=initial_parity,rows=rows,scope=reg['scope'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
