"""Operational counters only; never calculate significance before n40 completion."""
import argparse,json,collections
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();groups=collections.defaultdict(lambda:dict(n=0,strict=0,hold=0,joint=0,other_abort=0));workers=[]
    receipt=a.run/'launch-detached.json'
    names=[Path(w['output']).name for w in json.loads(receipt.read_text())['workers']] if receipt.exists() else ['g4-resume','g5-resume','g6-resume']
    for name in names:
        file=a.run/name/'manifest.json'
        if not file.exists():workers.append(dict(worker=name,status='LOADING'));continue
        m=json.loads(file.read_text());workers.append(dict(worker=name,status=m['status'],done=len(m['runs']),planned=m['planned_episodes'],active=m.get('active_episode'),controls=m['environment_controls'],error=m.get('error')))
        for r in m['runs']:
            g=groups[r['subset']+'/'+r['arm']];g['n']+=1;g['strict']+=int(r['final_success']);g['hold']+=int(r['final_hold']);g['joint']+=int(r['physical_abort']=='physical_joint_limit');g['other_abort']+=int(r['physical_abort'] is not None and r['physical_abort']!='physical_joint_limit')
    print(json.dumps(dict(completed=sum(g['n'] for g in groups.values()),planned=160,groups=groups,workers=workers),ensure_ascii=False))


if __name__=='__main__':main()
