"""Check saved A2 really consumed R and removed all future content, not only tau8."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
    for path in a.manifest:
        m=json.loads(path.read_text())
        for r in m['runs']:
            if r['arm']!='A2':continue
            with np.load(path.parent/r['id']/'trajectory.npz') as z:
                tau=z['tau'];forecast=z['forecast_hand'];hand=z['hand'][:,0];valid=z['tau_valid'];obs=z['observation'];adapter=z['adapter_raw_action'];actions=z['action']
                content=float(np.abs(tau).max());plan_error=float(np.abs(forecast[valid]-hand[valid,None]).max()) if valid.any() else 0.
                mask_match=np.array_equal(obs[:,-1],valid.astype(np.float32));r_count=bool(len(adapter)==len(actions)==r['actor_calls']==r['controls'])
                actor_nonzero=bool(np.abs(adapter).max()>1e-6) if len(adapter) else False
                match=bool(content==0 and plan_error==0 and mask_match and r_count and actor_nonzero)
            records.append(dict(id=r['id'],subset=r['subset'],controls=r['controls'],actor_calls=r['actor_calls'],tau_max=content,full24_plan_content_error=plan_error,valid_mask_match=mask_match,R_count_match=r_count,R_nonzero=actor_nonzero,match=match))
    report=dict(status='PASS' if records and all(r['match'] for r in records) else 'FAIL',n=len(records),matched=sum(r['match'] for r in records),records=records,scope='raw input/output and call-count evidence; common-state byte parity is a separate unit test, no claim different closed-loop states have identical observations')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('records','scope')}));return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
