"""Frozen actual full-success support screen, not a controller-utility claim."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();assert not (root/'results.json').exists()
    counts=np.zeros((2,3,4),np.int64);transient=counts.copy();totals=counts.copy()
    for si,seed in enumerate((629,630)):
        d=root/f's{seed}';assert json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED';rows=json.loads((d/'rows.json').read_text());assert len(rows)==768 and [r['environment'] for r in rows]==list(range(768))
        for row in rows:
            assert row['seed']==seed;idx=si,row['motion'],row['arm'];totals[idx]+=1;counts[idx]+=int(row['physical105']);transient[idx]+=int(row['any_joint_lift_in105'])
    assert np.all(totals==64);random_motion0=counts[:,0,2:].sum(-1);gates=dict(random_motion0_complete_at_least8=int(random_motion0.sum())>=8,random_motion0_complete_on_each_seed=bool(np.all(random_motion0>=1)))
    names=('p0','duplicate_p0','gaussian','antithetic');pooled=counts.sum(0);tr=transient.sum(0)
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,random_motion0_complete_per256=int(random_motion0.sum()),random_motion0_per_seed_per128={str(seed):int(random_motion0[i]) for i,seed in enumerate((629,630))},per_motion_complete_per128_per_arm={str(m):dict(zip(names,pooled[m].tolist())) for m in range(3)},per_motion_transient_per128_per_arm={str(m):dict(zip(names,tr[m].tolist())) for m in range(3)},per_seed_motion_complete_per64={str(seed):{str(m):dict(zip(names,counts[i,m].tolist())) for m in range(3)} for i,seed in enumerate((629,630))},native_trajectories=1536,native_control_ticks_each=202,preparation_ticks=24,new_optimizer_steps=0,no_cm_model_calls=True,no_same_state_oracle_or_policy_gain_claim=True,formal_validation=False)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
