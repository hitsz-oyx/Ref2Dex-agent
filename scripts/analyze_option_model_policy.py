"""Frozen actual trained-policy outcome gates, without replay/oracle assumptions."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve()
    counts=np.zeros((2,3,4),dtype=np.int64);totals=counts.copy()
    for s,seed in enumerate((611,612)):
        d=root/f's{seed}';assert json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED';rows=json.loads((d/'rows.json').read_text());assert len(rows)==768
        for row in rows:
            assert row['seed']==seed;counts[s,row['motion'],row['arm']]+=int(row['physical105']);totals[s,row['motion'],row['arm']]+=1
    assert np.all(totals==64)
    pooled=counts.sum((0,1));pooled_rates=pooled/384;byseed=counts.sum(1);motion=counts.sum(0)
    gates=dict(gain5pp_over_all_three=all(pooled_rates[1]>=pooled_rates[k]+.05 for k in (0,2,3)),each_seed_noninferiority_all=all(byseed[s,1]>=byseed[s,k] for s in range(2) for k in (0,2,3)),motion1_loss_at_most5pp_vs_p0=motion[1,1]/128>=motion[1,0]/128-.05)
    names=('p0','cm','dynamics_off','direct_q')
    result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',physical105_counts_per384=dict(zip(names,pooled.tolist())),success_rates=dict(zip(names,pooled_rates.tolist())),per_seed_counts_per192={str(seed):dict(zip(names,byseed[s].tolist())) for s,seed in enumerate((611,612))},per_motion_counts_per128={str(m):dict(zip(names,motion[m].tolist())) for m in range(3)},gates=gates,actual_trained_actors_evaluated=True,no_model_at_inference=True,no_replay_matching_or_retrospective_oracle=True,pretraining_trajectories=1536,evaluation_trajectories=1536,formal_validation=False,distinctive_method_unproved=True)
    assert not (root/'results.json').exists();(root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
