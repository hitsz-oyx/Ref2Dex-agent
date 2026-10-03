"""Frozen measured policy gains, with both equal-budget and matched aux controls."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve();assert not (root/'results.json').exists();counts=np.zeros((4,3,4),np.int64);totals=counts.copy()
    for i,seed in enumerate((655,656,657,658)):
        d=root/f's{seed}';assert json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED';rows=json.loads((d/'rows.json').read_text());assert len(rows)==768
        for r in rows:assert r['seed']==seed;counts[i,r['motion'],r['arm']]+=int(r['physical105']);totals[i,r['motion'],r['arm']]+=1
    assert np.all(totals==64);pooled=counts.sum((0,1));blockA=counts[:2].sum((0,1));blockB=counts[2:].sum((0,1));seed=counts.sum(1);motion=counts.sum(0)
    gates=dict(gain5pp_vs_off_pooled=pooled[1]/768>=pooled[2]/768+.05,gain5pp_vs_p0_pooled=pooled[1]/768>=pooled[0]/768+.05,gain5pp_vs_cold_q_blockA=blockA[1]/384>=blockA[3]/384+.05,gain5pp_vs_equal_budget_q_blockB=blockB[1]/384>=blockB[3]/384+.05,each_seed_noninferior_all_present=all(seed[i,1]>=seed[i,k] for i in range(4) for k in (0,2,3)),motion1_loss_at_most5pp_vs_p0=motion[1,1]/256>=motion[1,0]/256-.05)
    gates={k:bool(v) for k,v in gates.items()};names=('p0','cm','dynamics_off','control_q');result=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,pooled_physical105_cm_off_p0_per768={n:int(pooled[i]) for i,n in enumerate(names[:3])},blockA_counts_per384=dict(zip(('p0','cm','dynamics_off','cold_q'),blockA.tolist())),blockB_counts_per384=dict(zip(('p0','cm','dynamics_off','equal_budget_q'),blockB.tolist())),per_seed_counts_per192={str(s):dict(zip(names,seed[i].tolist())) for i,s in enumerate((655,656,657,658))},per_motion_pooled_per256_cm_off_p0={str(m):dict(zip(names[:3],motion[m,:3].tolist())) for m in range(3)},per_motion_blockA_per128={str(m):dict(zip(('p0','cm','dynamics_off','cold_q'),counts[:2,m].sum(0).tolist())) for m in range(3)},per_motion_blockB_per128={str(m):dict(zip(('p0','cm','dynamics_off','equal_budget_q'),counts[2:,m].sum(0).tolist())) for m in range(3)},env_control_ticks_cm_off=310272,env_control_ticks_extra_label_q=310272,new_collection_episodes={'short_physics':1536,'common_full':768,'extra_full':768,'evaluation':3072},actual_joint_critic_optimizer_steps=6000,physical_auxiliary_updates_subset=3000,actual_actor_optimizer_steps=4000,all_actual_actors_and_fullmesh105_audited=True,one_optimization_seed=True,formal_validation=False,distinctive_method_unproved=True)
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
