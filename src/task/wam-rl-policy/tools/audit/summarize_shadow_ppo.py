"""Summarize final paired native behavior without treating training return as success."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=a.run/'summary';out.mkdir(exist_ok=False)
    curve=json.loads((a.run/'train/curve.json').read_text())
    pairs=json.loads((a.run/'evaluation-pairs.json').read_text())
    rows=[]
    for pair in pairs:
        assert max(pair['initial_state_max_error'].values())==0
        for role in ('bc','ppo'):
            report=pair['roles'][role];native=report['native']
            rows.append(dict(seed=pair['seed'],role=role,held=native['held60_with_lift10cm'],
                official=native['official_success'],final=native['final_within5cm'],
                train_held=native['by_split']['train']['held60_with_lift10cm'],
                dev_held=native['by_split']['dev']['held60_with_lift10cm'],
                mean_return=report['mean_episode_return'],residual_rms=report['applied_residual_rms'],
                clipping=report['clip_fraction']))
    totals={role:{key:sum(r[key] for r in rows if r['role']==role)
                  for key in ('held','official','final','train_held','dev_held')} for role in ('bc','ppo')}
    audit=json.loads((a.run/'train/freeze-audit.json').read_text());assert audit['status']=='PASS'
    report=dict(status='UNCLEAR',rows=rows,totals=totals,paired_held_deltas=[p['held_delta'] for p in pairs],
        updates=len(curve),env_steps=curve[-1]['env_steps'],
        accepted_actor_steps=sum(x['ppo']['actor_steps'] for x in curve),
        max_joint_kl=max(x['ppo']['joint_kl'] for x in curve),
        max_behavior_logprob_error=max(x['ppo']['behavior_logprob_error'] for x in curve),
        max_execution_error={k:max(x['native_execution_error'][k] for x in curve) for k in curve[0]['native_execution_error']},
        frozen_audit=audit,bc_repeatability=json.loads((a.run/'reference-repeatability.json').read_text()),
        scope='one training seed/two TRAIN objects; three exploratory reset panels; position hold proxy, no WAM token intervention or formal RL efficacy claim')
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scratch=out/'plot-tmp';scratch.mkdir();os.environ['MPLCONFIGDIR']=str(scratch.resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.7));steps=[x['update'] for x in curve]
    axes[0].plot(steps,[x['mean_episode_return'] for x in curve]);axes[0].set(xlabel='PPO update',ylabel='Exploratory TRAIN episode return')
    axes[1].plot(steps,[x['ppo']['joint_kl'] for x in curve]);axes[1].axhline(.02,ls='--',color='gray')
    axes[1].set(xlabel='PPO update',ylabel='Joint latent KL')
    seeds=sorted({r['seed'] for r in rows})
    for role,offset in [('bc',-.18),('ppo',.18)]:
        axes[2].bar([i+offset for i in range(3)],[next(r['held'] for r in rows if r['role']==role and r['seed']==s) for s in seeds],width=.35,label=role)
    axes[2].set_xticks(range(3));axes[2].set_xticklabels(seeds);axes[2].set(xlabel='Paired reset seed',ylabel='Held60 + lift10cm /16',ylim=(0,16));axes[2].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report))


if __name__=='__main__':main()
