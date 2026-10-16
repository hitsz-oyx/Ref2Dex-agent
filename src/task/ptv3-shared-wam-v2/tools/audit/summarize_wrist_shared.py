"""Export fixed-endpoint matched wrist/shared outcomes without selecting DEV winners."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    rows=[];matching={}
    for seed in [196,201,203]:
        names={arm:arm if seed==196 else 'repeat'+str(seed)+'-'+arm for arm in ['none','aligned','shuffled']}
        initial={};draws={}
        for arm,name in names.items():
            with np.load(a.source/name/'initial-model.npz') as z:initial[arm]={k:z[k].copy() for k in z.files}
            draws[arm]=np.load(a.source/name/'first-draw.npy')
        for arm in ['aligned','shuffled']:
            assert initial['none'].keys()==initial[arm].keys()
            assert all(np.array_equal(v,initial[arm][k]) for k,v in initial['none'].items())
            assert np.array_equal(draws['none'],draws[arm])
        matching[str(seed)]=dict(initial_parameters_identical=True,first_draw_identical=True)
    runs=[('primary',s,arm,arm if s==196 else 'repeat'+str(s)+'-'+arm)
          for s in [196,201,203] for arm in ['none','aligned','shuffled']]
    runs += [('optimizer',196,n,n) for n in ['low-lr','high-wd']]
    runs += [('frozen-residual',196,arm,'residual-'+arm) for arm in ['none','aligned','shuffled']]
    for kind,seed,arm,name in runs:
        result=json.loads((a.source/name/'result.json').read_text());assert result['status']=='COMPLETED' and result['step']==6000
        m=result['panels'];dev=m['DEV_fixed256']
        if kind=='frozen-residual':assert result['baseline_parameters_unchanged'] and result['zero_condition_exactly_restores_baseline']
        rows.append(dict(kind=kind,seed=seed,condition=arm,run=name,train_mean_mm=m['TRAIN_balanced256']['all']['xyz_mean_mm'],
            dev_mean_mm=dev['all']['xyz_mean_mm'],dev_p95_mm=dev['all']['xyz_p95_mm'],dev_startup_mean_mm=dev['startup']['xyz_mean_mm'],
            all_train_mean_mm=m['train_all']['all']['xyz_mean_mm'],all_dev_mean_mm=m['val_all']['all']['xyz_mean_mm']))
    aggregate={}
    for arm in ['none','aligned','shuffled']:
        group=[r for r in rows if r['kind']=='primary' and r['condition']==arm]
        aggregate[arm]={key:dict(mean=float(np.mean([r[key] for r in group])),std=float(np.std([r[key] for r in group],ddof=1)))
                        for key in ['train_mean_mm','dev_mean_mm','dev_p95_mm','dev_startup_mean_mm']}
    paired=[dict(seed=s,aligned_minus_none_mm=next(r['dev_mean_mm'] for r in rows if r['kind']=='primary' and r['seed']==s and r['condition']=='aligned')-
                 next(r['dev_mean_mm'] for r in rows if r['kind']=='primary' and r['seed']==s and r['condition']=='none')) for s in [196,201,203]]
    summary=dict(status='COMPLETED',experiment_status='UNPROMISING',rows=rows,aggregate=aggregate,paired_differences=paired,matching=matching,
        scope='Exploratory fixed6000 matched conditions, three paired fits; not formal Validation or whole-WAM refutation',
        native_started=False,decision='Retain811D direct control; do not expand this pooled frozen shared condition into native. Diagnose original actor architecture/targets separately.')
    (a.output/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (a.output/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    colors={'none':'tab:blue','aligned':'tab:orange','shuffled':'tab:green'}
    for row in rows:
        if row['kind']!='primary':continue
        x=['none','aligned','shuffled'].index(row['condition'])
        axes[0].scatter(x,row['train_mean_mm'],color=colors[row['condition']],alpha=.8)
        axes[1].scatter(x,row['dev_mean_mm'],color=colors[row['condition']],alpha=.8)
    for arm in ['none','aligned','shuffled']:
        curve=[json.loads(line) for line in (a.source/arm/'metrics.jsonl').read_text().splitlines()]
        axes[2].plot([x['step'] for x in curve],[x['panels']['DEV_fixed256']['all']['xyz_mean_mm'] for x in curve],label=arm,color=colors[arm])
    for ax,title in zip(axes[:2],['Stored TRAIN256, three paired fits','Fixed DEV256, three paired fits']):
        ax.set_xticks(range(3));ax.set_xticklabels(['none','aligned','shuffled']);ax.set_title(title);ax.set_ylabel('First-control wrist XYZ error (mm)');ax.grid(alpha=.2)
    axes[2].set(title='Seed196 DEV curve',xlabel='Updates',ylabel='Mean wrist XYZ error (mm)',ylim=(25,75));axes[2].grid(alpha=.2);axes[2].legend()
    fig.savefig(a.output/'comparison.png',dpi=150)
    print(json.dumps(dict(aggregate=aggregate,paired_differences=paired),indent=2))


if __name__=='__main__':main()
