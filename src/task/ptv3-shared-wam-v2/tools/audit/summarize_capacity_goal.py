"""Report paired query-head factors at fixed endpoints, preserving Probe limits."""
import argparse
import csv
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    out=a.run/'summary';out.mkdir(exist_ok=False)
    rows=[];curves={}
    for name in ('w128_g0','w128_g1','w512_g0','w512_g1','w512_g1_bypass'):
        d=json.loads((a.run/name/'result.json').read_text());meta=json.loads((a.run/name/'manifest.json').read_text())
        assert d['status']==meta['status']=='COMPLETED' and d['step']==6000
        assert json.loads((a.run/name/'cache-replay.json').read_text())['status']=='PASS'
        val=d['panels']['DEV_fixed256'];train=d['panels']['TRAIN_balanced256']
        rows.append(dict(arm=name,params=meta['params'],train_xyz_mm=train['xyz_mean_mm'],
            dev_xyz_mm=val['xyz_mean_mm'],dev_p95_mm=val['xyz_p95_mm'],
            dev_startup_mm=val['startup']['xyz_mean_mm'],dev_action_l1=val['action_l1'],
            elapsed_s=d['elapsed_s']))
        curves[name]=[json.loads(line) for line in (a.run/name/'curve.jsonl').read_text().splitlines()]
    lookup={r['arm']:r for r in rows};effects={}
    for name,base,candidate in [('width_g0','w128_g0','w512_g0'),('width_g1','w128_g1','w512_g1'),
        ('goal_w128','w128_g0','w128_g1'),('goal_w512','w512_g0','w512_g1'),
        ('bypass_w512_g1','w512_g1','w512_g1_bypass')]:
        b,c=lookup[base],lookup[candidate]
        delta={k:(c[k]/b[k]-1)*100 for k in ('dev_xyz_mm','dev_p95_mm','dev_startup_mm')}
        effects[name]=dict(base=base,candidate=candidate,percent_change=delta,
            promising=delta['dev_xyz_mm']<=-10 and delta['dev_p95_mm']<=-10 and delta['dev_startup_mm']<=0)
    report=dict(rows=rows,effects=effects,matching=json.loads((a.run/'matching-audit.json').read_text()),
        scope='single-seed frozen-WAM Probe; width changes control capacity/optimization; bypass adds linear readout capacity; no WAM efficacy claim')
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    scratch=a.run/'plot-tmp';scratch.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(scratch.resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,curve in curves.items():
        axes[0].plot([x['step'] for x in curve],[x['panels']['DEV_fixed256']['xyz_mean_mm'] for x in curve],label=name)
    axes[0].set(xlabel='Optimizer updates',ylabel='DEV256 wrist XYZ error (mm)',ylim=(25,90))
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    labels=[r['arm'] for r in rows]
    axes[1].bar(labels,[r['dev_xyz_mm'] for r in rows]);axes[1].tick_params(axis='x',rotation=25,labelsize=8)
    axes[1].set(ylabel='DEV256 XYZ error (mm)',title='Fixed6000 endpoint; single-seed Probe')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report))


if __name__=='__main__':main()
