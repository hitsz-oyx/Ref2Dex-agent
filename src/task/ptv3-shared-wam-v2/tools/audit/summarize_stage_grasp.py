"""Join audited native stage outcomes with saved offline losses, without inference."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location('stage_probe', TASK/'tools/run/probe_stage_grasp.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--losses', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    raw = json.loads((a.run/'result.json').read_text())
    loss = {d['step']:d for d in json.loads(a.losses.read_text())}
    rows = []
    for run in raw['runs']:
        folder = Path(run.get('reused_from') or run.get('path') or '')
        if not str(folder) or str(folder)=='.':
            short = next(x[0] for x in module.PANEL if x[1]==run['sequence'])
            folder = a.run/('step%d-%s'%(run['step'],short))
        values = module.summarize(folder.resolve(),run['step'],run['sequence'])
        for row in values:
            row.update(action_loss=loss[row['step']]['action'],
                       forecast_loss=loss[row['step']]['forecast_residual'])
        rows.extend(values)
    if len(rows)!=81:
        raise ValueError('complete 9 x 3 x 3 panel required')
    out.mkdir(parents=True)
    (out/'outcomes.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields=sorted({key for r in rows for key in r})
    with (out/'outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    table=['| step | action loss | forecast loss | GT tau R held frames (air/mug/sphere) | WAM R held frames (air/mug/sphere) |',
           '|---:|---:|---:|---|---|']
    stage=[]
    for step in module.STEPS:
        r=[x for x in rows if x['step']==step]
        groups={role:[next(x for x in r if x['sequence']==seq and x['role']==role) for _,seq,_ in module.PANEL]
                for role in ['teacher','GT_tau_R','shared_WAM_R']}
        gt='/'.join(str(x['maximum_held_frames']) for x in groups['GT_tau_R'])
        own='/'.join(str(x['maximum_held_frames']) for x in groups['shared_WAM_R'])
        table.append('| %d | %.4f | %.4f | %s | %s |'%(step,loss[step]['action'],loss[step]['forecast_residual'],gt,own))
        stage.append(dict(step=step,action_loss=loss[step]['action'],forecast_loss=loss[step]['forecast_residual'],
            **{role:dict(held_frames=[x['maximum_held_frames'] for x in values],
                          held45_count=sum(x['held45'] for x in values),
                          hand_mean_mm=sum(x['hand_recorded_mean_mm'] for x in values)/3,
                          clipping_mean=sum(x['clipping'] for x in values)/3)
               for role,values in groups.items()}))
    (out/'stage-summary.json').write_text(json.dumps(stage,indent=2)+'\n')
    (out/'table.md').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
    steps=module.STEPS
    axes[0].plot(steps,[loss[s]['action'] for s in steps],'o-',label='Action dev')
    ax=axes[0].twinx();ax.plot(steps,[loss[s]['forecast_residual'] for s in steps],'s--',color='tab:orange',label='Forecast dev')
    axes[0].set_ylabel('Action loss',color='tab:blue');ax.set_ylabel('Forecast loss',color='tab:orange')
    axes[0].set_title('Fixed offline dev losses')
    for axis,role,title in zip(axes[1:],['GT_tau_R','shared_WAM_R'],['GT tau + R','Predicted tau + R']):
        for i,(short,_,_) in enumerate(module.PANEL):
            axis.plot(steps,[r[role]['held_frames'][i] for r in stage],'o-',label=short)
        axis.axhline(45,color='gray',ls='--',label='held45 threshold')
        axis.set(title=title,ylabel='Longest held run (frames)');axis.legend(fontsize=7);axis.set_ylim(bottom=-2)
    for axis in axes:axis.set_xlabel('Fine-tune step');axis.grid(alpha=.2)
    fig.suptitle('K4 / 3 dev sequences / 1 seed: holding proxy, not task success')
    fig.savefig(out/'stage-comparison.png',dpi=160);plt.close(fig)
    print('\n'.join(table))


if __name__=='__main__':
    main()
