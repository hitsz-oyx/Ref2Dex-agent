#!/usr/bin/env python3
"""Pool pre-treatment-frozen independent collection batches, same fixed gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'src/task/cm-interaction-oracle/src')]
from oracle_y_utility import CANDIDATES,oracle_gate,noise_curve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batches',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise ValueError('unique result required')
    reports=[json.loads(p.read_text()) for p in args.batches]
    first=reports[0]
    for report in reports:
        for key in ('candidates','utility_contract','z_contract','simulation_contract'):
            if report[key]!=first[key]: raise ValueError('incompatible panels:'+key)
        if not report['screen']['passed']: raise ValueError('unqualified batch screen')
        for p,h in report['input_sha256'].items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h: raise ValueError('panel drift')
    panels=[np.load(p.with_suffix('.npz')) for p in args.batches]
    y=np.concatenate([p['y'] for p in panels]);z=np.concatenate([p['z'] for p in panels]);motion=np.concatenate([p['motion'] for p in panels])
    result=oracle_gate(y,z,motion)
    selected=np.asarray(result['selection'])
    batch_id=np.concatenate([np.full(len(p['y']),k) for k,p in enumerate(panels)])
    trigger=np.concatenate([p['trigger'] for p in panels])
    result.update(candidates=list(CANDIDATES),utility_contract=first['utility_contract'],z_contract=first['z_contract'],
        simulation_contract=first['simulation_contract'],
        git_commit=first['git_commit'],
        batches=[dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                      npz_sha256=hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest(),
                      anchors=len(panel['y']),status=report['status']) for p,panel,report in zip(args.batches,panels,reports)],
        per_motion=[dict(motion_id=int(m),anchors=int((motion==m).sum()),
            baseline_success=float(z[motion==m,0].mean()),
            selected_success=float(z[motion==m,np.asarray(result['selection'])[motion==m]].mean()),
            candidate_upper_success=float(z[motion==m].max(1).mean())) for m in np.unique(motion)],
        per_fork_group=[dict(batch=int(b),trigger=int(t),anchors=int(((batch_id==b)&(trigger==t)).sum()),
            baseline_success=float(z[(batch_id==b)&(trigger==t),0].mean()),
            selected_success=float(z[(batch_id==b)&(trigger==t),selected[(batch_id==b)&(trigger==t)]].mean()),
            candidate_upper_success=float(z[(batch_id==b)&(trigger==t)].max(1).mean()))
            for b in np.unique(batch_id) for t in np.unique(trigger[batch_id==b])],
        scope='Fixed four synchronous co-treated groups; selected outcome is a potential-outcome mosaic, not an executed mixed-arm policy. Anchor bootstrap does not cover shared solver uncertainty.',
        gate_b=noise_curve(y,z) if result['passed'] else 'NOT_RUN_GATE_A_NOT_POSITIVE',
        gate_c='NOT_RUN_REQUIRES_QUALIFIED_A_B_AND_SEPARATE_PAIRED_TRAIN_TEST_PROTOCOL',
        gate_d='NOT_RUN_REQUIRES_PREVIOUS_GATES')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),y=y,z=z,motion=motion,
        batch=batch_id,trigger=trigger,
        rows=np.concatenate([p['rows'] for p in panels]))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    rates=z.mean(0);axes[0].bar(range(7),100*rates)
    axes[0].set_xticks(range(7),['base','thumb+','thumb−','middle+','middle−','grip+','wrist+'],rotation=30)
    axes[0].set_ylabel('Bounded stable-grasp Z (%)');axes[0].set_title('Executed candidates, identical prefixes')
    values=[result['baseline_success'],result['oracle_y_success'],result['oracle_z_upper_success']]
    axes[1].bar(['Baseline','GT-Y selected','GT-Z upper'],100*np.asarray(values),color=['#7b8a9b','#268475','#cca34d'])
    axes[1].set_ylabel('Z (%)');axes[1].set_title(f"{result['status']}: n={len(y)}")
    for ax in axes: ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Ref13 Oracle-Y Utility Probe • fixed local panel, not policy validation')
    fig.tight_layout();fig.savefig(args.output.with_suffix('.png'),dpi=170);plt.close(fig)
    if result['passed']:
        noise=result['gate_b']['rows']
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        supported=[row for row in noise if row['pairwise_accuracy'] is not None]
        if supported:
            x=np.asarray([row['pairwise_accuracy'] for row in supported])*100
            gain=np.asarray([row['gain']['gain'] for row in supported])*100
            lower=np.asarray([row['gain']['lower95'] for row in supported])*100
            upper=np.asarray([row['gain']['upper95'] for row in supported])*100
            axes[0].errorbar(x,gain,yerr=np.stack((gain-lower,upper-gain)),fmt='o-',capsize=3)
            for a,b,row in zip(x,gain,supported):axes[0].annotate(f"σ={row['sigma']:g}",(a,b),fontsize=8)
        else:
            axes[0].text(.5,.5,'No utility pairs meet the fixed gap threshold',ha='center',transform=axes[0].transAxes)
        axes[0].axhline(0,color='gray',linewidth=1)
        axes[0].set_xlabel('Pairwise utility ordering accuracy (%)')
        axes[0].set_ylabel('Selected Z gain over baseline (pp)')
        axes[1].plot([row['raw_y_rmse'] for row in noise],[row['top1_regret'] for row in noise],'o-')
        axes[1].set_xlabel('Synthetic raw-Y RMSE')
        axes[1].set_ylabel('Top-1 GT utility regret')
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle('Gate B • synthetic sensitivity on the fixed panel')
        fig.tight_layout();fig.savefig(args.output.with_name(args.output.stem+'-noise.png'),dpi=170);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
