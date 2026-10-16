"""Recompute partial-hold metrics and expose endpoint censoring in the fixed native panel."""
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
    rows=[]
    for index,obj in enumerate(['airplane','sphere','mug']):
        folder=a.source/obj/'native';m=json.loads((folder/'manifest.json').read_text())
        result=json.loads((folder/'result.json').read_text());audit=json.loads((folder/'independent_audit.json').read_text())
        assert m['status']==result['status']=='COMPLETED' and audit['pass'] and m['execution_steps']==1
        with np.load(folder/'trajectory.npz') as z:
            assert all(np.array_equal(z[k][0,0],z[k][0,i]) for k in ['q','dq','hand','obj','velocity'] for i in [1,2])
            for env,label in enumerate(['teacher','none','aligned']):
                h=z['obj'][:,env,2,3]-z['obj'][0,env,2,3];gap=z['surface_gap'][:,env]
                held=(gap<=.01)&(h>=.03)&~(z['table_footprint'][:,env]&(abs(z['support_gap'][:,env])<=.02));held[0]=False
                diff=np.diff(np.r_[False,held,False].astype(int));starts=np.flatnonzero(diff==1);ends=np.flatnonzero(diff==-1)
                spans=sorted([(int(stop-start),int(start),int(stop-1)) for start,stop in zip(starts,ends)],reverse=True)
                maximum=spans[0][0] if spans else 0
                assert maximum==result['outcomes'][env]['maximum_held_frames']
                clipping=abs(z['intended'][:,env]-z['applied'][:,env])>1e-6
                rows.append(dict(object=obj,condition=label,role=m['roles'][env],maximum_held_frames=maximum,
                    observed_hold45=maximum>=45,held_at_endpoint=bool(held[-1]),
                    terminal_hold_frames=int(len(held)-starts[-1]) if held[-1] else 0,
                    terminal_height_m=float(h[-1]),maximum_height_m=float(h.max()),terminal_surface_gap_m=float(gap[-1]),
                    clipping_fraction=float(clipping.any(-1).mean()),wrist_clipping_fraction=float(clipping[:,:6].any(-1).mean()),
                    finger_clipping_fraction=float(clipping[:,6:].any(-1).mean()),longest_spans=spans[:3],native=str(folder)))
                t=np.arange(len(h))/30
                axes[index,0].plot(t,h,label=label);axes[index,1].plot(t,gap,label=label)
            axes[index,0].axhline(.03,color='gray',ls=':',label='Lift3cm');axes[index,0].set(title=obj,ylabel='Object height over initial (m)',xlabel='Time (s)')
            axes[index,1].axhline(.01,color='gray',ls=':',label='Surface1cm');axes[index,1].set(yscale='symlog',title=obj,ylabel='Hand-object surface gap (m)',xlabel='Time (s)');axes[index,1].set_yscale('symlog',linthresh=.01)
            for ax in axes[index]:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(a.output/'native-traces.png',dpi=140)
    report=dict(status='COMPLETED',experiment_status='UNCLEAR',metric='partial_hold45_only_no_placement',
        scope='Three fixed val motions, one reset and wrist substitution; no population success-rate or placement claim',rows=rows,
        notes=['none airplane has40-frame and endpoint-censored37-frame holds; aligned60-frame hold ends before endpoint',
               'Shared helps longest airplane hold but reduces sphere/mug holds; two observed45-frame motions per condition is not a success-rate estimate',
               'Same shared finger controller, own states; finger commands can differ after wrist-induced state divergence'])
    (a.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (a.output/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:print(row['object'],row['condition'],row['maximum_held_frames'],'terminal_hold',row['terminal_hold_frames'],'fingerclip',round(row['finger_clipping_fraction'],3))


if __name__=='__main__':main()
