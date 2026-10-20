"""Measured Inspire outcome report; never infer grasp success from height alone."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def measured_outcome(path):
    with np.load(path) as z:data={k:z[k] for k in z.files}
    pose=data['obj'];height=pose[:,2,3]-pose[0,2,3]
    supported=data['table_footprint'].astype(bool)&(np.abs(data['support_gap'])<=.02)
    held=(data['surface_gap']<=.01)&~supported&(height>=.03);held[0]=False
    streak=[];run=0
    for value in held:
        run=run+1 if value else 0;streak.append(run)
    stable=np.asarray(streak)>=45
    error=np.linalg.norm(pose[:,:3,3]-data['reference_obj'][:,:3,3],axis=-1)
    return dict(controls=len(data['action']),maximum_lift_m=float(height.max()),
        terminal_lift_m=float(height[-1]),maximum_held_frames=max(streak),
        ever_held45=bool(stable.any()),final_held45=bool(stable[-1]),
        drop_after_hold=bool(any(not h and stable[:i].any() for i,h in enumerate(held))),
        reference_position_rmse_m=float(np.sqrt(np.mean(error**2))),
        canonical_action_capture_exact=bool(np.array_equal(data['action'],data['applied'])),
        metric='measured sampled surface gap<=1cm, unsupported, lift>=3cm for45 frames; proximity proxy, not force certificate'),data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);a=p.parse_args();results={};traces={}
    source=np.load(a.source/'trajectory.npz')
    for directory in sorted(a.output.rglob('sequences')):
        for sequence in directory.iterdir():
            path=sequence/'trajectory.npz'
            if not path.exists():continue
            name=str(sequence.parents[1].relative_to(a.output))
            result,data=measured_outcome(path)
            result['source_initial_exact']=({k:bool(np.array_equal(source[k][0],data[k][0])) for k in ('q','hand','obj','native_current_obs')} if sequence.name==a.source.name else None)
            results[name]=result;traces[name]=data
    packet=dict(scope='single TRAIN sequence engineering; BC rollout is not PPO/generalization evidence',runs=results)
    (a.output/'outcomes.json').write_text(json.dumps(packet,indent=2)+'\n')
    root=Path(__file__).resolve().parents[5]
    os.environ['MPLCONFIGDIR']=str(root/'tmp/inspire-plots')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
    for name,data in traces.items():
        if name.startswith('teacher') or 'smoke32' in name:continue
        t=np.arange(len(data['obj']))/30
        axes[0].plot(t,data['obj'][:,2,3]-data['obj'][0,2,3],label=name)
        axes[1].plot(t,data['surface_gap'],label=name)
    axes[0].set(ylabel='Object lift (m)',title='Inspire measured native rollouts (30 Hz)');axes[0].legend(fontsize=7)
    axes[1].axhline(.01,color='black',ls='--',lw=.8);axes[1].set(xlabel='Time (s)',ylabel='Hand-object surface gap (m)');axes[1].set_ylim(0,.1)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output/'measured-outcomes.png',dpi=160)
    print(json.dumps(packet,indent=2))


if __name__=='__main__':main()
