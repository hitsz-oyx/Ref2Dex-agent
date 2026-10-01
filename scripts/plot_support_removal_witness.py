#!/usr/bin/env python3
"""Standalone saved-physics support witness plot; every assigned primary row."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_contact_response_probe import sha

def main():
    torch.set_num_threads(2)
    base=ROOT/'src/task/CmResidual/research/contact_response/output'
    run=base/'P-20261002-support-removal-witness-r1';out=base/'P-20261002-support-removal-figure-r1'
    destinations=[ROOT/'paper/figures/support_removal.pdf',ROOT/'paper/figures/support_removal.png']
    if out.exists() or any(f.exists() for f in destinations):raise ValueError('retain existing figure')
    result=json.loads((run/'results.json').read_text())
    if result['label']!='PROMISING' or result['primary_motion']!=1:raise ValueError('witness provenance')
    fig,axes=plt.subplots(1,2,figsize=(9.0,3.0),sharey=True);inputs={str(run/'results.json'):sha(run/'results.json')};counts={}
    for axis,seed in zip(axes,(508,509)):
        panel=run/f's{seed}';r=json.loads((panel/'results.json').read_text())
        for name in ('initial','trace'):
            f=panel/(name+'.pt')
            if sha(f)!=r[name+'_sha256']:raise ValueError('trace provenance')
            inputs[str(f)]=sha(f)
        a=torch.load(panel/'initial.pt',map_location='cpu',weights_only=False);d=torch.load(panel/'trace.pt',map_location='cpu',weights_only=False)
        t=d['progress'][:,0].numpy()/30;rise=(d['object_root'][:,:,2].numpy()-a['initial_height'].numpy()[None])*1000
        for arm,label,color in ((0,'Hold target','#2166ac'),(1,'Open + retreat','#b2182b')):
            ids=(a['motion'].numpy()==1)&(a['arm_assignment'].numpy()==arm)
            if ids.sum()!=16:raise ValueError('complete randomized arm')
            counts[f'{seed}_{arm}']=int(ids.sum());values=rise[:,ids]
            axis.plot(t,values.mean(1),label=label,color=color,lw=1.7)
            axis.fill_between(t,values.min(1),values.max(1),color=color,alpha=.15)
        stop=int(a['phase_stop'][1]);axis.axvline(stop/30,color='.4',ls='--',lw=1)
        axis.axvspan((stop+30)/30,(stop+40)/30,color='.8',alpha=.35)
        axis.axhline(30,color='.65',ls=':',lw=1);axis.set_title(f'Fresh environment seed {seed}')
        axis.set_xlabel('Physical time (s)');axis.grid(alpha=.15);axis.set_xlim(0,202/30)
    axes[0].set_ylabel('Object-root rise from frame0 (mm)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout();out.mkdir()
    for path in destinations:fig.savefig(path,dpi=180,bbox_inches='tight')
    manifest=dict(run_status='COMPLETED',source='actual saved simulator trajectories',primary_motion=1,
        all_assigned_rows_included=counts,band='trajectory minimum/maximum, not confidence intervals',
        shaded_window='plateau_stop+30 through +40; plotted every actual tick',input_sha256=inputs,
        script_sha256=sha(Path(__file__)),output_sha256={str(f):sha(f) for f in destinations})
    (out/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
