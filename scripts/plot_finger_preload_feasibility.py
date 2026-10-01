#!/usr/bin/env python3
"""Standalone saved-trajectory physical figure; no neural or simulator compute."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'

def main():
    directory=BASE/'P-20261002-finger-preload-feasibility-r1'
    output=BASE/'P-20261002-finger-preload-figure-r1';output.mkdir(exist_ok=False)
    result=json.loads((directory/'results.json').read_text())
    if result['run_status']!='COMPLETED' or result['label']!='UNPROMISING':raise ValueError('evidence drift')
    inputs={str(directory/'results.json'):sha(directory/'results.json'),str(Path(__file__).resolve()):sha(Path(__file__))}
    group={(m,k):[] for m in range(3) for k in range(4)}
    for seed in (504,505):
        d=directory/f's{seed}';meta=json.loads((d/'results.json').read_text())
        for name in ('initial','trace'):
            path=d/(name+'.pt');inputs[str(path)]=sha(path)
            if inputs[str(path)]!=meta[name+'_sha256']:raise ValueError('data drift')
        initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        rise=(trace['object_root'][:,:,2]-initial['initial_height'][None]).numpy()*1000
        for m in range(3):
            for k in range(4):
                mask=(initial['motion']==m)&(initial['dose_assignment']==k);group[m,k].append(rise[:,mask.numpy()])
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(7,7.2),sharex=True,constrained_layout=True)
    colors=['#505050','#56a7d0','#e09835','#9c3d54'];time=np.arange(1,91)/30
    for m,ax in enumerate(axes):
        for k,dose in enumerate((0.,.05,.15,.30)):
            values=np.concatenate(group[m,k],axis=1);mean=values.mean(1)
            ax.plot(time,mean,color=colors[k],label=f'{dose:.2f} rad',lw=1.8)
        ax.axhline(-10,color='black',linestyle='--',lw=1,label='Height condition (-10 mm)')
        counts=[result['arms'][str(k)]['motions'][str(m)]['retained75_count'] for k in range(4)]
        ax.set_title(f'Motion {m}: retained75 counts {counts} / 16 per dose',loc='left',fontsize=10)
        ax.set_ylabel('Root rise from\nelevated start (mm)');ax.grid(alpha=.2)
    axes[0].legend(ncol=3,fontsize=8,loc='lower left');axes[-1].set_xlabel('Physical time (s)')
    for suffix in ('pdf','png'):
        path=ROOT/f'paper/figures/finger_preload.{suffix}'
        if path.exists():raise ValueError('unique retained figure required')
        fig.savefig(path,dpi=180)
    plt.close(fig)
    manifest=dict(run_status='COMPLETED',physical_trajectories=192,means_per_motion_dose_n=16,input_sha256=inputs,
        output_sha256={str(ROOT/f'paper/figures/finger_preload.{s}'):sha(ROOT/f'paper/figures/finger_preload.{s}') for s in ('pdf','png')},
        boundary='physical trajectory means; curve is not success, independent confidence interval, policy or force closure')
    (output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(status='COMPLETED',counts=[result['arms'][str(k)]['pooled']['retained75_count'] for k in range(4)])))
if __name__=='__main__':main()
