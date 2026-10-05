#!/usr/bin/env python3
"""Descriptive direction facets; no gates, selection, or neural fitting."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
from intervention import ARM_NAMES, surface_force_projection
from probe_duration_response import residual_fit, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--analysis-dir',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(2)
    p=torch.load(args.dataset,weights_only=False)
    d=torch.load(args.analysis_dir/'diagnostic.pt',weights_only=False)
    levels=p['amplitude_levels']; n_cells=6*len(levels)
    tr=p['trajectory']; force=tr[:,:,48:63].reshape(-1,32,5,3)
    _,tangent=surface_force_projection(force,p['surface_normals'])
    # Fixed source config contactBodies ends in thumb_distal, body4.
    y=torch.stack((tr[:,:,70]*1000,torch.log1p(force[:,:,4].norm(dim=-1)),
        torch.log1p(tangent[:,:,4]),(force.norm(dim=-1)>.1).float().sum(-1),
        tr[:,:,71],(tr[:,:,2]-p['rest_height'][:,None]<.02).float()),-1).numpy().astype(float)
    beta=residual_fit(d['design'],d['labels'],y.reshape(len(y),-1),n_cells)[0].reshape(n_cells,32,6)
    names=('thumb surface distance (mm)','log1p thumb net-force norm (N)',
        'log1p thumb tangent proxy (N)','hand bodies with net force >0.1N',
        'global contact proxy','physical height loss (<2cm)')
    fig,axs=plt.subplots(6,6,figsize=(22,16),sharex=True)
    for arm in range(1,7):
        for axis in range(6):
            ax=axs[arm-1,axis]
            for j,alpha in enumerate(levels):
                ax.plot(np.arange(1,33),beta[j*6+arm-1,:,axis],label=f'alpha{alpha:g}')
            ax.axvline(8,color='black',linestyle='--',linewidth=.7)
            ax.axhline(0,color='black',linewidth=.5)
            if arm==1: ax.set_title(names[axis],fontsize=9)
            if axis==0: ax.set_ylabel(ARM_NAMES[arm],fontsize=9)
            if arm==6: ax.set_xlabel('post-step')
    axs[0,0].legend(fontsize=8)
    fig.suptitle('Descriptive adjusted arm minus pooled zero: local thumb response and global outcomes\nFixed K=8; force/proximity proxies do not certify paired contact or slip',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.965))
    output=args.analysis_dir/'direction_facets.png'; assert not output.exists()
    fig.savefig(output,dpi=150); plt.close(fig)
    manifest=dict(dataset_sha256=sha(args.dataset),diagnostic_sha256=sha(args.analysis_dir/'diagnostic.pt'),
        plot_script_sha256=sha(__file__),fit_helper_sha256=sha(Path(__file__).with_name('probe_duration_response.py')),
        note='Descriptive reviewed body4 thumb response; no additional tests or gate changes',metric_names=list(names))
    (args.analysis_dir/'direction_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__': main()
