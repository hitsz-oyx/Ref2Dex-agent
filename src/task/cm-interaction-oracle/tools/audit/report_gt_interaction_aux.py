#!/usr/bin/env python3
"""Export bounded PPO GT auxiliary results without changing any model or gate."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();r=json.loads((a.run_dir/'result.json').read_text())
    names=['plain','conditioned','shuffle','stopgrad'];labels=['PPO','GT conditioned','Chunk shuffle','Stop gradient']
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    x=np.arange(4);ev=r['evaluation']
    axes[0].bar(x-.18,[ev[n]['stable45'] for n in names],.36,label='Held ≥1.5s')
    axes[0].bar(x+.18,[ev[n]['sustained_success'] for n in names],.36,label='Held ≥1.5s, no later drop')
    axes[0].set_ylabel('Successful first episodes / 96');axes[0].set_title('Actor-only full episode evaluation');axes[0].legend(fontsize=8)
    axes[1].bar(x,[r['gt_future'][n]['heldout_smooth_l1'] for n in names])
    axes[1].axhline(r['common_source_pool']['initial_decoder_gt_loss'],color='black',ls='--',label='Source + initial decoder')
    axes[1].set_ylabel('Held-out GT SmoothL1');axes[1].set_title('Training-only future prediction');axes[1].legend(fontsize=8)
    axes[2].bar(x,[r['latent_readouts'][n]['mse'][0] for n in names])
    axes[2].set_ylabel('Finite source-policy MC return MSE');axes[2].set_title('Frozen z, fixed ridge readout')
    for ax in axes:
        ax.set_xticks(x,labels,rotation=20,ha='right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('GT interaction auxiliary: '+r['status']+' — single-seed Probe; source-return readout is not current PPO advantage',fontsize=11)
    fig.savefig(a.run_dir/'comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
