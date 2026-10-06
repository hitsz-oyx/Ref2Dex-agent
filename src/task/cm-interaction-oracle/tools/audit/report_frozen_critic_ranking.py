#!/usr/bin/env python3
"""Standalone paired selector chart from completed immutable critic scores."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    out=args.run_dir/'comparison.png'
    if out.exists():raise ValueError('refusing existing figure')
    result=json.loads((args.run_dir/'result.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Baseline','GT-Y','Critic Q8','V8 only','Reward8 only','Candidate upper']
    arms=result['arms'];n=result['anchors']
    counts=[arms['GT_Y']['baseline_successes'],arms['GT_Y']['successes'],arms['Critic_Q8']['successes'],arms['Value8_only']['successes'],arms['Reward8_only']['successes'],result['candidate_z_upper']]
    fig,ax=plt.subplots(figsize=(9,4.4));bars=ax.bar(names,np.array(counts)/n*100,color=['#8796a5','#287f72','#427bc1','#809fce','#c9a45b','#ba7546'])
    for bar,count in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{count}/{n}',ha='center')
    ax.set_ylim(0,100);ax.set_ylabel('Saved candidate Z success (%)')
    ax.set_title('Frozen source_e260 critic vs GT-Y: same-prefix one-shot panels')
    ax.text(.5,-.17,'Potential-outcome mosaic; no composed rolling critic policy was executed',ha='center',transform=ax.transAxes,fontsize=9)
    fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)


if __name__=='__main__':main()
