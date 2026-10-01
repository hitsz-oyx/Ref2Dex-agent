"""Actual randomized physical-retention counts, not learned-policy results."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result_path=BASE/'P-20261002-selective-finger-feasibility-r1/results.json'
    record=json.loads(result_path.read_text())
    out=BASE/'P-20261002-selective-finger-figure-r1'
    if out.exists():raise FileExistsError(out)
    assert record['run_status']=='COMPLETED' and record['label']=='UNPROMISING'
    labels=['None','Shared\ncurl','Index','Middle','Pinky','Ring','Thumb\ncurl','Thumb\nyaw']
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),sharey=True)
    counts=[]
    for mo,ax in enumerate(axes):
        values=[record['motions'][str(mo)][str(a)]['physical105_count'] for a in range(8)]
        assert all(record['motions'][str(mo)][str(a)]['n']==64 for a in range(8))
        counts.append(values)
        ax.bar(range(8),values,color=['#5b6573','#a2633b']+['#397b89']*6,width=.72)
        for x,y in enumerate(values):ax.text(x,y+1,str(y),ha='center',fontsize=9)
        ax.set_xticks(range(8));ax.set_xticklabels(labels,fontsize=8)
        ax.set_ylim(0,67);ax.set_title('Motion '+str(mo)+(' (preselected primary)' if mo==2 else ''))
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Physical105 successes /64 assigned trajectories')
    fig.suptitle('Independent positive finger-target adjustments: fixed feasibility gate fails',fontsize=12)
    fig.tight_layout();out.mkdir()
    for suffix in ('pdf','png'):fig.savefig(out/('selective_fingers.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    report=dict(run_status='COMPLETED',counts=counts,all1536_trajectories=True,no_learned_Cm_policy=True,
        input_sha256={str(result_path):sha(result_path),str(Path(__file__).resolve()):sha(Path(__file__))},
        output_sha256={p.name:sha(p) for p in out.iterdir()})
    (out/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
