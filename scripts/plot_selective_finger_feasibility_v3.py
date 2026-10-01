"""Readable at paper width: all randomized success counts, without utility claims."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=BASE/'P-20261002-selective-finger-feasibility-r1/results.json';r=json.loads(source.read_text())
    out=BASE/'P-20261002-selective-finger-figure-r3'
    if out.exists() or r['run_status']!='COMPLETED' or r['label']!='UNPROMISING':raise ValueError('new completed-data figure only')
    counts=np.array([[r['motions'][str(m)][str(a)]['physical105_count'] for a in range(8)] for m in range(3)])
    assert all(r['motions'][str(m)][str(a)]['n']==64 for m in range(3) for a in range(8))
    fig,ax=plt.subplots(figsize=(6.5,2.65))
    im=ax.imshow(counts,cmap='Blues',vmin=0,vmax=64,aspect='auto')
    ax.set_xticks(range(8));ax.set_xticklabels(['None','Shared\ncurl','Index','Middle','Pinky','Ring','Thumb\ncurl','Thumb\nyaw'],fontsize=8)
    ax.set_yticks(range(3));ax.set_yticklabels(['Motion 0','Motion 1','Motion 2\n(primary)'],fontsize=8)
    ax.tick_params(length=0,pad=7)
    for m in range(3):
        for a in range(8):ax.text(a,m,str(counts[m,a]),ha='center',va='center',fontsize=10,color='white' if counts[m,a]>40 else '#172332')
    ax.add_patch(Rectangle((-.5,1.5),8,1,fill=False,edgecolor='#a05d22',linewidth=1.6))
    ax.set_title('Physical105 successes / 64 randomized trials per cell',fontsize=10,pad=10)
    cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);cb.set_ticks([0,32,64]);cb.ax.tick_params(labelsize=8)
    fig.tight_layout();out.mkdir()
    for suffix in ('pdf','png'):fig.savefig(out/('selective_fingers_heatmap.'+suffix),dpi=200,bbox_inches='tight')
    plt.close(fig)
    report=dict(run_status='COMPLETED',counts=counts.tolist(),all1536_trajectories=True,no_learned_Cm_policy=True,input_sha256={str(source):sha(source),str(Path(__file__)):sha(Path(__file__))},output_sha256={p.name:sha(p) for p in out.iterdir()})
    (out/'figure_manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
