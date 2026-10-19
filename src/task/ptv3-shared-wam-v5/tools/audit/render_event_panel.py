"""Render exported event timelines and joint/object geometry without a model."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {0:'OakInk2', 1:'GRAB', 2:'ARCTIC', 4:'Shadow', 5:'TACO',
         6:'HOT3D', 8:'DexYCB', 9:'H2O', 10:'HOI4D'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--input', type=Path, required=True)
    a=p.parse_args(); panel=json.loads((a.input/'panel.json').read_text())
    fig, axes=plt.subplots(9, 1, figsize=(12, 16), constrained_layout=True)
    for ax, row in zip(axes, panel['timelines']):
        n=len(row['state']); time=np.arange(n)/30; stride=row['observation_stride']
        ax.plot(time[::stride], np.minimum(row['leave_count'], 20)[::stride], color='.7', lw=.8, label='queries c>=0.5 (cap20)')
        ax.plot(time[::stride], np.minimum(row['enter_count'], 20)[::stride], color='#34495e', lw=.8, label='queries c>=0.8 (cap20)')
        ax.axhline(3, color='.4', ls=':', lw=.8)
        state=np.asarray(row['state']); known=np.asarray(row['event_valid'])
        ax.fill_between(time, 0, 20, where=state==1, color='#27ae60', alpha=.12, step='post', label='near state')
        ax.fill_between(time, -2, 0, where=~known, color='#c0392b', alpha=.35, step='post', label='unknown event condition')
        events=np.asarray(row['transitions']).reshape(-1, 3)
        for kind,color,marker in [(1,'#00843d','^'), (2,'#d35400','v')]:
            t=events[events[:,2]==kind,0]/30
            ax.scatter(t, np.full(len(t), 21), color=color, marker=marker, s=25, label='CONTACT' if kind==1 else 'RELEASE')
        ax.set_title('{} / {} / object slot{}'.format(NAMES[row['source_id']], row['key'], row['slot']), fontsize=10, loc='left')
        ax.set_ylim(-2,23); ax.set_ylabel('queries'); ax.set_xlabel('seconds (30Hz WAM)')
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8)
    fig.savefig(a.input/'event-timelines.png',dpi=120); plt.close(fig)
    with np.load(a.input/'geometry.npz') as data:
        fig=plt.figure(figsize=(12,12))
        for r,sid in enumerate((0,4,10)):
            obj=data['s{}_object'.format(sid)]; hands=data['s{}_hand'.format(sid)]
            valid=data['s{}_valid'.format(sid)]; frames=data['s{}_frames'.format(sid)]; event=data['s{}_event'.format(sid)]
            surface=data['s{}_surface'.format(sid)]; sv=data['s{}_surface_valid'.format(sid)]
            contact=data['s{}_contact'.format(sid)]
            allpoints=np.concatenate((obj, hands[valid], surface[sv].reshape(-1,3)))
            low,high=allpoints.min(0),allpoints.max(0);center=(low+high)/2
            radius=max(.06,float((high-low).max()/2)*1.05)
            for c in range(3):
                ax=fig.add_subplot(3,3,r*3+c+1,projection='3d')
                ax.scatter(*obj[::4].T,s=2,color='.6',alpha=.4)
                near=contact[c]>=.8
                ax.scatter(*obj[near].T,s=4,color='#00843d',alpha=.8)
                for hand in range(2):
                    points=hands[c,hand]; mask=valid[c,hand]
                    if sv[c,hand]: ax.scatter(*surface[c,hand,::4].T,s=1,color=['#2166ac','#b2182b'][hand],alpha=.2)
                    ax.scatter(*points[mask].T,s=9,color=['#2166ac','#b2182b'][hand])
                    for finger in range(5):
                        ids=[0]+list(range(1+4*finger,5+4*finger))
                        if mask[ids].all(): ax.plot(*points[ids].T,lw=1,color=['#2166ac','#b2182b'][hand])
                for setter,mid in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):setter(mid-radius,mid+radius)
                ax.set_box_aspect((1,1,1)); ax.view_init(25,40)
                ax.set_title('{} {} {:+.2f}s'.format(NAMES[sid], 'CONTACT' if event[2]==1 else 'RELEASE', (frames[c]-event[0])/30),fontsize=10)
        fig.suptitle('Actual object-frame joints around annotated onset; not force contact')
        fig.tight_layout(); fig.savefig(a.input/'event-geometry.png',dpi=120);plt.close(fig)


if __name__=='__main__':main()
