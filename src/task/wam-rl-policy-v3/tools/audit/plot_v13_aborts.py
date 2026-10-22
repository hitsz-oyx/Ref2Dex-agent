"""Export the 17-case gate decomposition from the final audit JSON."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output-prefix',type=Path,required=True);a=p.parse_args();data=json.loads(a.audit.read_text());rows=data['records'];fig,(ax,bar)=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[1.5,1]})
    colors={'in_footprint_threshold_crossing':'#1f77b4','fully_below_slab_footprint_reentry':'#d55e00','below_plane_footprint_reentry':'#ccaa33'};labels={'in_footprint_threshold_crossing':'In-footprint threshold crossing','fully_below_slab_footprint_reentry':'Fully below slab, projection reentry','below_plane_footprint_reentry':'Below plane, edge reentry'}
    for y,r in enumerate(rows):
        first=r['first_below_plane_tick']/30;abort=r['abort_state_tick']/30;color=colors[r['table_gate_event']]
        ax.plot([first,abort],[y,y],color=color,lw=2);ax.scatter(first,y,s=28,facecolors='white',edgecolors=color,zorder=3);ax.scatter(abort,y,s=40,color=color,marker='s' if r['phase_last_command']=='close' else 'o',zorder=4)
    ax.set_yticks(range(len(rows)),[r['id']+' '+r['object'] for r in rows],fontsize=8);ax.invert_yaxis();ax.set_xlabel('Physical time (s)');ax.set_title('First below -20 mm (open) to safety abort (filled)',fontsize=10);ax.grid(axis='x',alpha=.2)
    objects=list(data['by_object']);bottom=np.zeros(len(objects))
    for key,color in colors.items():
        values=np.array([sum(r['object']==o and r['table_gate_event']==key for r in rows) for o in objects]);bar.barh(objects,values,left=bottom,color=color,label=labels[key]);bottom+=values
    bar.set_xlim(0,5.2);bar.set_xticks(range(6));bar.invert_yaxis();bar.set_xlabel('Aborts / 5 repeats per object');bar.set_title('17/40 unseen A1 aborts; same frozen scoring',fontsize=10);bar.grid(axis='x',alpha=.2);bar.legend(loc='lower left',bbox_to_anchor=(-.05,-.30),fontsize=8,frameon=False)
    fig.suptitle('Table safety gate is not an exact mesh-collision test',fontsize=12);fig.tight_layout(rect=[0,.03,1,.96]);a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(a.output_prefix.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
