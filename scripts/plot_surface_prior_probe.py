"""Post-closeout scientific figure; no new prediction or model selection."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    result=json.loads((args.run/'results.json').read_text());out=args.run/'figures';out.mkdir()
    reports=result['reports'];sizes=[512,2048,7168]
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    records=[]
    for ax,hand,title,color in zip(axes[:2],['mano','inspire'],['MANO held trajectories','Inspire held trajectories'],['#1768AC','#C75B12']):
        values=[reports[hand+'_'+str(n)][hand+'_eval']['parent_epe_mm'] for n in sizes]
        ax.plot(sizes,values,'o-',color=color);ax.set_xscale('log',base=2);ax.set_xticks(sizes,[str(n) for n in sizes])
        ax.set_xlabel('Training bank windows');ax.set_ylabel('Parent mean EPE (mm)');ax.set_title(title);ax.grid(alpha=.2)
        for n,value in zip(sizes,values):
            ax.annotate(f'{value:.3f}',(n,value),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
            records.append([hand,n,'same_hand',value])
        ax.margins(y=.25)
    names=['mano_7168','adapt_mano_7168','adapt_scratch','adapt_mano_shuffled','adapt_mano_motion_off']
    labels=['MANO\nzero-shot','MANO\nadapted','Scratch\nadapted','Shuffled\nadapted','Motion-off\nadapted']
    values=[reports[name]['inspire_eval']['parent_epe_mm'] for name in names]
    axes[2].bar(range(len(values)),values,color=['#1768AC','#1768AC','#777777','#8764AB','#BC9251'])
    axes[2].set_xticks(range(len(values)),labels,fontsize=8);axes[2].set_ylabel('Parent mean EPE (mm)');axes[2].set_title('Inspire: fixed 256-window adaptation')
    persistence=result['baselines']['inspire_eval']['persistence']['parent_epe_mm']
    axes[2].axhline(persistence,color='black',ls='--',lw=1,label='Previous object flow');axes[2].legend(fontsize=8,loc='upper left');axes[2].set_ylim(0,7.2)
    for n,value in enumerate(values):axes[2].text(n,value+.08,f'{value:.3f}',ha='center',fontsize=8)
    for name,value in zip(names,values):records.append([name,7168 if name!='adapt_scratch' else 0,'inspire_eval',value])
    fig.suptitle('Motion-conditioned Cm: one-initialization offline diagnostic',fontsize=12)
    fig.savefig(out/'scale-cross-hand.png',dpi=180);fig.savefig(out/'scale-cross-hand.pdf');plt.close(fig)
    with (out/'metrics.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','pretrain_bank_size','evaluation','parent_epe_mm']);writer.writerows(records)


if __name__=='__main__':main()
