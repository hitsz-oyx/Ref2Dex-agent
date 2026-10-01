#!/usr/bin/env python3
"""Standalone source-reference figure; never present a target as physical success."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('retain existing figure')
    record=json.loads(args.audit.read_text());fig,axes=plt.subplots(3,1,figsize=(7,5.5),sharex=False,constrained_layout=True)
    inputs={str(args.audit.resolve()):sha(args.audit),str(Path(__file__).resolve()):sha(Path(__file__))}
    for axis,row in zip(axes,record['reference_records']):
        path=Path(row['path'])
        if sha(path)!=row['sha256']:raise ValueError('reference drift')
        raw=torch.load(path,map_location='cpu',weights_only=False);height=(raw[:,200]-raw[0,200]).numpy()*100
        inputs[str(path)]=sha(path);time=np.arange(len(height))/30
        axis.plot(time,height,color='#236c96',linewidth=1.4,label='Reference object height')
        axis.axhline(3,color='#a54f35',linestyle='--',linewidth=1,label='3 cm threshold')
        axis.set_title(f"{path.parent.name}: longest lifted interval {row['max_lift_run_frames']/30:.2f} s",fontsize=10,loc='left')
        axis.set_ylabel('Rise (cm)');axis.set_xlabel('Reference time (s)');axis.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('Source lift cycles versus a 1.5 s holding specification',fontsize=12)
    args.output.mkdir(parents=True);fig.savefig(args.output/'reference_hold.pdf');fig.savefig(args.output/'reference_hold.png',dpi=160)
    plt.close(fig)
    result=dict(run_status='COMPLETED',reference_labels_only=True,no_physical_success_claim=True,input_sha256=inputs,
        output_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'figure_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
