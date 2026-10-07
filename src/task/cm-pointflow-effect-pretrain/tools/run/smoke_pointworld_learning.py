#!/usr/bin/env python3
"""Bounded repeated-batch optimizer check; not generalization or scientific utility."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from oakink_wm.data import Windows
from oakink_wm.pointworld import PointWorldWM,capped_collate
from oakink_wm.model import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--stats',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=80)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(9)
    if a.output.exists():raise FileExistsError('preserve previous learning check')
    d=Windows(a.data,'train')
    samples=[]
    for idx in np.random.default_rng(9).choice(d.groups[0],128,replace=False):
        s=d[int(idx)];anchor=np.flatnonzero(s['object_features'][:,13]>.5)[0]
        E=s['effect'][anchor,-1];points=s['points'][anchor]
        epe=np.linalg.norm(points@E[:3,:3].T+E[:3,3]-points,axis=-1).mean()
        if epe>.02:samples.append(s)
        if len(samples)==2:break
    if len(samples)!=2:raise ValueError('no two moving anchors for learning check')
    b={k:v.cuda() for k,v in capped_collate(samples).items()}
    m=PointWorldWM(json.loads(a.stats.read_text())).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.01)
    def measure():
        m.eval()
        with torch.no_grad():
            pred=m(b);loss,_=m.loss(pred,b)
            anchor=b['object_valid']&(b['object_features'][...,13]>.5)
            epe=metrics(pred,b)['point_epe'][...,-1][anchor].mean()
        return dict(loss=float(loss),anchor_h24_epe=float(epe))
    before=measure();start=time.monotonic()
    for step in range(a.steps):
        m.train();opt.zero_grad(set_to_none=True)
        loss,_=m.loss(m(b),b)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite smoke loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        if step%20==0:print(json.dumps(dict(step=step+1,loss=float(loss))),flush=True)
    after=measure()
    result=dict(engineering_only=True,steps=a.steps,parameters=sum(p.numel() for p in m.parameters()),
                sample_ids=b['sample_id'].cpu().tolist(),before=before,after=after,
                seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if after['loss']>=before['loss']*.8:raise AssertionError('normalized repeated-batch loss did not decrease by20%')


if __name__=='__main__':main()
