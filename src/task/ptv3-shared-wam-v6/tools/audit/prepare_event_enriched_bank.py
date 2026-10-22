"""Larger existing-TRAIN candidate bank, enriched for inside-horizon events.

Candidate windows may overlap; variance groups use one window/native capture,
and a capture only once per gate. Neural evaluation panels remain disjoint.
This avoids spending GPU inference on candidate-bank coverage.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from audit_goal_chain import MANIFEST,load,dump
import audit_identifiability as audit
from ptv3_shared_wam_v6.data import V6Windows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=182);a=p.parse_args()
    root=a.output;out=root/'event-bank';out.mkdir(exist_ok=True)
    dataset=V6Windows(MANIFEST,'train');ix=np.asarray(dataset.index);by=defaultdict(list)
    for i,row in enumerate(ix):by[int(row[0])].append(i)
    rng=np.random.default_rng(a.seed);rows=[];near_count=0
    for sid,ids in sorted(by.items()):
        meta=dataset.meta['sequences'][sid];m=load(Path(meta['path'])/'meta.json');cap=str(meta['source_id'])+':'+m.get('source_sequence',meta['key'])
        ep=Path(dataset.event_rows[meta['key']]['path']);et=np.load(ep/'event_type.npy',mmap_mode='r');ed=np.load(ep/'event_delta_steps.npy',mmap_mode='r')
        near=[i for i in ids if et[ix[i,2],ix[i,1]]>0 and 0<ed[ix[i,2],ix[i,1]]<24]
        picked=set(rng.choice(ids,min(16,len(ids)),replace=False).tolist())
        if near:picked.update(rng.choice(near,min(16,len(near)),replace=False).tolist())
        for i in sorted(picked):
            _,slot,t=map(int,ix[i]);near_count+=int(et[t,slot]>0 and 0<ed[t,slot]<24)
            rows.append(dict(seq_id=sid,slot=slot,tick=t,index=i,split='train',source_id=meta['source_id'],
                capture=cap,object=meta['objects'][slot],phase_group=0))
    dump(out/'panels.json',dict(bank=rows));dump(out/'selection.json',dict(seed=a.seed,candidates=len(rows),sequences=len(by),
        inside_horizon_event_candidates=near_count,selection='up to16 uniform plus16 inside-horizon-event windows per sequence;existing frozen TRAIN only',
        overlap='candidate pool may overlap;one native recording per variance group and one group per gate;neural panels unchanged'))
    a.output=out;a.manifest=MANIFEST;audit.extract_bank(a)
    bank=load(out/'bank.json')
    for row in bank['samples']:
        if row['source_id']!=1:row['interaction_type']='UNAVAILABLE_NON_GRAB'
    dump(out/'bank.json',bank)


if __name__=='__main__':main()
