#!/usr/bin/env python3
"""Load only the fixed, terminal, independently audited HF19 panel."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha


def load(source,audit_path):
    import torch
    manifest_path=source/'run_manifest.json'
    m=json.loads(manifest_path.read_text());audit=json.loads(audit_path.read_text())
    if m['run_status']!='COMPLETED' or m['smoke_only'] or m.get('child_exit_code')!=0:
        raise ValueError('terminal scientific source required')
    if m['family']!='HF19' or [p['seed'] for p in m['phases']]!=list(range(571,583)):
        raise ValueError('fixed twelve scientific phases required')
    if audit['run_status']!='COMPLETED' or audit['run_manifest_sha256']!=sha(manifest_path):
        raise ValueError('independent actual source audit required')
    if [p['seed'] for p in audit['phases']]!=list(range(571,583)):
        raise ValueError('full source audit required')
    if any(sha(Path(k))!=v for k,v in m['input_sha256'].items()):
        raise ValueError('source input drift')
    records=[];hashes=dict(m['input_sha256'])
    hashes.update({str(p.resolve()):sha(p) for p in [manifest_path,audit_path]})
    for phase in m['phases']:
        path=Path(phase['directory'])/'records.pt'
        if phase['run_status']!='COMPLETED' or phase['exit_code']!=0 or sha(path)!=phase['result']['record_sha256']:
            raise ValueError('terminal phase/hash required')
        b=torch.load(path,map_location='cpu',weights_only=False)
        if b['schema']!='ref2dex.contact_geometry_source.v1' or b['future_done'].any() or b['cm_used']:
            raise ValueError('actual complete randomized source required')
        records.append(b);hashes[str(path.resolve())]=sha(path)
    counts={};adequate=True
    for name,lower,upper in [('fit',0,50),('cal',50,70),('held',70,100)]:
        rows=0;arms=torch.zeros(8,dtype=torch.long);episodes=set();groups=set();early=0
        for b in records:
            take=(b['split_group_bucket']>=lower)&(b['split_group_bucket']<upper)
            rows+=int(take.sum());arms+=torch.bincount(b['assignment'][take],minlength=8)
            early+=int((take&~b['outcome']['initially_clear']).sum())
            episodes.update(ep for ep,flag in zip(b['episode_id'],take) if flag)
            groups.update(f'{int(i)}/{int(j)}' for i,j in zip(b['motion_id'][take],b['start_frame'][take]))
        ok=rows>=150 and len(episodes)>=32 and len(groups)>=8 and int(arms[0])>=24 and int(arms[1])>=24 and int(arms[2:].sum())>=80
        counts[name]=dict(rows=rows,episodes=len(episodes),initial_groups=len(groups),arms=arms.tolist(),
                          early=early,initially_clear=rows-early,adequate=ok)
        adequate &= ok
    return records,counts,adequate,hashes


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();begin=time.monotonic()
    if args.output.exists():raise ValueError('unique qualification output required')
    import torch
    torch.set_num_threads(2)
    records,counts,adequate,hashes=load(args.source,args.audit)
    result=dict(run_status='COMPLETED',supervision_adequate=adequate,counts=counts,input_sha256=hashes,
                elapsed_seconds=time.monotonic()-begin,scope='fixed split/actual allocation support; no observed held effect selection')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
