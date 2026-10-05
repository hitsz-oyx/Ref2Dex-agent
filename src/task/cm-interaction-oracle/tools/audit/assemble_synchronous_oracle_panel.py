#!/usr/bin/env python3
"""Assemble matched same-H candidate rows from separate synchronous groups."""
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT/'src/task/cm-interaction-oracle/src'),str(Path(__file__).parent)]
from audit_oracle_y_panel import prefix_ok,identity
from oracle_y_utility import CANDIDATES


def main():
    root=ROOT/'outputs/cm-interaction-oracle'
    for base in ['oracle-y-utility-s263','oracle-y-utility-extra-s264']:
        p0=torch.load(root/(base+'-sync-reference')/'panel.pt',weights_only=False)
        schedule=json.loads((root/(base+'-sync-schedule.json')).read_text());groups=torch.tensor(schedule['groups'])
        screen=json.loads((root/(base+'-sync-screen.json')).read_text())
        if not screen['passed']: raise ValueError('unqualified baseline')
        n=len(groups)
        for k in range(1,7):
            value={key:(v.clone() if isinstance(v,torch.Tensor) else v) for key,v in p0.items()}
            provenance={};covered=torch.zeros(n,dtype=torch.bool)
            for group in range(2):
                path=root/(base+f'-sync-g{group}-candidate{k}')/'panel.pt'
                raw=torch.load(path,weights_only=False);mask=groups==group
                if raw['candidate']!=k or not torch.equal(raw['triggers'][mask],p0['triggers'][mask]): raise ValueError('group identity')
                if not prefix_ok(raw)[mask].all() or not raw['valid_steps'][mask].all(): raise ValueError('unmatched/incomplete assigned group')
                for key in ('initial_fingerprint','simulation_contract','model_fingerprint','rms_fingerprint'):
                    if raw[key]!=p0[key]: raise ValueError('initial/backend/actor drift')
                for key in ('motion_id','start_frame','rest_height','delta'):
                    if not torch.equal(raw[key],p0[key]): raise ValueError('physical identity drift')
                for key,v in value.items():
                    if isinstance(v,torch.Tensor) and v.ndim and len(v)==n:v[mask]=raw[key][mask]
                provenance[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest();covered|=mask
            if not torch.equal(covered,p0['triggers']>=0): raise ValueError('missing/duplicate assignment')
            value.update(candidate=k,candidate_name=CANDIDATES[k],group_source_sha256=provenance,
                         assembly='Same-H rows from separate synchronous coldprefix forks; not one mixed-arm execution')
            identity(p0,value)
            out=root/(base+f'-sync-merged-candidate{k}');out.mkdir(exist_ok=False)
            torch.save(value,out/'panel.pt')
            (out/'assembly.json').write_text(json.dumps(dict(candidate=k,inputs=provenance),indent=2)+'\n')
    print('Complete seven-arm same-H panels assembled; no outcome filtering.')
if __name__=='__main__':main()
