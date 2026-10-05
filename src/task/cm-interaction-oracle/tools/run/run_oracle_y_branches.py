#!/usr/bin/env python3
"""Sequential, resource-bounded candidate branches with immediate pairing guard."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src'),str(ROOT/'src/task/cm-interaction-oracle/tools/audit')]
import torch
from audit_oracle_y_panel import identity,prefix_ok


def main():
    outputs=ROOT/'outputs/cm-interaction-oracle'
    cohort=json.loads((outputs/'oracle-y-utility-sync-cohort.json').read_text())
    for path,h in cohort['screen_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h: raise ValueError('frozen cohort drift')
    begin=time.monotonic()
    existing=sum(json.loads(p.read_text()).get('wall_seconds',0) for p in outputs.glob('oracle-y-*/run_manifest.json'))
    # Conservatively charge300s for preflight/log/analysis overhead not in completed manifests.
    budget=2400-existing-300
    if budget<=0: raise TimeoutError('registered aggregate resource budget exhausted')
    wrapper=ROOT/'src/task/cm-interaction-oracle/tools/run/run_oracle_y_candidate.sh'
    for base,n,seed in [('oracle-y-utility-s263',96,263),('oracle-y-utility-extra-s264',48,264)]:
        reference=outputs/(base+'-sync-reference')
        screen=json.loads((outputs/(base+'-sync-screen.json')).read_text())
        schedule=json.loads((outputs/(base+'-sync-schedule.json')).read_text())
        if not screen['passed']: raise ValueError('baseline repeat rejected')
        p0=torch.load(reference/'panel.pt',map_location='cpu',weights_only=False)
        accepted=torch.tensor(screen['indices'],dtype=torch.long)
        groups=torch.tensor(schedule['groups'])
        for group in range(2):
            rows=accepted[groups[accepted]==group]
            for k in range(1,7):
                if time.monotonic()-begin+240>budget: raise TimeoutError('insufficient branch budget')
                memory=subprocess.check_output(['nvidia-smi','--id=6','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                if int(memory.strip())>128: raise RuntimeError('GPU6 not idle; do not interfere')
                run_id=base+f'-sync-g{group}-candidate{k}'
                if (outputs/run_id).exists(): raise FileExistsError(run_id)
                print(json.dumps(dict(status='STARTING',run_id=run_id,candidate=k,group=group)),flush=True)
                subprocess.run(['bash',str(wrapper),run_id,str(n),str(k),base+'-sync-reference','','host',str(seed),
                    base+'-sync-schedule.json',str(group)],cwd=ROOT,check=True)
                p=torch.load(outputs/run_id/'panel.pt',map_location='cpu',weights_only=False)
                for key in ('initial_fingerprint','simulation_contract','model_fingerprint','rms_fingerprint'):
                    if p[key]!=p0[key]: raise ValueError('backend/actor drift')
                if not torch.equal(p['triggers'][rows],p0['triggers'][rows]): raise ValueError('group trigger mismatch')
                error=p['prefix_errors'][rows].amax(0).tolist()
                if p['candidate']!=k or not prefix_ok(p)[rows].all() or not p['valid_steps'][rows].all():
                    raise RuntimeError('candidate prefix mismatch; no treatment row dropping:'+str(error))
                print(json.dumps(dict(status='PAIRED',run_id=run_id,anchors=len(rows),prefix_max=error)),flush=True)
    print(json.dumps(dict(status='COMPLETED',wall_seconds=time.monotonic()-begin)),flush=True)
if __name__=='__main__':main()
