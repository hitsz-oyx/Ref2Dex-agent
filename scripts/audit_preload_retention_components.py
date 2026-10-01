#!/usr/bin/env python3
"""Separate geometric retention from historical force-proxy interruption."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_static_hold_feasibility import longest
BASE=ROOT/'src/task/CmResidual/research/contact_response/output'

def main():
    source=BASE/'P-20261002-finger-preload-feasibility-r1';metadata=BASE/'P-20261002-frame0-tracking-feasibility-r1/s506/physical_metadata.json'
    output=source/'retention_component_audit_r1';output.mkdir(exist_ok=False)
    m=json.loads((source/'run_manifest.json').read_text());primary=json.loads((source/'results.json').read_text())
    if m['run_status']!='COMPLETED' or primary['label']!='UNPROMISING':raise ValueError('original outcome drift')
    if any(sha(Path(p))!=h for p,h in m['input_sha256'].items()):raise ValueError('protected input drift')
    inputs={str(source/'run_manifest.json'):sha(source/'run_manifest.json'),str(source/'results.json'):sha(source/'results.json'),str(metadata):sha(metadata),str(Path(__file__).resolve()):sha(Path(__file__))}
    physical=json.loads(metadata.read_text());masses=[r['mass'] for r in physical['object_body_properties']]
    rows=[]
    for seed in (504,505):
        d=source/f's{seed}';r=json.loads((d/'results.json').read_text())
        for name in ('initial','trace'):
            p=d/(name+'.pt');inputs[str(p)]=sha(p)
            if inputs[str(p)]!=r[name+'_sha256']:raise ValueError('retained trace hash drift')
        a=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);b=torch.load(d/'trace.pt',map_location='cpu',weights_only=False)
        z=b['object_root'][:,:,2].numpy();pair=b['contact'].bool().all(-1).numpy()
        geometric=(z-a['rest_height'].numpy()[None]>=np.float32(.03))&(z-a['initial_height'].numpy()[None]>=np.float32(-.01))&(b['clearance'].numpy()>=np.float32(.02))
        for env in range(96):
            strict=longest(geometric[:,env]&pair[:,env]);g=longest(geometric[:,env])
            if strict!=int(b['max_hold_steps'][env]):raise ValueError('primary label reconstruction')
            rows.append(dict(seed=seed,env=env,motion=int(a['motion'][env]),arm=int(a['dose_assignment'][env]),strict75=strict>=75,
                geometry75=g>=75,geometry_only_max_steps=g,force_proxy_false_ticks=int((geometric[:,env]&~pair[:,env]).sum()),
                max_false_proxy_run_while_geometrically_held=longest(geometric[:,env]&~pair[:,env])))
    def summarize(group):return dict(n=len(group),primary_strict75_count=sum(x['strict75'] for x in group),geometry75_count=sum(x['geometry75'] for x in group),
        geometry75_but_strict75_failed=sum(x['geometry75'] and not x['strict75'] for x in group),
        geometry75_with_proxy_lost6_count=sum(x['geometry75'] and x['max_false_proxy_run_while_geometrically_held']>=6 for x in group))
    arms={str(k):dict(pooled=summarize([x for x in rows if x['arm']==k]),motions={str(j):summarize([x for x in rows if x['arm']==k and x['motion']==j]) for j in range(3)}) for k in range(4)}
    result=dict(run_status='COMPLETED',status='POSTHOC_MEASUREMENT_DIAGNOSTIC',primary_label_unchanged='UNPROMISING',arms=arms,no_new_physics=True,input_sha256=inputs,
        native_tracking_mass_kg_range=[min(masses),max(masses)],weight_N_range=[min(masses)*abs(physical['gravity'][2]),max(masses)*abs(physical['gravity'][2])],
        proxy_threshold_N=.1,threshold_to_native_weight_ratio=.1/(np.mean(masses)*abs(physical['gravity'][2])),
        boundary='component disagreement from retained suspended-start data; native mass measured in separate tracking runtime; geometry does not identify pairwise forces or certify force closure; no primary gate changed')
    torch.save(dict(rows=rows),output/'rows.pt');(output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
