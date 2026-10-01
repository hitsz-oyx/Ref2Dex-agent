"""Recheck all evidence and quantify requested versus executed target deltas."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'src/task/CmResidual/research/contact_response/output'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def main():
    parent=BASE/'P-20261002-selective-finger-feasibility-r1'
    out=BASE/'P-20261002-selective-finger-closeout-r1'
    if out.exists():raise FileExistsError(out)
    m=json.loads((parent/'run_manifest.json').read_text())
    assert m['run_status']=='COMPLETED' and m['label']=='UNPROMISING'
    for path,digest in m['input_sha256'].items():assert sha(path)==digest,path
    torch.set_num_threads(2)
    events=[]
    for seed in (545,546):
        i=torch.load(parent/f's{seed}/initial.pt',map_location='cpu',weights_only=False)
        t=torch.load(parent/f's{seed}/trace.pt',map_location='cpu',weights_only=False)
        rows=np.arange(768);tick=i['decision_steps'].numpy()
        actual=t['effective_finger_delta'].numpy()[tick,rows]
        # This is a realized TARGET offset, not measured future hand movement.
        requested=i['primitive_parameters'].numpy()
        for env in range(768):
            selected=requested[env]>0
            events.append(dict(seed=seed,motion=int(i['motion'][env]),arm=int(i['arm_assignment'][env]),
                selected_coordinates=int(selected.sum()),target_null_coordinates=int((np.abs(actual[env,selected])<=1e-7).sum()),
                target_saturated_coordinates=int((actual[env,selected]<requested[env,selected]-1e-6).sum()),
                requested_delta=requested[env].tolist(),executed_target_delta=actual[env].tolist()))
    stats={}
    for mo in range(3):
        stats[str(mo)]={}
        for arm in range(1,8):
            group=[r for r in events if r['motion']==mo and r['arm']==arm]
            stats[str(mo)][str(arm)]=dict(n=len(group),selected_coordinates=sum(r['selected_coordinates'] for r in group),
                target_null_coordinates=sum(r['target_null_coordinates'] for r in group),target_saturated_coordinates=sum(r['target_saturated_coordinates'] for r in group))
    report=dict(run_status='COMPLETED',scientific_label='UNPROMISING',protected_inputs_verified=len(m['input_sha256']),
        own_run_bytes=sum(p.stat().st_size for p in parent.rglob('*') if p.is_file()),wall_seconds=m['wall_seconds'],
        target_delta_statistics=stats,not_measured_hand_motion=True,
        input_sha256={str(p):sha(p) for p in (parent/'run_manifest.json',parent/'results.json',parent/'collection_audit.json',Path(__file__).resolve())})
    out.mkdir()
    (out/'requested_and_executed_target_rows.json').write_text(json.dumps(events,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
