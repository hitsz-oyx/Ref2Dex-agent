"""Recompute actual task and reset model-domain admission for the fixed cohort."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.task_admission import load_executable_tasks
from wam_rl_policy_v3.model_domain import current_v4_domain_sample,DOMAIN_POLICY
from wam_rl_policy_v3.wam_v4_runtime import load_geometry,V4Runtime
from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/manifest.json'
    entries,admission=load_executable_tasks(manifest)
    objects=[dict(object_code='dexplore/'+r['object'],scale=1.) for r,_,_ in entries]
    geometry=load_geometry(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache',objects)
    ckpt=ROOT/'outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/step80000-model.pt'
    packet=torch.load(ckpt,map_location='cpu',weights_only=False);runtime=SimpleNamespace(config=packet['config']);del packet
    records=[]
    for (row,spec,side),geo in zip(entries,geometry):
        with np.load(Path(row['source'])/'trajectory.npz') as z:
            sample=current_v4_domain_sample(z['obj'][0],z['hand'][0,NATIVE_TO_WAM],geo)
        records.append(dict(sequence=row['sequence'],terminal_mode=spec.terminal_mode.value,
            model_domain_admitted=bool(V4Runtime.workspace_valid(runtime,[sample])[0])))
    legacy=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json'
    try:load_executable_tasks(legacy);legacy_error=None
    except ValueError as error:legacy_error=str(error)
    report=dict(status='PASS' if all(r['model_domain_admitted'] for r in records) and legacy_error else 'FAIL',
        task_admission=admission,domain_policy=DOMAIN_POLICY,actual_v4_checkpoint=str(ckpt),
        domain_admitted=sum(r['model_domain_admitted'] for r in records),domain_rejected=sum(not r['model_domain_admitted'] for r in records),
        legacy_manifest_rejected=legacy_error,records=records,
        scope='Saved initial states only; no physics, no claim about later model-domain occupancy or complete task success.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','domain_admitted','domain_rejected','legacy_manifest_rejected']}))
    return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
