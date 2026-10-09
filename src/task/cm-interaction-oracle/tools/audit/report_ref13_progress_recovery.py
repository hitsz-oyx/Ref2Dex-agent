"""Report complete executed Z90 outcomes; retain partial runs without claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
import numpy as np
import torch
from oracle_y_utility import stable_grasp_z,paired_bootstrap
from rolling_control import execution_z,TOLERANCE


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); root=args.run_dir.resolve()
    audit=json.loads((root/'engineering-audit.json').read_text())
    rolling=json.loads((root/'rolling-status.json').read_text())
    result=dict(status='UNCLEAR',engineering=audit['repeat'],execution_status=rolling['status'],
        scope='one seed, synchronous s3 cohort, reconstructed actor; three replans and bounded Z90 only',
        full_episode_gate1_passed=False,formal_validation=False,records=rolling['records'])
    paths=[root/'engineering-audit.json',root/'s3-schedule.json',root/'rolling-status.json']
    if rolling['status']=='COMPLETED':
        baseline=load(root/'reanchor-r2/panel.pt')
        rows=torch.tensor(audit['s3_group_rows']);origin=int(baseline['triggers'][rows][0])
        z0=stable_grasp_z(baseline['height'][rows],baseline['pair'][rows],baseline['rest_height'][rows])[0].numpy().astype(int)
        outcomes={'baseline':z0}; one_shot={'baseline':z0}
        for arm in ('old','new'):
            records=[v for v in rolling['records'] if v['arm']==arm]
            if [v['offset'] for v in records]!=[0,8,16]:
                raise ValueError('three real replans required')
            folder=root/records[-1]['actual_path'];panel=load(folder/'panel.pt');trace=load(folder/'trace.pt')
            for key in ('initial_fingerprint','simulation_contract','model_fingerprint','rms_fingerprint'):
                if baseline[key]!=panel[key]:
                    raise ValueError('executed outcome provenance mismatch')
            if not (panel['full_world_prefix_errors']<=TOLERANCE).all():
                raise ValueError('executed replay mismatch')
            outcomes[arm]=execution_z(trace,rows,origin,baseline['rest_height'][rows])[0].numpy().astype(int)
            paths.extend([folder/'panel.pt',folder/'trace.pt'])
            first=root/records[0]['actual_path']
            one_shot[arm]=execution_z(load(first/'trace.pt'),rows,origin,baseline['rest_height'][rows])[0].numpy().astype(int)
            paths.extend([first/'panel.pt',first/'trace.pt'])
        counts={k:int(v.sum()) for k,v in outcomes.items()}
        comparisons={}
        for left,right in (('old','baseline'),('new','baseline'),('new','old')):
            a,b=outcomes[left],outcomes[right]
            comparisons[left+'_vs_'+right]=dict(gain=paired_bootstrap(a-b),
                rescued=int(((a==1)&(b==0)).sum()),harmed=int(((a==0)&(b==1)).sum()))
        start=baseline['start_frame'][rows].numpy()
        strata={label:dict(n=int(mask.sum()),counts={k:int(v[mask].sum()) for k,v in outcomes.items()})
                for label,mask in (('frame0',start==0),('hybrid_nonzero',start>0))}
        result.update(anchors=len(rows),origin=origin,counts=counts,comparisons=comparisons,
            one_shot_counts={k:int(v.sum()) for k,v in one_shot.items()},
            one_shot_outcomes={k:v.tolist() for k,v in one_shot.items()},
            adequate_old_probe_support=False,strata=strata,
            outcomes={k:v.tolist() for k,v in outcomes.items()},
            interpretation='Matched bounded exploratory observations; one motion cannot satisfy historical >=30anchors/>=2motions utility gate.')
    result['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (root/'result-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('engineering','records','input_sha256','outcomes')},indent=2))


if __name__=='__main__':
    main()
