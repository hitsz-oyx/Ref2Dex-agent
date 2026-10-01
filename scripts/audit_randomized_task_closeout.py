#!/usr/bin/env python3
"""Check policy choices, fit-only globals and all protected inputs at closure."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2)
    root=args.directory;m=json.loads((root/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED':raise ValueError('nonterminal run')
    for path,value in m['input_sha256'].items():
        if sha(Path(path))!=value:raise ValueError('protected input drift '+path)
    controls=torch.load(root/'frozen_controls.pt',map_location='cpu',weights_only=False)
    fit=torch.load(Path(controls['models'])/'fit_data.pt',map_location='cpu',weights_only=False)
    y=fit['y'][:,2].double().numpy();a=fit['arm'].numpy();p=fit['propensity'].double().numpy()
    globals_=np.array([(y*(a==arm)/p[:,arm]).mean() for arm in range(7)])
    np.testing.assert_allclose(globals_,controls['global_vertical_scores'].numpy(),atol=1e-14,rtol=1e-12)
    decision_count=0;episodes_count=0
    for actor,evaluation in m['panels']:
        folder=root/f't{actor}_s{evaluation}';r=json.loads((folder/'results.json').read_text())
        for name in ('episodes','decisions'):
            if sha(folder/(name+'.pt'))!=r[name+'_sha256']:raise ValueError('output drift')
        episodes=torch.load(folder/'episodes.pt',map_location='cpu',weights_only=False)
        d=torch.load(folder/'decisions.pt',map_location='cpu',weights_only=False)
        assignment=episodes['assignment'].numpy();ref=d['reference_action'].numpy();valid=np.ones((len(ref),7),dtype=bool)
        for arm in range(1,7):
            requested=ref[:,(arm-1)//2]+np.float32(.01 if arm%2 else -.01)
            valid[:,arm]=(requested>=-1)&(requested<=1)
        if not np.array_equal(valid,d['valid'].numpy()):raise ValueError('candidate validity drift')
        expected=np.zeros(len(ref),dtype=np.int64);policies=assignment[d['environment'].numpy()]
        global_arm=np.where(valid,globals_[None],-np.inf).argmax(-1)
        conditional=np.where(valid,d['prediction'][:,:,2].numpy(),-np.inf).argmax(-1)
        expected[policies==2]=global_arm[policies==2];expected[policies==3]=conditional[policies==3]
        for env in range(768):
            rows=(d['environment']==env).nonzero().flatten().tolist()
            generator=torch.Generator().manual_seed(21000+1000*evaluation+env)
            for row in rows:
                options=np.flatnonzero(valid[row]);random=int(options[int(torch.randint(len(options),(1,),generator=generator))])
                if assignment[env]==1:expected[row]=random
        if not np.array_equal(expected,d['selected'].numpy()):raise ValueError('policy choice mismatch')
        decision_count+=len(ref);episodes_count+=len(episodes['episodes'])
    trajectory=json.loads((root/'trajectory_audit.json').read_text())
    if trajectory['status']!='PASS' or trajectory['episodes_independently_recomputed']!=3072:raise ValueError('trajectory audit incomplete')
    result=dict(status='PASS',protected_inputs_verified=len(m['input_sha256']),episodes_verified=episodes_count,
        policy_choices_verified=decision_count,fit_only_global_scores_independently_recomputed=True,
        trajectory_audit=trajectory,primary_gate_unchanged=True)
    (root/'closeout_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
