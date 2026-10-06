#!/usr/bin/env python3
"""Reconstruct frozen checkpoint values and all paired stats without simulation."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src'),str(ROOT/'src/task/cm-interaction-oracle/tools/run')]
from critic_ranking import FrozenCritic,bootstrapped_score,selection_summary
from run_rolling_gt_y import CHECKPOINT,sha
from oracle_y_utility import utility,stable_grasp_z
from rolling_y_audit import short_y


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True);args=ap.parse_args()
    if (args.run_dir/'replay.json').exists():raise ValueError('refusing existing replay')
    start=time.monotonic();torch.set_num_threads(2)
    manifest=json.loads((args.run_dir/'manifest.json').read_text())
    if manifest['status']!='COMPLETED':raise ValueError('completed main required')
    for path,expected in manifest['input_sha256'].items():
        target=args.run_dir/'protocol.md' if '/docs/experiments/probes/' in path else Path(path)
        if sha(target)!=expected:raise ValueError('frozen input/code drift: '+str(target))
    result=json.loads((args.run_dir/'result.json').read_text());saved=np.load(args.run_dir/'scores.npz')
    for path,h in result['input_sha256'].items():
        if sha(Path(path))!=h:raise ValueError('matched source/output drift')
    critic=FrozenCritic(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)).to('cuda:0')
    actual=[];ys=[];zs=[];rs=[];ds=[];max_error=0.;identity=[]
    with torch.no_grad():
        for seed in (263,264):
            for group in (0,1):
                raw_values=[];group_y=[];group_z=[];reward=[];done=[];rows_ref=None
                for k in range(7):
                    if time.monotonic()-start>100:raise TimeoutError('120second GPU replay budget')
                    packet=torch.load(args.run_dir/f's{seed}-g{group}-k{k}'/'critic.pt',map_location='cpu',weights_only=False)
                    rows=packet['rows']
                    if rows_ref is not None and not torch.equal(rows_ref,rows):raise ValueError('row join drift')
                    rows_ref=rows
                    if (packet['candidate'],packet['group'])!=(k,group):raise ValueError('candidate join drift')
                    pred=critic(packet['observations8'].to('cuda:0')).cpu()
                    max_error=max(max_error,float((pred-packet['live_values8']).abs().max()))
                    old=torch.load(packet['original_candidate_path'],map_location='cpu',weights_only=False)
                    y,_=short_y(old['before'][rows,2],old['before'][rows,71]>.5,old['height'][rows,:32],old['pair'][rows,:32],old['rest_height'][rows])
                    z,_=stable_grasp_z(old['height'][rows],old['pair'][rows],old['rest_height'][rows])
                    raw_values.append(pred);group_y.append(y);group_z.append(z);reward.append(packet['raw_rewards8']);done.append(packet['dones8'])
                actual.append(torch.stack(raw_values,1));ys.append(torch.stack(group_y,1));zs.append(torch.stack(group_z,1))
                rs.append(torch.stack(reward,1));ds.append(torch.stack(done,1));identity.extend([(seed,group,int(r)) for r in rows_ref])
    v,y,z,r,d=[torch.cat(x).numpy() for x in (actual,ys,zs,rs,ds)]
    s,rp,vp=bootstrapped_score(r,v,d)
    arrays=dict(value8=v,y=y,z=z,rewards8=r,score=s,reward_component=rp,bootstrap_component=vp)
    array_errors={k:float(np.max(np.abs(value.astype(float)-saved[k].astype(float)))) for k,value in arrays.items()}
    arms={a:selection_summary(score,z,utility(y)) for a,score in [('Critic_Q8',s),('Value8_only',v),('Reward8_only',rp),('GT_Y',utility(y))]}
    exact=arms==result['arms']
    if identity != [(r['seed'],r['group'],r['row']) for r in result['records']]:raise ValueError('ordered anchor identities drift')
    report=dict(status='PASS' if max_error<=1e-5 and max(array_errors.values())<=1e-5 and exact else 'FAIL',
                live_value_max_error=max_error,array_max_errors=array_errors,exact_statistics=exact,
                checkpoint_sha256=sha(CHECKPOINT),audit_tool_sha256=sha(Path(__file__).resolve()),
                elapsed_seconds=time.monotonic()-start,no_fitting_or_simulation=True)
    (args.run_dir/'replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise ValueError('critic replay mismatch')


if __name__=='__main__':main()
