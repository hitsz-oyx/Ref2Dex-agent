"""Independently rebuild labels, first PD metrics and candidate selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/tools/audit')]
from ptv3_shared_wam.paths import add_dependencies
add_dependencies()
from trajectory_policy.chunk_retargeter import ChunkRetargeter
from audit_competent_pd import independent_pd,ACTIVE


def arrays(source,split,indices):
    corpus=json.loads((source/'manifest.json').read_text());records=[r for r in corpus['records'] if r['split']==split]
    keys=[(tick,i) for i,r in enumerate(records) for tick in r['admitted_ticks']]
    selected=np.asarray([keys[int(i)] for i in indices]);n=len(indices)
    state=np.zeros((n,87),np.float32);tau=np.zeros((n,24,33),np.float32)
    action=np.zeros((n,8,12),np.float32);q=np.zeros((n,18),np.float32);pd=np.zeros((n,18),np.float32)
    for seq in np.unique(selected[:,1]):
        rows=np.flatnonzero(selected[:,1]==seq);t=selected[rows,0]
        with np.load(source/records[seq]['path']/'trajectory.npz') as d:
            rotation=d['obj'][t,:3,:3];hand=d['hand'][t];center=d['obj'][t,:3,3]
            local=(hand-center[:,None])@rotation
            state[rows]=np.concatenate((d['q'][t],d['dq'][t],local.reshape(-1,33),d['obj'][t,:3].reshape(-1,12),d['velocity'][t]),axis=-1)
            delta=d['hand'][t[:,None]+np.arange(1,25)]-hand[:,None]
            tau[rows]=np.einsum('ntpj,njk->ntpk',delta,rotation).reshape(-1,24,33)
            action[rows]=d['applied'][t[:,None]+np.arange(8)][...,ACTIVE]
            q[rows]=d['q'][t];pd[rows]=d['pd_targets'][t]
    return state,tau,action,q,pd,corpus['native_controller']


@torch.no_grad()
def replay(path,panel):
    packet=torch.load(path,map_location='cpu',weights_only=False);s=packet['model'];net=ChunkRetargeter(128)
    if s['action_queries'].shape[0]==1:
        net.action_queries=torch.nn.Parameter(net.action_queries[:1].clone())
        net.action_mean=net.action_mean[:1].clone();net.action_scale=net.action_scale[:1].clone()
    net.load_state_dict(s);net.eval();state,tau,target,q,actual_pd,controller=panel
    pred=np.concatenate([net(torch.tensor(state[i:i+128]),torch.tensor(tau[i:i+128])).numpy() for i in range(0,len(state),128)])
    native=np.zeros((len(state),18),np.float32);native[:,ACTIVE]=pred[:,0].clip(-1,1)
    difference=independent_pd(native,q,controller)[:,ACTIVE]-actual_pd[:,ACTIVE]
    xyz=np.linalg.norm(difference[:,:3],axis=-1)*1000
    return dict(xyz_mean_mm=float(xyz.mean()),xyz_p95_mm=float(np.percentile(xyz,95)),
        rotation_p95_rad=float(np.percentile(np.abs(difference[:,3:6]).max(-1),95)),
        finger_p95_rad=float(np.percentile(np.abs(difference[:,6:]).max(-1),95)),
        clipping=float((np.abs(pred[:,0])>1).any(-1).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--memorization',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.run/'manifest.json').read_text());source=Path(manifest['source'])
    with np.load(a.run/'panels.npz') as f:
        train=arrays(source,'train',f['fit_indices'][f['TRAIN_report_ids']]);dev=arrays(source,'val',f['dev_indices'])
    result=json.loads((a.run/'result.json').read_text());checks={};maxima={}
    first=[torch.load(a.run/(arm+'-step0.pt'),map_location='cpu',weights_only=False)['model'] for arm in ['A_all8','B_control']]
    checks['A_B_initial_state_exact']=all(torch.equal(first[0][k],first[1][k]) for k in first[0])
    for arm in result['arms']:
        curve=json.loads((a.run/(arm+'-curve.json')).read_text())
        selected=min(curve,key=lambda r:(r['panels']['dev']['xyz_p95_mm'],r['panels']['dev']['raw_first_MAE']))
        path=a.run/(arm+'_best.pt');packet=torch.load(path,map_location='cpu',weights_only=False)
        checks[arm+'_selection']=packet['step']==selected['step']==result['arms'][arm]['selected']['step']
        for panel_name,panel in [('TRAIN',train),('dev',dev)]:
            actual=replay(path,panel);expected=selected['panels'][panel_name]
            maxima[arm+'/'+panel_name]={k:abs(v-expected[k]) for k,v in actual.items()}
    with np.load(a.memorization/'panels.npz') as f:small=arrays(source,'train',f['TRAIN_indices'])
    memo=json.loads((a.memorization/'result.json').read_text())
    for name,panel in [('TRAIN',small),('dev',dev)]:
        actual=replay(a.memorization/'step2000.pt',panel);expected=memo['final']['panels'][name]
        maxima['memorization/'+name]={k:abs(v-expected[k]) for k,v in actual.items()}
    passed=all(checks.values()) and all(value<1e-3 for row in maxima.values() for value in row.values())
    report=dict(pass_=passed,checks=checks,maxima=maxima,scope='independent CPU label/PD replay; no physics or causal claim')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    if not passed:raise SystemExit('audit mismatch')


if __name__=='__main__':main()
