#!/usr/bin/env python3
"""Freeze baseline-repeat cohort before treatment runs; evaluate GT-Y selector."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from consequence_sufficiency import readouts
from oracle_y_utility import CANDIDATES,stable_grasp_z,oracle_gate,noise_curve,utility
from src.task.CmResidual.paired_evaluation import wilson_upper


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return torch.load(path,map_location='cpu',weights_only=False)
def outcomes(p):
    _,_,_,y,_=readouts(p)
    z,detail=stable_grasp_z(p['height'],p['pair'],p['rest_height'])
    return y,z,detail


def prefix_ok(p):
    tolerance=torch.tensor([1e-4,1e-4,1e-4,1e-4,1e-4,1e-5,0.])
    return (p['prefix_errors']<=tolerance).all(-1)


def identity(a,b):
    for key in ('triggers','motion_id','start_frame','rest_height','delta'):
        if not torch.equal(a[key],b[key]): raise ValueError('identity drift:'+key)
    if a['initial_fingerprint']!=b['initial_fingerprint']: raise ValueError('initial drift')
    if not b['valid_steps'][b['triggers']>=0].all(): raise ValueError('incomplete assigned panel')


def screen(reference,repeat,output):
    if output.exists(): raise ValueError('screen output must be unique/frozen')
    a,b=load(reference/'panel.pt'),load(repeat/'panel.pt'); identity(a,b)
    ya,za,_=outcomes(a);yb,zb,_=outcomes(b)
    assigned=a['triggers']>=0
    eligible=assigned&prefix_ok(b)&((ya-yb).abs().amax(-1)<=.05)&(za==zb)
    n=int(assigned.sum()); accepted=int(eligible.sum())
    result=dict(kind='PRE-TREATMENT-BASELINE-REPEAT-SCREEN',passed=n>0 and accepted>=.8*n,
        assigned=n,accepted=accepted,indices=eligible.nonzero().flatten().tolist(),
        motion_counts=[int((eligible&(a['motion_id']==m)).sum()) for m in range(3)],
        prefix_max=b['prefix_errors'][assigned].amax(0).tolist() if n else [],
        y_max_abs=float((ya-yb)[assigned].abs().max()) if n else None,
        z_disagreements=int((za!=zb)[assigned].sum()),
        z_discordance_upper95=wilson_upper(int((za!=zb)[assigned].sum()),n) if n else None,
        initial_fingerprint=a['initial_fingerprint'],
        inputs={str(p.resolve()):sha(p) for p in (reference/'panel.pt',repeat/'panel.pt')},
        limitations='Probe empirical repeatability; Wilson bound is reported, not a population-noise validation gate.')
    output.write_text(json.dumps(result,indent=2)+'\n'); return result


def evaluate(reference,branches,screen_path,output):
    if output.exists(): raise ValueError('result must be unique')
    frozen=json.loads(screen_path.read_text())
    if not frozen['passed']: raise ValueError('baseline repeat screen failed')
    if len(branches)!=6: raise ValueError('need all six alternative branches')
    p0=load(reference/'panel.pt'); rows=torch.tensor(frozen['indices'],dtype=torch.long)
    panels=[p0,*[load(p/'panel.pt') for p in branches]]
    inputs=[reference/'panel.pt',screen_path,*[p/'panel.pt' for p in branches]]
    for p,h in frozen['inputs'].items():
        if sha(Path(p))!=h: raise ValueError('screen input mutated')
    y,z=[],[]; diagnostics=[]
    for k,p in enumerate(panels):
        identity(p0,p)
        if p['candidate']!=k or not prefix_ok(p)[rows].all():
            raise ValueError('candidate identity/prefix screen failed; do not discard treatment rows')
        yi,zi,details=outcomes(p);y.append(yi[rows]);z.append(zi[rows])
        displacement=(p['fingertip_positions'][:,7]-p0['fingertip_positions'][:,7])[rows].norm(dim=-1)
        diagnostics.append(dict(candidate=CANDIDATES[k],z_success=int(zi[rows].sum()),
            y_mean=yi[rows].mean(0).tolist() if len(rows) else [],
            utility_mean=float(utility(yi[rows]).mean()) if len(rows) else None,
            clipped_steps=int(p['clipped_steps'][rows].sum()),
            realized_step8_tip_delta_mean_mm=float(displacement.mean()*1000) if len(rows) else None,
            realized_step8_tip_delta_max_mm=float(displacement.max()*1000) if len(rows) else None,
            prefix_max=p['prefix_errors'][rows].amax(0).tolist() if len(rows) else []))
    yy=torch.stack(y,1).numpy();zz=torch.stack(z,1).numpy();motion=p0['motion_id'][rows].numpy()
    result=oracle_gate(yy,zz,motion)
    result.update(candidates=list(CANDIDATES),per_candidate=diagnostics,screen=frozen,
        input_sha256={str(p.resolve()):sha(p) for p in inputs},
        git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        utility_contract='height-held fraction32 + .25 contact32 - late height-failure32; ties baseline first',
        z_contract='45 consecutive held steps bystep60, no subsequent drop throughstep90',
        gate_b=noise_curve(yy,zz) if result['passed'] else 'NOT_RUN_GATE_A_NOT_POSITIVE',
        gate_c='NOT_RUN_REQUIRES_QUALIFIED_A_B_AND_SEPARATE_PAIRED_TRAIN_TEST_PROTOCOL',
        gate_d='NOT_RUN_REQUIRES_PREVIOUS_GATES')
    output.write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(output.with_suffix('.npz'),y=yy,z=zz,motion=motion,rows=rows.numpy())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--repeat',type=Path)
    parser.add_argument('--screen',type=Path,required=True);parser.add_argument('--branches',type=Path,nargs='*')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.repeat: result=screen(args.reference,args.repeat,args.screen)
    else: result=evaluate(args.reference,args.branches,args.screen,args.output)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
