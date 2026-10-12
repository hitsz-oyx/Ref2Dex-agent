"""Reconstruct joint raw labels and metrics from measured sources, without NN compute."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'tools/audit'),str(TASK/'src')]
from audit_wam_ref2_shadow import sha,write


def audit(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    if (manifest.get('status')!='COMPLETED' or result.get('status')!='COMPLETED'
            or manifest.get('result_sha256')!=sha(directory/'result.json')
            or manifest.get('predictions_sha256')!=sha(directory/'predictions.npz')):
        raise ValueError('completed authenticated joint fit required')
    owner=ROOT/'outputs/trajectory-policy'
    paths={'nominal':owner/'competent-pd-teacher-20261011-r2/trajectory.npz',
        'recovery':owner/'wam-ref2-execution-startup-anchor-20261012-r1/native/trajectory.npz'}
    data={}
    for name,path in paths.items():
        if manifest['input_sha256'].get(str(path))!=sha(path):raise ValueError('measured label source drift')
        with np.load(path,allow_pickle=False) as f:
            data[name]={k:f[k] for k in ('q','hand','obj','pd_targets')+(() if name=='nominal' else ('expert_PD',))}
    checks={};maxima={}
    with np.load(directory/'predictions.npz',allow_pickle=False) as f:
        raw={k:f[k] for k in f.files}
    for phase in ('initial','final'):
        for panel,stats in result[phase].items():
            prefix=phase+'_'+panel+'_';ids=raw[prefix+'row_ids']
            valid=raw[prefix+'tau_label_valid'];true=raw[prefix+'tau_true'];PD=raw[prefix+'pd_true']
            for j,index in enumerate(ids):
                kind,tick,env=raw['rows_kind'][index],int(raw['rows_tick'][index]),int(raw['rows_role'][index])
                d=data[str(kind)];expected_valid=kind=='nominal' or env==3
                if valid[j]!=expected_valid:raise ValueError('future admission mask differs')
                expected_pd=d['pd_targets'][tick,env] if kind=='nominal' else d['expert_PD'][tick,env-2]
                maxima['PD_label_max']=max(maxima.get('PD_label_max',0.),float(np.abs(PD[j]-expected_pd).max()))
                if expected_valid:
                    future=d['hand'][tick+np.arange(1,25),env]
                    expected=((future-d['hand'][tick,env]) @ d['obj'][tick,env,:3,:3]).reshape(24,33)
                    maxima['tau_label_max']=max(maxima.get('tau_label_max',0.),float(np.abs(true[j]-expected).max()))
                elif np.any(true[j]):raise ValueError('pure PD rows have a nonzero placeholder future')
            delta=raw[prefix+'pd']-PD
            recomputed={'xyz_p95_mm':np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000,
                'rotation_p95_rad':np.percentile(np.abs(delta[:,3:6]).max(-1),95),
                'finger_p95_rad':np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95),
                'clipping_rate':raw[prefix+'clipped'].mean()}
            if valid.any():
                e=np.linalg.norm((raw[prefix+'tau'][valid]-true[valid]).reshape(-1,24,11,3),axis=-1)*1000
                recomputed.update(first8_epe_mm=e[:,:8].mean(),all24_epe_mm=e.mean(),h1_epe_mm=e[:,0].mean(),wrist_h1_epe_mm=e[:,0,0].mean())
                gt=(raw[prefix+'GT_tau_R_PD']-PD)[valid]
                recomputed.update(GT_tau_R_xyz_p95_mm=np.percentile(np.linalg.norm(gt[:,:3],axis=-1),95)*1000,
                    GT_tau_R_rotation_p95_rad=np.percentile(np.abs(gt[:,3:6]).max(-1),95),
                    GT_tau_R_finger_p95_rad=np.percentile(np.abs(gt[:,[6,8,10,12,14,15]]).max(-1),95),
                    GT_tau_R_clipping_rate=raw[prefix+'GT_tau_R_clipped'][valid].mean())
            for key,value in recomputed.items():
                checks[prefix+key]=float(abs(stats[key]-value))
    maxima['metric_max']=max(checks.values())
    passed=(maxima['PD_label_max']<1e-6 and maxima['tau_label_max']<1e-6 and maxima['metric_max']<1e-5
        and all(sha(p)==h for p,h in manifest['input_sha256'].items()))
    report=dict(schema='ref2dex.joint-fit-root-audit.v1',pass_=bool(passed),maxima=maxima,
        checks=checks,metric_checks=len(checks),pure_future_geometry_denied=True,
        source_hashes={str(p):sha(p) for p in paths.values()},
        note='CPU file/label/metric reconstruction only; no model or physics quality claim')
    report['pass']=report.pop('pass_')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--cross-retargeter-gpu',type=int,default=None);a=p.parse_args()
    directory=a.fit.resolve()
    if ROOT/'outputs/trajectory-policy' not in directory.parents:raise ValueError('task-owned fit required')
    report=audit(directory)
    if a.cross_retargeter_gpu is not None:
        report['cross_retargeter']=cross_retargeter(directory,a.cross_retargeter_gpu)
    write(directory/'root_joint_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
    if not report['pass']:raise SystemExit(1)


def cross_retargeter(directory,gpu):
    """Same measured state, swap only R/forecast; descriptive fit audit."""
    if gpu!=2 or subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('free assigned GPU2 required for R model compute')
    scratch=ROOT/'tmp/trajectory-policy/wam-joint-audit';scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='2',TMPDIR=str(scratch))
    import torch
    from trajectory_policy.pd_retargeter import load_model,decode_target
    from trajectory_policy.wam_retargeter_admission import validate_r_packet
    from trajectory_policy.wam_joint import state_hash,policy_patch_admission
    torch.set_num_threads(2)
    owner=ROOT/'outputs/trajectory-policy';manifest=json.loads((directory/'manifest.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    bp=torch.load(owner/'competent-pd-fit-20261011-r1/tau.pt',map_location='cpu',weights_only=False)
    rp=torch.load(directory/'tau.pt',map_location='cpu',weights_only=False);validate_r_packet(rp,bp)
    head=torch.load(directory/'head.pt',map_location='cpu',weights_only=False)
    policy_patch_admission(head,manifest,result)
    if (sha(directory/'head.pt')!=result['head_sha256'] or sha(directory/'tau.pt')!=result['checkpoint_sha256']
            or state_hash({k[4:]:v for k,v in rp['model'].items() if k.startswith('net.')})!=result['train_final_hash']['R']):
        raise ValueError('serialized learned pair differs')
    del head
    models={'original_R':load_model(bp,'cuda'),'joint_R':load_model(rp,'cuda')}
    data={}
    for kind,path in (('nominal',owner/'competent-pd-teacher-20261011-r2/trajectory.npz'),
            ('recovery',owner/'wam-ref2-execution-startup-anchor-20261012-r1/native/trajectory.npz')):
        with np.load(path,allow_pickle=False) as f:data[kind]={k:f[k] for k in ('q','dq','hand','obj','velocity')}
    with np.load(directory/'predictions.npz',allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    output={}
    for panel in ('nominal_report','startup_all','recovery_training','pure_PD_training','pure_PD_diagnostic'):
        prefix='final_'+panel+'_';ids=raw[prefix+'row_ids'];states=[];qs=[]
        for index in ids:
            kind,tick,env=str(raw['rows_kind'][index]),int(raw['rows_tick'][index]),int(raw['rows_role'][index])
            d=data[kind];obj=d['obj'][tick,env]
            local=(d['hand'][tick,env]-obj[:3,3])@obj[:3,:3]
            states.append(np.r_[d['q'][tick,env],d['dq'][tick,env],local.ravel(),obj[:3,:].ravel(),d['velocity'][tick,env]])
            qs.append(d['q'][tick,env])
        state=torch.as_tensor(np.asarray(states),device='cuda');q=torch.as_tensor(np.asarray(qs),device='cuda')
        target=raw[prefix+'pd_true'];output[panel]={}
        with torch.inference_mode():
            for model_name,model in models.items():
                forecasts={'initial_tau':raw['initial_'+panel+'_tau'],'final_tau':raw[prefix+'tau']}
                if raw[prefix+'tau_label_valid'].all():forecasts['true_tau']=raw[prefix+'tau_true']
                for forecast,tau in forecasts.items():
                    pd=decode_target(model(state,torch.as_tensor(tau,device='cuda')),q).cpu().numpy()
                    delta=pd-target
                    output[panel][model_name+'_'+forecast]=dict(
                        xyz_p95_mm=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000),
                        rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)))
    return dict(panels=output,scope='same measured state/input; R/forecast swap only; no new policy rollout',
        paired_checkpoint_tensor_hashes_pass=True,original_R_normalization_bytes_preserved=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)


if __name__=='__main__':main()
