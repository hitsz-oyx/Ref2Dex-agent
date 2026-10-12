"""Audit multi-takeover fit labels and metrics against each actual source."""
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
    aggregation=Path(manifest['aggregation_directory'])
    am=json.loads((aggregation/'manifest.json').read_text())
    for sid,entry in enumerate(am['admissions']):
        paths['multi%d'%sid]=(Path(entry['native'])/'trajectory.npz').resolve()
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
    a=p.parse_args()
    directory=a.fit.resolve()
    if ROOT/'outputs/trajectory-policy' not in directory.parents:raise ValueError('task-owned fit required')
    report=audit(directory)
    write(directory/'root_joint_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
    if not report['pass']:raise SystemExit(1)



if __name__=='__main__':main()
