"""Reconstruct actual chunk labels and first-only physical execution, without NN calls."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'tools/audit'),str(TASK/'src')]
from audit_wam_ref2_shadow import write
from audit_competent_pd import independent_pd,independent_features,ACTIVE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--execution',type=Path);a=p.parse_args();fit=a.fit.resolve()
    fm=json.loads((fit/'manifest.json').read_text());fr=json.loads((fit/'result.json').read_text())
    if fm['status']!='COMPLETED' or fr['status']!='COMPLETED':raise ValueError('completed fit required')
    source=Path(fm['source'])
    with np.load(source/'trajectory.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    am=json.loads((Path(fm['recovery_aggregation'])/'manifest.json').read_text())
    maxima={};expected={}
    for name,envs in dict(train=list(range(16)),development=list(range(16,24)),test=list(range(24,32))).items():
        ticks=np.repeat(np.arange(519),len(envs));roles=np.tile(envs,519)
        expected[name]=d['applied'][ticks[:,None]+np.arange(8),roles[:,None]][...,ACTIVE]
    rec=[]
    for entry in am['admissions']:
        with np.load(Path(entry['native'])/'trajectory.npz',allow_pickle=False) as f:
            ticks=np.arange(entry['takeover'],105)
            if (not f['expert_applied'][entry['takeover']:,1].all() or f['expert_clipped'][entry['takeover']:,1].any()):
                raise ValueError('clipped/unapplied expert source')
            rec.append(f['applied'][ticks[:,None]+np.arange(8),3][...,ACTIVE])
    expected['recovery_TRAIN']=np.concatenate(rec);expected['train']=np.concatenate([expected['train'],expected['recovery_TRAIN']])
    scale=np.asarray(json.loads((source/'manifest.json').read_text())['native_controller']['scale'])[ACTIVE].copy();scale[6:]/=2
    with np.load(fit/'predictions.npz',allow_pickle=False) as f:
        for name,true in expected.items():
            maxima[name+'_actual_action_label']=float(np.abs(f[name+'_true']-true).max())
            delta=(f[name+'_prediction'][:,0].clip(-1,1)-true[:,0])*scale
            metric=float(np.percentile(np.linalg.norm(delta[:,:3],axis=-1),95)*1000)
            maxima[name+'_first_PD_metric']=abs(metric-fr['reports'][name]['first_PD_xyz_p95_mm'])
    report=dict(schema='ref2dex.chunk-R-root-audit.v1',maxima=maxima,
                actual_eight_action_labels_verified=True,no_pure_future_used=True)
    if a.execution:
        out=a.execution.resolve();m=json.loads((out/'manifest.json').read_text());r=json.loads((out/'result.json').read_text())
        if m['status']!='COMPLETED' or r['status']!='COMPLETED':raise ValueError('completed physical trace required')
        with np.load(out/'trajectory.npz',allow_pickle=False) as f:trace={k:f[k] for k in f.files}
        if len(trace['q'])!=129 or len(trace['action'])!=128:raise ValueError('terminal state/control count')
        if not np.array_equal(trace['action'],trace['applied']):raise ValueError('actually applied commands differ')
        maxima['native_PD_conversion']=float(np.abs(independent_pd(trace['applied'],trace['q'][:-1],m['native_controller'])-trace['pd_targets']).max())
        maxima['execute_first_only']=float(np.abs(trace['applied'][:,2,ACTIVE]-trace['chunk_action'][:,0].clip(-1,1)).max())
        passive=[k for k in range(18) if k not in ACTIVE]
        if trace['applied'][:,2,passive].any():raise ValueError('passive chunk action columns nonzero')
        ticks=np.repeat(np.arange(128),2);envs=np.tile([1,2],128)
        future=np.repeat(d['hand'][np.arange(128)[:,None]+np.arange(1,25),24],2,axis=0)
        state,tau=independent_features(trace,ticks,envs,future)
        maxima['own_current_state']=float(np.abs(state-trace['R_state'].reshape(-1,87)).max())
        maxima['GT_own_current_frame_tau']=float(np.abs(tau-trace['R_tau'].reshape(-1,24,33)).max())
        report.update(replan_every_tick=True,execute_only_first=True,controls=128,measured_states=129,
                      GT_execution_gate=r['GT_execution_gate'],outcomes=r['outcomes'])
    report['pass']=all(v<=1e-6 for v in maxima.values())
    write((a.execution.resolve() if a.execution else fit)/'root_chunk_audit.json',report)
    print(json.dumps(report))
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
