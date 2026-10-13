"""Independent own-state labels, first PD metrics, and GT native execution audit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'tools/audit'),str(TASK/'src'),str(TASK/'tools/run')]
from audit_competent_pd import independent_features,independent_pd,ACTIVE
from audit_wam_ref2_shadow import write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--execution',type=Path,required=True)
    a=p.parse_args();fm=json.loads((a.fit/'manifest.json').read_text());fr=json.loads((a.fit/'result.json').read_text())
    def read(path):
        with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
    source=read(Path(fm['source'])/'trajectory.npz');controller=json.loads((Path(fm['source'])/'manifest.json').read_text())['native_controller']
    if sha(Path(fm['source'])/'trajectory.npz')!=fm['source_sha256']:raise ValueError('nominal source drift')
    truth={};maxima={};parts={k:[] for k in ('exact_train','contact_train','exact_dev','contact_dev')}
    def rows(d,ticks,roles):
        future=d['hand'][ticks[:,None]+np.arange(1,25),roles[:,None]]
        state,tau=independent_features(d,ticks,roles,future)
        return state,tau,d['applied'][ticks[:,None]+np.arange(8),roles[:,None]][...,ACTIVE]
    ticks=np.repeat(np.arange(519),8);roles=np.tile(np.arange(16,24),519)
    truth['nominal_dev']=rows(source,ticks,roles)
    truth['startup_dev']=tuple(v[:128] for v in truth['nominal_dev'])
    for name,q in fm['queries'].items():
        path=Path(q['native']);d=read(path/'trajectory.npz');takeover=q['takeover']
        if sha(path/'trajectory.npz')!=q['trajectory_sha256']:raise ValueError('source drift')
        if (q['family']==3)!=(q['split']=='dev'):raise ValueError('whole-prefix split violation')
        if not np.array_equal(d['applied'][takeover:,3],d['expert_action'][takeover:,1]):raise ValueError('expert not applied')
        valid=np.array([t for t in range(takeover,105) if d['expert_applied'][t:t+24,1].all() and not d['expert_clipped'][t:t+24,1].any()])
        if len(valid)!=q['valid_windows']:raise ValueError('valid window count')
        for prefix,ts in [('exact',valid[valid==takeover]),('contact',valid[valid<takeover+16])]:
            parts[prefix+'_'+q['split']].append(rows(d,ts,np.full(len(ts),3)))
    truth.update({k:tuple(np.concatenate([v[j] for v in vs]) for j in range(3)) for k,vs in parts.items()})
    inputs=read(a.fit/'panel_inputs.npz');pred=read(a.fit/'selected.npz')
    for name,(state,tau,true) in truth.items():
        for field,v in [('state',state),('tau',tau),('true',true)]:maxima[name+'_'+field]=float(np.abs(inputs[name+'_'+field]-v).max())
        maxima[name+'_label']=float(np.abs(pred[name+'_true']-true).max())
        def full(x):
            y=np.zeros((len(x),18),np.float32);y[:,ACTIVE]=x;return y
        q=np.zeros((len(true),18),np.float32)
        delta=independent_pd(full(pred[name+'_prediction'][:,0].clip(-1,1)),q,controller)-independent_pd(full(true[:,0]),q,controller)
        error=np.linalg.norm(delta[:,:3],axis=-1)*1000
        maxima[name+'_PD_mm']=abs(float(np.percentile(error,95))-fr['reports'][name]['xyz_p95_mm'])
    curve=json.loads((a.fit/'curve.json').read_text())['curve']
    selected=min(curve,key=lambda r:(not all(r['checks'].values()),r['score']))
    if selected['step']!=fr['selected_update']:raise ValueError('selection differs')
    m=json.loads((a.execution/'manifest.json').read_text());r=json.loads((a.execution/'result.json').read_text());trace=read(a.execution/'trajectory.npz')
    if m['status']!='COMPLETED' or sha(a.fit/'chunk.pt')!=m['checkpoint_sha256']:raise ValueError('qualification identity')
    maxima['actual_command']=float(np.abs(trace['action']-trace['applied']).max())
    maxima['native_PD']=float(np.abs(independent_pd(trace['applied'],trace['q'][:-1],m['native_controller'])-trace['pd_targets']).max())
    maxima['first_only']=float(np.abs(trace['applied'][:,2,ACTIVE]-trace['chunk_action'][:,0].clip(-1,1)).max())
    ticks=np.repeat(np.arange(128),2);envs=np.tile([1,2],128)
    future=np.repeat(source['hand'][np.arange(128)[:,None]+np.arange(1,25),24],2,axis=0)
    state,tau=independent_features(trace,ticks,envs,future)
    maxima['own_state']=float(np.abs(state-trace['R_state'].reshape(-1,87)).max())
    maxima['own_tau_frame']=float(np.abs(tau-trace['R_tau'].reshape(-1,24,33)).max())
    from execute_wam_native_bridge import partial_grasp_outcome
    for env in range(3):
        outcome=partial_grasp_outcome(dict(object_pose=trace['obj'][:,env],object_velocity=trace['velocity'][:,env],surface_gap=trace['surface_gap'][:,env],support_gap=trace['support_gap'][:,env],table_footprint=trace['table_footprint'][:,env]))
        if any(outcome[k]!=r['outcomes'][env][k] for k in outcome):raise ValueError('physical outcome mismatch')
    passed=all(v<(.0001 if k.endswith('_PD_mm') else 1e-6) for k,v in maxima.items())
    write(a.fit/'root_repair_audit.json',dict(pass_audit=passed,maxima=maxima,exact_visited_queries=len(fm['queries']),physical_qualification=r['GT_execution_gate'],selected_update=selected['step']))
    if not passed:raise ValueError('audit tolerance')
    write(a.fit/'qualification.json',dict(directory=str(a.execution.resolve()),scope='nominal privileged GT qualification; no autonomous guarantee'))
    print(json.dumps(dict(pass_audit=passed,max_error=max(maxima.values()),qualification=r['GT_execution_gate'])))

if __name__=='__main__':main()
