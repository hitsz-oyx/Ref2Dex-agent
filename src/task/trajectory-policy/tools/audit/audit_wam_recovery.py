"""Independent NumPy audit of live expert PD labels and the takeover boundary."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_wam_ref2_shadow import sha, write

ROOT=Path(__file__).resolve().parents[5]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true',help='read-only recheck; preserve prior audit artifact')
    a=parser.parse_args(); out=a.native.resolve()
    if ROOT/'outputs/trajectory-policy' not in out.parents:
        raise ValueError('task-owned trace required')
    destination=out/'root_recovery_audit.json'
    if destination.exists() and not a.check_only:
        raise FileExistsError(destination)
    manifest=json.loads((out/'manifest.json').read_text())
    result=json.loads((out/'result.json').read_text())
    controls=manifest.get('controls',542);takeover=manifest.get('takeover_tick')
    if (manifest['status']!='COMPLETED' or not isinstance(controls,int) or not 1<=controls<=542
            or not isinstance(takeover,int) or not 0<=takeover<controls):
        raise ValueError('completed bounded recovery required')
    with np.load(out/'trajectory.npz',allow_pickle=False) as f:
        d={k:f[k].copy() for k in f.files}
    checks={}
    def check(name,x,y):
        if x.shape!=y.shape:
            raise ValueError(name+' shape mismatch')
        checks[name]=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))
    if len(d['action'])!=controls or len(d['q'])!=controls+1:
        raise ValueError('declared control/state rollout required')
    if controls<542 and (result.get('partial_episode') is not True
            or result.get('calibration_pass') is not None or result.get('recovery_pass') is not None
            or any('success' in x or 'controlled_final_place' in x for x in result['outcomes'])):
        raise ValueError('partial collection cannot carry placement or full recovery verdict')
    offset=np.asarray(manifest['native_controller']['offset'],np.float32)
    scale=np.asarray(manifest['native_controller']['scale'],np.float32)
    def pd(action,q):
        normalized=action.copy(); normalized[...,6:]=(1+normalized[...,6:])/2
        value=offset+scale*normalized
        value[...,:6]=value[...,:6]+q[...,:6]
        for distal,parent,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):
            value[...,distal]=value[...,parent]*ratio
        return value
    check('actual_applied_action',d['action'],d['applied'])
    check('actual_native_PD_inverse',d['pd_targets'],pd(d['applied'],d['q'][:-1]))
    check('expert_action_clamp',d['expert_action'],np.clip(d['expert_raw'],-1,1))
    check('expert_clipping_mask',d['expert_clipped'],np.max(np.abs(d['expert_raw']-d['expert_action']),axis=-1)>1e-6)
    check('expert_live_q_PD',d['expert_PD'],pd(d['expert_action'],d['q'][:-1,2:]))
    check('takeover_applied_expert_action',d['applied'][takeover:,3],d['expert_action'][takeover:,1])
    check('takeover_actual_expert_PD',d['pd_targets'][takeover:,3],d['expert_PD'][takeover:,1])
    # Independent PhysX rows need physical comparability, not hidden-state bit equality.
    pair_errors={k:float(np.max(np.abs(d[k][:takeover+1,2].astype(float)-d[k][:takeover+1,3].astype(float))))
        for k in ('q','dq','hand','obj','velocity','surface_gap','support_gap','table_footprint')}
    qdelta=d['q'][:takeover+1,2]-d['q'][:takeover+1,3]
    handdelta=d['hand'][:takeover+1,2]-d['hand'][:takeover+1,3]
    odelta=d['obj'][:takeover+1,2]-d['obj'][:takeover+1,3]
    physical=dict(q_XYZ_norm_m=float(np.linalg.norm(qdelta[:,:3],axis=-1).max()),
        q_other_max_rad=float(np.abs(qdelta[:,3:]).max()),
        hand_point_norm_m=float(np.linalg.norm(handdelta,axis=-1).max()),
        object_XYZ_norm_m=float(np.linalg.norm(odelta[:,:3,3],axis=-1).max()),
        rotation_matrix_frobenius_over_sqrt2=float(np.linalg.norm(odelta[:,:3,:3],axis=(1,2)).max()/np.sqrt(2)))
    limits=dict(q_XYZ_norm_m=1e-4,q_other_max_rad=1e-3,hand_point_norm_m=1e-4,
        object_XYZ_norm_m=1e-4,rotation_matrix_frobenius_over_sqrt2=1e-3)
    own_row=manifest.get('own_row_recovery',False)
    pair_pass=all(physical[k]<=v for k,v in limits.items())
    if own_row:
        if manifest.get('policy_query_role')!=3:raise ValueError('own-row query identity')
        # A same-row takeover needs no equality to the unrelated replay row.
        pair_pass=True
        check('own_row_R_state',d['state_features'][:,1],d['learner_state'][:,1])
        past=np.maximum(np.arange(controls)[:,None]+np.arange(-3,1),0)
        check('own_row_actor_hand',d['wam_hand_history'],d['hand'][past,3])
        check('own_row_actor_object',d['wam_object_history'],d['obj'][past,3])
        expected=d['encoded_target'][:takeover,1,0].copy()
        active=[0,1,2,3,4,5,6,8,10,12,14,15]
        shift=np.asarray(manifest['prefix_exploration']['xyz_mm'])/1000
        expected[24:min(32,takeover),:3]+=shift/scale[:3]
        check('own_row_actor_actually_applied',d['applied'][:takeover,3][:,active],expected.clip(-1,1))
        if 'control_proprio' in d:
            prev=np.concatenate((np.zeros((1,18),np.float32),d['applied'][:-1,3]),0)
            check('own_row_proprio',d['control_proprio'],np.concatenate((d['q'][:-1,3],d['dq'][:-1,3],prev[:,active]),-1))
    if takeover:
        for key in ('applied',):
            check('paired_prefix_'+key,d[key][:takeover,2],d[key][:takeover,3])
    with np.load(ROOT/'outputs/trajectory-policy/competent-reference-20261011-r1/reference.npz',allow_pickle=False) as f:
        reference=f['obj'].copy()
    for tick in range(controls):
        past=np.maximum(tick+np.arange(-3,1),0)
        for role in range(2):
            check('H4_hand_'+str(tick)+'_'+str(role),d['learner_hand_history'][tick,role],d['hand'][past,role+2])
            check('H4_object_'+str(tick)+'_'+str(role),d['learner_object_history'][tick,role],d['obj'][past,role+2])
        check('planned_goal_'+str(tick),d['learner_goal'][tick],np.repeat(reference[min(tick+24,542)][None],2,axis=0))
    q,dq,hand,obj,vel=[d[k][:-1,2:] for k in ('q','dq','hand','obj','velocity')]
    local=np.einsum('...pj,...jk->...pk',hand-obj[...,:3,3][...,None,:],obj[...,:3,:3])
    state=np.concatenate((q,dq,local.reshape(controls,2,33),obj[...,:3,:].reshape(controls,2,12),vel),-1).astype(np.float32)
    check('current_R_state',d['learner_state'],state)
    check('expert_applied_mask',d['expert_applied'],np.c_[np.zeros(controls,bool),np.arange(controls)>=takeover])
    check('tau_valid_mask',d['learner_wam_tau_valid'],np.c_[np.zeros(controls,bool),np.ones(controls,bool)] if own_row else np.c_[np.ones(controls,bool),np.zeros(controls,bool)])
    check('no_command_replay_tau_label',d['learner_wam_tau'][:,0 if own_row else 1],np.zeros_like(d['learner_wam_tau'][:,0 if own_row else 1]))
    # The durable pre-physics query records must equal their measured full trace.
    for tick in range(controls):
        path=out/'expert_queries'/('%04d.npz'%tick)
        if sha(path)!=manifest['expert_query_sha256'][str(path.relative_to(out))]:
            raise ValueError('durable query source hash mismatch')
        with np.load(path,allow_pickle=False) as f:
            check('durable_q_'+str(tick),f['current_q'],d['q'][tick,2:])
            for key in f.files:
                if key not in ('tick','current_q'):
                    check('durable_'+key+'_'+str(tick),f[key],d[key][tick])
    if sha(out/'trajectory.npz')!=manifest['trajectory_sha256']:
        raise ValueError('full source trajectory hash mismatch')
    finite=all(np.isfinite(v).all() for v in d.values())
    valid=(finite and pair_pass and max(checks.values())<=1e-6 and manifest.get('training_allowed') is True and
        manifest.get('expert_PD_training_allowed') is True and manifest.get('mixed_policy_future_hand_training_allowed') is False)
    summary=dict(pass_=bool(valid),all_arrays_finite=bool(finite),checks_max=max(checks.values()),
        nonzero_checks={k:v for k,v in checks.items() if v!=0},check_count=len(checks),
        native_data_sha256=sha(out/'trajectory.npz'),
        recovery_pass=None if result.get('recovery_pass') is None else bool(result['recovery_pass']),
        partial_episode=controls<542,takeover_tick=takeover,local_grasp_calibration=result.get('local_grasp_calibration'),
        controls=controls,measured_states=controls+1,prefix_pair_physical=physical,prefix_pair_limits=limits,
        prefix_pair_all_raw_differences=pair_errors,approximate_pair_not_hidden_state_fork=not own_row,own_row_takeover=own_row,
        training_scope='live-state expert PD only; no mixed-policy tau labels')
    summary['pass']=summary.pop('pass_')
    if not a.check_only:
        write(destination,summary)
    print(json.dumps(summary,indent=2))
    if not valid:
        raise SystemExit(1)


if __name__=='__main__':
    main()
