"""Versioned failure prebuffer admission for immutable native rollout windows."""
import numpy as np

SCHEMA='ref2dex.rollout-position-prebuffer.v1'
DEFAULT_POLICY=dict(schema=SCHEMA,max_object_error_m=.1,preceding_frames=50,history=4,horizon=24)


def admissible_ticks(data,ticks,policy):
    if policy.get('schema')!=SCHEMA:raise ValueError('explicit position-prebuffer schema required')
    limit=policy.get('max_object_error_m');before=policy.get('preceding_frames')
    history=policy.get('history');horizon=policy.get('horizon')
    if (not isinstance(before,int) or before<0 or not isinstance(history,int) or history<1
            or not isinstance(horizon,int) or horizon<1 or not np.isfinite(limit) or limit<=0):
        raise ValueError('invalid position-prebuffer policy')
    error=np.linalg.norm(data['obj'][:,:3,3]-data['reference_obj'][:,:3,3],axis=-1)
    failures=np.flatnonzero(~np.isfinite(error)|(error>limit))
    # Inclusive [bad-before,bad], merging all failure intervals without mutation.
    difference=np.zeros(len(error)+1,np.int64)
    np.add.at(difference,np.maximum(failures-before,0),1)
    np.add.at(difference,failures+1,-1)
    unsafe=np.cumsum(difference[:-1])>0
    ticks=np.asarray(ticks,dtype=np.int64)
    if len(ticks) and ((ticks<0).any() or (ticks+horizon>=len(error)).any()):
        raise ValueError('incomplete history/future window')
    prefix=np.r_[0,np.cumsum(unsafe)]
    low=np.maximum(ticks-history+1,0);high=ticks+horizon+1
    keep=prefix[high]-prefix[low]==0
    return keep,dict(frames=len(error),failure_frames=len(failures),unsafe_frames=int(unsafe.sum()),
        input_windows=len(ticks),retained_windows=int(keep.sum()),excluded_windows=int((~keep).sum()))


def restrict_rollout_admission(dataset,policy):
    keys=[];ticks_by_sequence=[];pools=[];records=[]
    for seq,ticks in enumerate(dataset.ticks_by_sequence):
        data,_=dataset.sequence(seq);keep,report=admissible_ticks(data,ticks,policy)
        admitted=ticks[keep];begin=len(keys);keys.extend((int(t),seq) for t in admitted)
        ticks_by_sequence.append(admitted);pools.append(np.arange(begin,len(keys),dtype=np.int64))
        records.append(dict(sequence=dataset.records[seq]['sequence'],split=dataset.split,
            admitted_ticks=admitted.tolist(),excluded_ticks=ticks[~keep].tolist(),**report))
    if not keys:raise ValueError('no windows survive strict rollout admission')
    phases=[]
    for startup in (True,False):
        phase=[p[(t<16) if startup else (t>=16)] for p,t in zip(pools,ticks_by_sequence)]
        phases.append([p for p in phase if len(p)] or [p for p in pools if len(p)])
    before=len(dataset.keys);dataset.keys=keys;dataset.ticks_by_sequence=ticks_by_sequence
    dataset.pools=pools;dataset.phase_pools=phases
    return dict(policy=dict(policy),split=dataset.split,original_windows=before,retained_windows=len(keys),
                excluded_windows=before-len(keys),records=records,
                semantics='exclude any H4/current/future24 support touching a failure or its50 preceding frames')
