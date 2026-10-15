"""Physical motion diagnostics for recorded simulator trajectories, in SI units."""
import numpy as np


def motion_features(time, objects, hand):
    time=np.asarray(time,np.float64)
    objects=np.asarray(objects,np.float64)
    hand=np.asarray(hand,np.float64)
    if (time.ndim!=1 or len(time)<4 or objects.shape!=(len(time),7)
            or hand.shape!=(len(time),11,3) or not np.isfinite(time).all()
            or not np.isfinite(objects).all() or not np.isfinite(hand).all()
            or not np.allclose(np.diff(time),1/30,atol=1e-6,rtol=0)):
        raise ValueError('finite recorded30Hz simulator time/xyzquat/11-hand required')
    q=objects[:,3:]
    if not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-4):
        raise ValueError('unit quaternion required')
    dt=np.diff(time)
    velocity=np.diff(objects[:,:3],axis=0)/dt[:,None]
    features={
        'step_m':np.r_[0.,np.linalg.norm(np.diff(objects[:,:3],axis=0),axis=1)],
        'speed_mps':np.r_[0.,np.linalg.norm(velocity,axis=1)],
        'acceleration_mps2':np.r_[0.,0.,np.linalg.norm(np.diff(velocity,axis=0)/((dt[:-1]+dt[1:])/2)[:,None],axis=1)],
        'angular_speed_radps':np.r_[0.,2*np.arccos(np.clip(abs((q[1:]*q[:-1]).sum(1)),0,1))/dt],
        'hand_speed_mps':np.r_[0.,np.linalg.norm(np.diff(hand,axis=0)/dt[:,None,None],axis=2).max(1)],
        'hand_object_distance_m':np.linalg.norm(hand-objects[:,None,:3],axis=2).min(1)}
    features['distance_step_m']=np.r_[0.,np.diff(features['hand_object_distance_m'])]
    return features


def validate_policy(policy):
    if policy.get('schema') != 'ref2dex.unidex-motion-policy.v1':
        raise ValueError('versioned simulator motion policy required')
    positive = ('min_step_m','speed_mps','acceleration_mps2','min_object_hand_speed_ratio',
                'severe_step_m','severe_speed_mps','initial_speed_mps',
                'extreme_speed_mps','angular_speed_radps')
    if any(not np.isfinite(policy[k]) or policy[k] <= 0 for k in positive):
        raise ValueError('finite positive SI thresholds required')
    if type(policy['initial_transitions']) is not int or not 1 <= policy['initial_transitions'] <= 2:
        raise ValueError('initialization check is limited to first two transitions')


def events(features,policy):
    """Return transition-end frame IDs; a policy is recorded, never inferred from success."""
    validate_policy(policy)
    speed=features['speed_mps'];step=features['step_m'];acc=features['acceleration_mps2']
    omega=features['angular_speed_radps']
    hand=np.maximum(features['hand_speed_mps'],np.r_[0.,features['hand_speed_mps'][:-1]])
    outruns_hand=speed>policy['min_object_hand_speed_ratio']*hand
    initial=(np.arange(len(speed))>0) & (np.arange(len(speed))<=policy['initial_transitions'])
    return dict(
        translational_impulse=np.flatnonzero((step>policy['min_step_m']) & (speed>policy['speed_mps']) & (acc>policy['acceleration_mps2']) & outruns_hand),
        severe_shared_impulse=np.flatnonzero((step>policy['severe_step_m']) & (speed>policy['severe_speed_mps']) & (acc>policy['acceleration_mps2'])),
        initial_fast_flight=np.flatnonzero(initial & (step>policy['min_step_m']) & (speed>policy['initial_speed_mps']) & outruns_hand),
        extreme_translation=np.flatnonzero(speed>policy['extreme_speed_mps']),
        extreme_rotation=np.flatnonzero(omega>policy['angular_speed_radps']))


def rejected_ticks(ticks,event_frames):
    """Keep the prefix before first suspect transition; never splice/recover an episode tail.

    Full H4+K24 must end before the transition.  Removal is a recorded-quality
    decision, independent of grasp success or a window's net future motion.
    """
    ticks=np.asarray(ticks,np.int64)
    if not len(event_frames):return np.zeros(len(ticks),bool)
    onset=int(np.min(event_frames))
    return ticks+24>=onset
