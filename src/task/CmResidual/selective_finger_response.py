"""Actual independent finger target coordinates with native child coupling."""
import torch

INDEPENDENT = (6,8,10,12,14,15)
COUPLING = {6:((7,1.05),),8:((9,1.05),),10:((11,1.05),),12:((13,1.05),),15:((16,.6),(17,.8))}
PRIMITIVES = ((0.,0.,0.,0.,0.,0.),(.15,.15,.15,.15,0.,.15),
    (.15,0.,0.,0.,0.,0.),(0.,.15,0.,0.,0.,0.),(0.,0.,.15,0.,0.,0.),
    (0.,0.,0.,.15,0.,0.),(0.,0.,0.,0.,0.,.15),(0.,0.,0.,0.,.15,0.))


def primitive_target(base,parameters,lower,upper):
    if base.shape[-1] != 18 or parameters.shape != (len(base),6) or not torch.isfinite(parameters).all():
        raise ValueError('six independent finite finger target coordinates')
    goal = base.clone()
    for column,parent in enumerate(INDEPENDENT):
        children = COUPLING.get(parent, ())
        lo = max(float(lower[parent]),*(float(lower[j])/ratio for j,ratio in children)) if children else float(lower[parent])
        hi = min(float(upper[parent]),*(float(upper[j])/ratio for j,ratio in children)) if children else float(upper[parent])
        if lo > hi:
            raise ValueError('incompatible native dependent bounds')
        goal[:,parent] = (base[:,parent]+parameters[:,column]).clamp(lo,hi)
        for child,ratio in children:
            goal[:,child] = goal[:,parent]*ratio
    if not torch.equal(goal[:,:6],base[:,:6]):
        raise ValueError('finger intervention altered wrist target')
    if ((goal[:,6:] < lower[6:]-1e-6) | (goal[:,6:] > upper[6:]+1e-6)).any():
        raise ValueError('finger target outside native bounds')
    return goal


def validate_dof_names(names):
    if len(names) != 18 or names[:6] != ['joint'+str(i) for i in range(1,7)]:
        raise ValueError('native wrist DOF mapping')
    fingers = []
    for parent,child in ((6,7),(8,9),(10,11),(12,13)):
        name = names[parent]
        if not name.endswith('_proximal_joint'):
            raise ValueError('native independent curl name')
        finger = name[:-len('_proximal_joint')]
        if names[child] != finger+'_intermediate_joint':
            raise ValueError('native dependent curl name')
        fingers.append(finger)
    if set(fingers) != {'index','middle','ring','pinky'}:
        raise ValueError('native independent finger coverage')
    if names[14:] != ['thumb_proximal_yaw_joint','thumb_proximal_pitch_joint','thumb_intermediate_joint','thumb_distal_joint']:
        raise ValueError('native thumb coordinate mapping')
