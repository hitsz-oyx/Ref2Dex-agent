"""Training-only target association weights; all World labels remain admitted."""
import numpy as np
import torch
from torch.nn import functional as F


def target_mismatch(metrics):
    """Census columns: goal, peak object, peak/mean hand, target now/future/end,
    any-object future, target approach (all mm). This is a geometric proxy,
    not a semantic annotation of the manipulated object.
    """
    m=np.asarray(metrics)
    if m.ndim!=2 or m.shape[1]!=9 or not np.isfinite(m).all():raise ValueError('finite nine-column census required')
    g,mo,mh,_,d0,df,_,da,approach=m.T
    return (g<2)&(mo<5)&(mh>=5)&(np.minimum(d0,df)>100)&~((approach>=20)&(df<=100))&(da<=50)


def weighted_trajectory_loss(pred, batch, scale, weights):
    """Preserve baseline valid-point denominator; reduce only marked actor terms."""
    valid=batch['trajectory_valid'].bool()
    if weights.shape!=(len(pred),) or not torch.isfinite(weights).all() or ((weights<0)|(weights>1)).any():raise ValueError('one finite [0,1] actor weight per window')
    error=F.huber_loss(pred.float()/scale,batch['trajectory_target'].float()/scale,reduction='none').mean(-1)
    return (error*valid*weights[:,None,None,None]).sum()/valid.sum().clamp_min(1)
