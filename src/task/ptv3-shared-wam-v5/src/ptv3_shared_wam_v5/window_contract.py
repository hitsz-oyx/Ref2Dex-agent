"""Shared H4 admission predicate for freezing and reading joint21 windows."""
import numpy as np


def history_joint_valid(joint_valid, tick):
    if tick < 3 or tick >= len(joint_valid):
        raise ValueError('H4 history outside sequence')
    return np.asarray(joint_valid[tick-3:tick+1], dtype=bool).all(0)


def history_admitted(joint_valid, ticks):
    ticks=np.asarray(ticks,np.int64)
    if ((ticks<3)|(ticks>=len(joint_valid))).any():raise ValueError('H4 history outside sequence')
    # Bounded temporary: four per-joint intersections, not N*H4 geometry.
    result=np.empty(len(ticks),bool)
    for start in range(0,len(ticks),4096):
        chosen=ticks[start:start+4096]
        shared=np.asarray(joint_valid[chosen],bool).copy()
        for lag in (1,2,3):shared &= joint_valid[chosen-lag]
        result[start:start+4096]=shared.any((1,2))
    return result
