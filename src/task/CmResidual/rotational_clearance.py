"""Observed plane-clearance algebra; oracle hybrids are not physical interventions."""
import numpy as np


def weighted_quantile(values,weights,quantile):
    order=np.argsort(values,kind='stable')
    cumulative=np.cumsum(weights[order],dtype=np.float64)
    index=np.searchsorted(cumulative,quantile*cumulative[-1],side='left')
    return float(values[order[min(index,len(order)-1)]])


def report(rows,mask):
    if not np.any(mask):return dict(windows=0,episodes=0,episode_flip_rate=None,weighted_rotation_p95_m=None,unresolved_windows=0)
    env=rows['env'][mask];ids,counts=np.unique(env,return_counts=True)
    rates=[rows['resolved_flip'][mask][env==i].mean() for i in ids]
    mapping={int(i):int(c) for i,c in zip(ids,counts)}
    weights=np.asarray([1/mapping[int(e)] for e in env],np.float64)
    return dict(windows=int(mask.sum()),episodes=len(ids),episode_flip_rate=float(np.mean(rates)),
                weighted_rotation_p95_m=weighted_quantile(np.abs(rows['rotation_delta_m'][mask]),weights,.95),
                unresolved_windows=int(rows['unresolved'][mask].sum()))


def classify(held):
    enough=held['windows']>=128 and held['episodes']>=32
    gates=dict(coverage=enough,rotation_changes_decision=held['episode_flip_rate'] is not None and held['episode_flip_rate']>=.10,
               rotation_magnitude=held['weighted_rotation_p95_m'] is not None and held['weighted_rotation_p95_m']>=.002)
    return gates,'UNCLEAR' if not enough else ('PROMISING' if all(gates.values()) else 'UNPROMISING')
