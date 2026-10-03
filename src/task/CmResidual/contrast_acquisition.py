"""Block-preserving offline acquisition; scores never consume pool outcomes."""
import numpy as np
import torch

NAMES=('contrast','absolute','uniform')
QUOTAS=(43,43,43,43,42,42)


def blocks(rows, seed_key, actor_key):
    groups={}
    for i,r in enumerate(rows):groups.setdefault((r[seed_key],r['environment']),[]).append(i)
    keys=sorted(groups);indices=[];strata=[]
    for k in keys:
        ids=sorted(groups[k],key=lambda i:rows[i][actor_key])
        assert len(ids)==2 and len({rows[i][actor_key] for i in ids})==2
        assert rows[ids[0]]['motion']==rows[ids[1]]['motion']
        indices.append(ids);strata.append((k[0],rows[ids[0]]['motion']))
    return np.array(indices),strata


def initial_blocks(strata):
    rng=np.random.default_rng(4500);selected=[]
    for s,q in zip(sorted(set(strata)),QUOTAS):
        candidates=np.array([i for i,t in enumerate(strata) if t==s]);assert len(candidates)==256
        selected.extend(rng.choice(candidates,q,replace=False))
    return np.sort(selected)


def acquire(strata, initial, scores):
    result={};rng=np.random.default_rng(4530)
    for name in NAMES:
        selected=[]
        for s,q in zip(sorted(set(strata)),QUOTAS):
            candidates=np.array([i for i,t in enumerate(strata) if t==s and i not in initial])
            if name=='uniform':chosen=rng.choice(candidates,q,replace=False)
            else:chosen=candidates[np.lexsort((candidates,-scores[name][candidates]))[:q]]
            selected.extend(chosen)
        result[name]=np.sort(selected)
    return result


def scores(predictions):
    # [model,row,arm,xyz], metres. Common per-model offsets cancel.
    contrast=predictions[:,:,1::2]-predictions[:,:,2::2]
    return dict(contrast=contrast.var(0).sum((1,2)),absolute=predictions.var(0).sum((1,2)))


def initialized(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        net=torch.nn.Sequential(torch.nn.Linear(87,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,3))
    return net


def metrics(predictions,pseudo,rows):
    columns=('uniform','absolute','zero');d=predictions['contrast'].astype(np.float64)
    risk=np.stack([((d*d-predictions[c]**2)-2*(d-predictions[c])*pseudo).sum((1,2))*1e6 for c in columns],-1)
    idx,strata=blocks(rows,'evaluation_seed','training_seed');rng=np.random.default_rng(4550);boot=np.zeros((2000,3))
    for s in sorted(set(strata)):
        ids=idx[np.array([t==s for t in strata])];draw=rng.integers(len(ids),size=(2000,len(ids)))
        boot+=risk[ids][draw].sum((1,2))/len(rows)
    points=risk.mean(0);upper=np.quantile(boot,.95,axis=0)
    each={str(s):float(risk[[i for i,r in enumerate(rows) if r['evaluation_seed']==s],0].mean()) for s in sorted({r['evaluation_seed'] for r in rows})}
    gates=dict(uniform_margin=bool(points[0]<=-1),uniform_upper=bool(upper[0]<0),absolute_margin=bool(points[1]<=-.5),absolute_upper=bool(upper[1]<0),both_test_seeds=all(v<0 for v in each.values()),better_zero=bool(points[2]<0))
    report=dict(run_status='COMPLETED',label='PROMISING' if all(gates.values()) else 'UNPROMISING',gates=gates,
                relative_risk_mm2={c:dict(point=float(points[j]),upper95=float(upper[j]),central95=np.quantile(boot[:,j],[.025,.975]).tolist()) for j,c in enumerate(columns)},test_seed_uniform_points=each)
    return report,risk,boot
