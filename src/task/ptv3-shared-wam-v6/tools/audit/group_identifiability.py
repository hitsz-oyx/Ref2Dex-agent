"""Strict complete-link TRAIN condition groups; pooled unbiased within-group variance.

Complements pair dispersion. Future labels never determine baseline grouping.
One native recording appears at most once per gate. Singleton partitions do not
pretend to provide zero variance: they contribute zero degrees of freedom.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

NAMES=('hand_prefix8','hand_all24','object_prefix8','object_all24')


def choose_groups(samples,z,gate,timing=False):
    by=defaultdict(list)
    for i,s in enumerate(samples):by[(s['object'],tuple(s['presence']),s['event_type'])].append(i)
    groups=[];used=set()
    for ids in by.values():
        if len(ids)<4:continue
        hand=z['hand_history'][ids][:,:,z['present'][ids[0]]].reshape(len(ids),-1).astype(np.float64)
        obj=z['object_history'][ids].reshape(len(ids),-1).astype(np.float64);goal=z['goal'][ids].reshape(len(ids),-1).astype(np.float64)
        h=cdist(hand,hand)/np.sqrt(hand.shape[1]/3);o=cdist(obj,obj)/np.sqrt(obj.shape[1]/3);g=cdist(goal,goal)/np.sqrt(32)
        distance=np.maximum(np.maximum(h,o),g);allowed=distance<=gate
        if timing:
            t=np.array([samples[i]['event_delta'] for i in ids]);allowed&=abs(t[:,None]-t[None])<=3
        for anchor in np.argsort(-allowed.sum(1),kind='stable'):
            if samples[ids[anchor]]['capture'] in used:continue
            picked=[anchor];caps={samples[ids[anchor]]['capture']}
            for j in np.argsort(distance[anchor],kind='stable'):
                cap=samples[ids[j]]['capture']
                if cap in used or cap in caps or not allowed[j,picked].all():continue
                picked.append(j);caps.add(cap)
                if len(picked)>=8:break
            if len(picked)>=4:groups.append([ids[j] for j in picked]);used.update(caps)
    return groups


def statistics(groups,samples,z,permutation=None):
    totals=np.zeros((len(groups),5,6),np.float64)
    for gi,group in enumerate(groups):
        partitions=[defaultdict(list) for _ in range(5)]
        for i in group:
            s=samples[permutation[i] if permutation is not None else i]
            keys=[(),(s['contact_available'],),(s['contact_available'],s['region']),
                (s['contact_available'],s['region'],s['finger']),
                (s['contact_available'],s['region'],s['finger'],s['interaction_type'])]
            for j,k in enumerate(keys):partitions[j][k].append(i)
        present=z['present'][group[0]]
        for stage,parts in enumerate(partitions):
            for ids in parts.values():
                if len(ids)<2:continue
                hand=z['hand'][ids][:,:,present];scene=z['scene'][ids]
                h=(hand-hand.mean(0)).astype(np.float64);s=(scene-scene.mean(0)).astype(np.float64)
                sums=[(h[:,:8]**2).sum(-1).mean((1,2,3)).sum(),(h**2).sum(-1).mean((1,2,3)).sum(),
                    (s[:,:8]**2).sum(-1).mean((1,2)).sum(),(s**2).sum(-1).mean((1,2)).sum()]
                totals[gi,stage,:4]+=np.asarray(sums)*1e6;totals[gi,stage,4]+=len(ids)-1;totals[gi,stage,5]+=len(ids)
    return totals


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    samples=json.loads((out/'bank.json').read_text())['samples']
    with np.load(out/'bank.npz') as archive:z={k:archive[k] for k in archive.files}
    rng=np.random.default_rng(180);report={}
    names=('base','plus_future_contact_presence','plus_object_region','plus_finger','plus_interaction')
    for gate in (.02,.05,.1):
        for timing in (False,True):
            groups=choose_groups(samples,z,gate,timing);tag=str(round(gate*1000))+('_time3' if timing else '')
            if not groups:report[tag]=dict(groups=0,status='INSUFFICIENT_GROUP_SUPPORT');continue
            sums=statistics(groups,samples,z);pool=sums.sum(0);variance=pool[:,:4]/np.maximum(pool[:,4,None],1)
            boot=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=200)
            bs=np.einsum('bg,gsc->bsc',boot,sums);bv=bs[:,:,:4]/np.maximum(bs[:,:,4,None],1)
            perm=[]
            for repeat in range(100):
                shuffle=np.arange(len(samples))
                for ids in groups:shuffle[ids]=rng.permutation(ids)
                ps=statistics(groups,samples,z,shuffle).sum(0);pv=ps[:,:4]/np.maximum(ps[:,4,None],1);pv[ps[:,4]==0]=np.nan;perm.append(pv)
            perm=np.asarray(perm);stages={}
            for j,name in enumerate(names):
                stages[name]=dict(windows_with_variance=int(pool[j,5]),degrees_of_freedom=int(pool[j,4]),variance_mm2=dict(zip(NAMES,variance[j].tolist())) if pool[j,4] else None,
                    permutation_variance95_mm2=np.nanpercentile(perm[:,j],[2.5,97.5],axis=0).T.tolist() if np.isfinite(perm[:,j]).any() else None)
                if j and pool[j,4]:
                    good=(bs[:,j,4]>0)&(bs[:,j-1,4]>0)
                    drop=100*(1-bv[good,j]/np.maximum(bv[good,j-1],1e-12))
                    stages[name]['drop_vs_previous_percent']=(100*(1-variance[j]/np.maximum(variance[j-1],1e-12))).tolist()
                    stages[name]['drop95_group_bootstrap']=np.percentile(drop,[2.5,97.5],axis=0).T.tolist() if len(groups)>=8 else None
            report[tag]=dict(status='LOW_GROUP_SUPPORT' if len(groups)<8 else 'DESCRIPTIVE',groups=len(groups),windows=sum(map(len,groups)),captures=sum(map(len,groups)),group_indices=groups,stages=stages)
    result=dict(status='COMPLETED',gate_groups=report,metrics=NAMES,seed=180,
        estimator='pooled within-group trace variance per mean joint/point/time,mm²;sum residual squared/(sum n_group-1);exclude singleton df',
        matching='same CAD,presence,Etype;complete-link H4 hand,object64 andgoal32 RMS gates;4..8 distinct captures/group;one capture used once/gate;no future target-based group selection',
        caveats=['finite-radius groups do not certify exact-input irreducibility','degrees of freedom and support fall as labels refine;permutations within original groups preserve label-count effects','oracle realized future-contact labels are not causal model inputs','time3 control also matches v6 event timing;100mm groups are weak-condition sensitivity'])
    (out/'group-identifiability.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n')
    print({k:v['groups'] for k,v in report.items()})


if __name__=='__main__':main()
