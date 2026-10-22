"""Same-group raw/event-phase-aligned GT dispersion with condition-width controls.

Piecewise time warps preserve observed t0 and endpoint t24, aligning a known
inside-window E_next at the BASE group's median time. Group membership and label
partitions are fixed before looking at trajectory targets. No fitting.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import group_identifiability as groups


def warp_trajectory(value,event_time,anchor):
    """Map phase ticks1..24 back to original ticks; preserve both endpoints."""
    if not 0<event_time<24 or not 0<anchor<24:return value.copy()
    t=np.arange(1,25,dtype=float)
    source=np.where(t<=anchor,t*event_time/anchor,
                    event_time+(t-anchor)*(24-event_time)/(24-anchor))
    x=np.concatenate((np.zeros_like(value[:1]),value),axis=0).reshape(25,-1)
    lo=np.floor(source).astype(int).clip(0,24);hi=np.minimum(lo+1,24);f=source-lo
    return (x[lo]*(1-f[:,None])+x[hi]*f[:,None]).reshape(value.shape).astype(value.dtype)


def partitions(group,samples,stage,permutation=None):
    result=defaultdict(list)
    for i in group:
        s=samples[permutation[i] if permutation is not None else i]
        labels=(s['contact_available'],s['region'],s['finger'],s['interaction_type'])
        result[labels[:stage]].append(i)
    return [ids for ids in result.values() if len(ids)>=2]


def condition_distances(parts,z):
    distances=[]
    for ids in parts:
        present=z['present'][ids[0]]
        hh=z['hand_history'][ids][:,:,present].reshape(len(ids),-1)
        oh=z['object_history'][ids].reshape(len(ids),-1);g=z['goal'][ids].reshape(len(ids),-1)
        ij=np.triu_indices(len(ids),1)
        h=cdist(hh,hh)/np.sqrt(hh.shape[1]/3)
        o=cdist(oh,oh)/np.sqrt(oh.shape[1]/3);d=cdist(g,g)/np.sqrt(32)
        distances.extend(np.stack((h[ij],o[ij],d[ij]),-1).tolist())
    v=np.array(distances)*1000
    return dict(pairs=len(v),mean_mm=v.mean(0).tolist() if len(v) else None,
                p95_mm=np.percentile(v,95,axis=0).tolist() if len(v) else None,
                max_mm=v.max(0).tolist() if len(v) else None,
                order=['H4_hand','H4_object64','goal32'])


def pooled(parts,z):
    total=np.zeros(4);df=0;count=0
    for ids in parts:
        n=len(ids);h=z['hand'][ids][:,:,z['present'][ids[0]]];o=z['scene'][ids]
        h=(h-h.mean(0)).astype(np.float64);o=(o-o.mean(0)).astype(np.float64)
        total+=np.array([(h[:,:8]**2).sum(-1).mean((1,2,3)).sum(),(h**2).sum(-1).mean((1,2,3)).sum(),
                         (o[:,:8]**2).sum(-1).mean((1,2)).sum(),(o**2).sum(-1).mean((1,2)).sum()])*1e6
        df+=n-1;count+=n
    return total,df,count


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=188);p.add_argument('--event-only',action='store_true');a=p.parse_args()
    out=a.output;samples=json.loads((out/'bank.json').read_text())['samples']
    with np.load(out/'bank.npz') as f:z={k:f[k] for k in f.files}
    if a.event_only:
        keep=np.array([i for i,s in enumerate(samples) if s['event_type']>0 and 0<s['event_delta']<24])
        samples=[samples[i] for i in keep];z={k:v[keep] for k,v in z.items()}
    report={};names=('base','contact_presence','object_region','hand_region','interaction')
    for gate in (.02,.05,.1):
        for timing in (False,True):
            selected=groups.choose_groups(samples,z,gate,timing);aligned=dict(z,hand=z['hand'].copy(),scene=z['scene'].copy())
            align_groups=0;align_windows=0;anchors=[]
            for g in selected:
                times=np.array([samples[i]['event_delta'] for i in g]);eligible=np.array([samples[i]['event_type']>0 for i in g])&(times>0)&(times<24)
                if eligible.all():
                    anchor=float(np.median(times));anchors.append(anchor);align_groups+=1;align_windows+=len(g)
                    for i in g:
                        for k in ('hand','scene'):aligned[k][i]=warp_trajectory(z[k][i],samples[i]['event_delta'],anchor)
            stages={};previous=None;raw_group=[];al_group=[]
            for stage,name in enumerate(names):
                per_group=[partitions(g,samples,stage) for g in selected];parts=[p for gg in per_group for p in gg]
                raw=[pooled(pp,z) for pp in per_group];al=[pooled(pp,aligned) for pp in per_group]
                sums=sum((v[0] for v in raw),np.zeros(4));aligned_sums=sum((v[0] for v in al),np.zeros(4))
                df=sum(v[1] for v in raw);n=sum(v[2] for v in raw)
                row=dict(windows=n,df=df,partitions=len(parts),condition_distances=condition_distances(parts,z),
                         raw_variance_mm2=(sums/df).tolist() if df else None,
                         event_aligned_variance_mm2=(aligned_sums/df).tolist() if df else None)
                if df:
                    row['alignment_drop_percent']=(100*(1-aligned_sums/np.maximum(sums,1e-10))).tolist()
                    if previous and previous['df']:
                        row['raw_drop_vs_previous_percent']=(100*(1-(sums/df)/np.maximum(previous['raw_variance_mm2'],1e-10))).tolist()
                        row['aligned_drop_vs_previous_percent']=(100*(1-(aligned_sums/df)/np.maximum(previous['event_aligned_variance_mm2'],1e-10))).tolist()
                stages[name]=row;previous=row;raw_group.append(raw);al_group.append(al)
            # Label count/width control: shuffle full correlated label tuples
            # within original base groups. No narrow subgroup is treated as zero.
            rng=np.random.default_rng(a.seed);permuted=[]
            for repeat in range(100):
                perm=np.arange(len(samples))
                for g in selected:perm[g]=rng.permutation(g)
                row=[]
                for stage in range(1,5):
                    parts=[p for g in selected for p in partitions(g,samples,stage,perm)]
                    raw,df,_=pooled(parts,z);al,_,_=pooled(parts,aligned)
                    row.append(np.r_[raw/df,al/df] if df else np.full(8,np.nan))
                permuted.append(row)
            permuted=np.array(permuted)
            for j,name in enumerate(names[1:]):
                v=permuted[:,j];stages[name]['label_permutation95_mm2']=np.nanpercentile(v,[2.5,97.5],axis=0).T.tolist() if np.isfinite(v).any() else None
            # Paired bootstrap over independent BASE groups, common draw for
            # raw/aligned and label stages. This is exploratory, not validation.
            if len(selected)>=8:
                draws=rng.multinomial(len(selected),np.ones(len(selected))/len(selected),size=500)
                for j,name in enumerate(names):
                    rr=np.array([np.r_[v[0],v[1]] for v in raw_group[j]]);aa=np.array([np.r_[v[0],v[1]] for v in al_group[j]])
                    r=draws@rr;al=draws@aa;ok=r[:,-1]>0
                    stages[name]['raw95_mm2']=np.percentile(r[ok,:4]/r[ok,-1,None],[2.5,97.5],axis=0).T.tolist() if ok.any() else None
                    stages[name]['aligned95_mm2']=np.percentile(al[ok,:4]/al[ok,-1,None],[2.5,97.5],axis=0).T.tolist() if ok.any() else None
            report[str(round(gate*1000))+('_time3' if timing else '')]=dict(groups=len(selected),windows=sum(map(len,selected)),
                aligned_groups=align_groups,aligned_windows=align_windows,aligned_event_median_ticks=anchors,stages=stages)
            print({k:v for k,v in report.items() if k==str(round(gate*1000))+('_time3' if timing else '')},flush=True)
    result=dict(status='COMPLETED',seed=a.seed,event_inside_horizon_only=a.event_only,bank_windows=len(samples),bank_captures=len(set(s['capture'] for s in samples)),
                metric_order=groups.NAMES,groups=report,
                alignment='piecewise-linear phase warp preserving t0/t24;BASE group median E_next;only groups whose common event lies in1..23;other groups unchanged and counted',
                caveat='Aligned prefix8 means phase ticks, not actual first8 execution ticks. No exact-input irreducibility claim; unseen Ref unavailable.')
    (out/('event-only-aligned-variance.json' if a.event_only else 'event-aligned-variance.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
