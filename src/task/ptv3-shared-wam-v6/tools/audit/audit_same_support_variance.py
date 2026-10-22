"""Control label-stage singleton removal: compare EXACT same kept windows.

Reuses saved complete-link groups; no new neighborhood search. The previous
label partition is restricted to the current stage's nonsingleton windows.
This distinguishes variance reduction from merely dropping rare-label samples.
"""
import argparse
from pathlib import Path
import numpy as np
from audit_event_aligned_variance import partitions,pooled,warp_trajectory,condition_distances
from audit_identifiability import read,write


def percent_drop(reference,current):
    ratio=np.divide(current,reference,out=np.full_like(np.asarray(current,dtype=float),np.nan),where=np.asarray(reference)>1e-12)
    return 100*(1-ratio)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    source=read(out/'event-only-aligned-variance.json');samples=read(out/'bank.json')['samples']
    with np.load(out/'bank.npz') as f:z={k:f[k] for k in f.files}
    keep=np.array([i for i,s in enumerate(samples) if s['event_type']>0 and 0<s['event_delta']<24])
    samples=[samples[i] for i in keep];z={k:v[keep] for k,v in z.items()};results={}
    for gate,report in source['groups'].items():
        groups=report.get('group_indices',[]);aligned=dict(z,hand=z['hand'].copy(),scene=z['scene'].copy())
        for g in groups:
            anchor=float(np.median([samples[i]['event_delta'] for i in g]))
            for i in g:
                for k in ('hand','scene'):aligned[k][i]=warp_trajectory(z[k][i],samples[i]['event_delta'],anchor)
        stages={};rng=np.random.default_rng(188)
        draws=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=500) if groups else None
        for stage,name in enumerate(('contact_presence','object_region','hand_region','interaction'),1):
            current=[];previous=[]
            for g in groups:
                pp=partitions(g,samples,stage);eligible=set(i for ids in pp for i in ids)
                reference=[[i for i in ids if i in eligible] for ids in partitions(g,samples,stage-1)]
                reference=[ids for ids in reference if len(ids)>=2]
                if set(i for ids in reference for i in ids)!=eligible:raise AssertionError('same-support invariant')
                current.append(pp);previous.append(reference)
            record={}
            for alignment,values in [('raw',z),('event_aligned',aligned)]:
                cc=np.array([np.r_[v[0],v[1],v[2]] for v in (pooled(pp,values) for pp in current)]).reshape(-1,6)
                rr=np.array([np.r_[v[0],v[1],v[2]] for v in (pooled(pp,values) for pp in previous)]).reshape(-1,6)
                c=cc.sum(0);r=rr.sum(0)
                if c[5]!=r[5]:raise AssertionError('kept N differs')
                result=dict(windows=int(c[5]),reference_df=int(r[4]),refined_df=int(c[4]))
                if c[4] and r[4]:
                    cv=c[:4]/c[4];rv=r[:4]/r[4];result.update(previous_variance_mm2=rv.tolist(),refined_variance_mm2=cv.tolist(),drop_percent=percent_drop(rv,cv).tolist())
                    bs=draws@cc;br=draws@rr;ok=(bs[:,4]>0)&(br[:,4]>0)
                    if len(groups)>=8 and ok.any():
                        drop=percent_drop(br[ok,:4]/br[ok,4,None],bs[ok,:4]/bs[ok,4,None])
                        result['drop95']=np.nanpercentile(drop,[2.5,97.5],axis=0).T.tolist()
                record[alignment]=result
            record['previous_same_kept_windows_distances']=condition_distances([p for pp in previous for p in pp],z)
            record['refined_distances']=condition_distances([p for pp in current for p in pp],z);stages[name]=record
        results[gate]=dict(groups=len(groups),stages=stages)
    write(out/'event-only-same-support.json',dict(status='COMPLETED',seed=188,metric_order=source['metric_order'],results=results,
        estimator='Paired unbiased pooled within-group variance with identical current/referral kept windows;previous partitions restricted before comparison;capture-group bootstrap500'))
    print({k:v['groups'] for k,v in results.items()},flush=True)


if __name__=='__main__':main()
