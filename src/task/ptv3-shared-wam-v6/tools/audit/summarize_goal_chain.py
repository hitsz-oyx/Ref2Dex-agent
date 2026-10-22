"""Recomputable sums/counts, capture-paired intervals and goal contract gaps."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from audit_goal_chain import GOALS,METRICS,BANDS,PHASES,NOISE_SEEDS,MANIFEST,VERSION
from audit_identifiability import read,write


def summary(total,nwindows,valid_windows):
    result={}
    for gi,goal in enumerate(GOALS):
        goal_result={}
        for bi,band in enumerate(BANDS):
            byphase={}
            for pi,phase in enumerate(PHASES):
                values={}
                for mi,name in enumerate(METRICS):
                    s,c=total[gi,bi,pi,mi];values[name]=dict(mean=float(s/c) if c else None,sum=float(s),count_per_draw=float(c),windows=int(valid_windows[gi,bi,pi,mi]))
                for prefix in ('C','GTobj_geom','ownobj_geom'):
                    tp=values[prefix+'_tp']['sum'];fp=values[prefix+'_fp']['sum'];fn=values[prefix+'_fn']['sum'];den=2*tp+fp+fn
                    values[prefix+'_F1']=dict(value=2*tp/den if den else None,tp=tp,fp=fp,fn=fn,queries_per_draw=values[prefix+'_tp']['count_per_draw'])
                byphase[phase]=values
            goal_result[band]=byphase
        result[goal]=goal_result
    return dict(windows=nwindows,goals=result)


def bootstrap_contrasts(records,lookup,seed=188):
    result={};rng=np.random.default_rng(seed)
    # Contrast on exactly matched rows; averages over K before capture inference.
    for model in records:
        for panel in ('heldout_sequence','train_prefix','heldout_prefix_local_recipe'):
            ids=[k for k in records[model] if lookup[k]['panel']==panel]
            if not ids:continue
            cap=sorted(set(lookup[k]['capture'] for k in ids));ci={c:i for i,c in enumerate(cap)}
            totals=np.zeros((len(cap),3,len(METRICS),2))
            for k in ids:totals[ci[lookup[k]['capture']]]+=records[model][k][:,1,0]
            draws=rng.multinomial(len(cap),np.ones(len(cap))/len(cap),size=500)
            sums=np.einsum('bc,cgms->bgms',draws,totals)
            means=sums[...,0]/np.maximum(sums[...,1],1e-20)
            contrasts={}
            for goal in (1,2):
                metrics={}
                for name in ('hand_epe_mm','object_epe_mm','C_location_mm','GTobj_contact_distance_mm','ownobj_contact_distance_mm','GTobj_oriented_depth_proxy_mm','ownobj_oriented_depth_proxy_mm'):
                    mi=METRICS.index(name);v=means[:,goal,mi]-means[:,0,mi]
                    metrics[name]=dict(delta95=np.percentile(v,[2.5,97.5]).tolist())
                tp=METRICS.index('C_tp');fp=METRICS.index('C_fp');fn=METRICS.index('C_fn')
                f1=2*sums[:,:,tp,0]/np.maximum(2*sums[:,:,tp,0]+sums[:,:,fp,0]+sums[:,:,fn,0],1e-20)
                metrics['C_F1']=dict(delta95=np.percentile(f1[:,goal]-f1[:,0],[2.5,97.5]).tolist())
                contrasts[GOALS[goal]+'_minus_oracle']=metrics
            result[model+'/'+panel]=dict(windows=len(ids),captures=len(cap),contrasts=contrasts)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    panel=read(out/'panels.json');rows=[r for r in panel['rows'] if r['evaluate']];lookup={tuple(r[k] for k in ('seq_id','slot','tick')):r for r in rows}
    totals={};valid={};counts=Counter();captures=defaultdict(set);records=defaultdict(dict);seed_totals={};diversity={}
    for folder in sorted(out.glob('worker[0-9]')):
        for modeldir in sorted(folder.iterdir()):
            if not modeldir.is_dir():continue
            label=modeldir.name
            for path in sorted(modeldir.glob('*.npz')):
                with np.load(path) as z:ids=z['ids'];m=z['metrics'];d=z['diversity']
                average=m.mean(2)
                for i,identity in enumerate(ids):
                    key=tuple(map(int,identity));r=lookup[key]
                    if key in records[label]:raise ValueError('duplicate evaluation window')
                    records[label][key]=average[i]
                    for source in ('all',str(r['source_id'])):
                        for core in ('full','nonoverlap_core'):
                            if core=='nonoverlap_core' and not r['nonoverlap_core']:continue
                            k=(label,r['panel'],source,core)
                            if k not in totals:totals[k]=np.zeros_like(average[i]);valid[k]=np.zeros_like(average[i,...,0],dtype=int)
                            totals[k]+=average[i];valid[k]+=average[i,...,1]>0;counts[k]+=1;captures[k].add(r['capture'])
                    k=(label,r['panel'])
                    if k not in seed_totals:seed_totals[k]=np.zeros_like(m[i]);diversity[k]=[]
                    seed_totals[k]+=m[i];diversity[k].append(d[i])
    results={}
    for k,total in totals.items():
        report=summary(total,counts[k],valid[k]);report['captures']=len(captures[k]);results['/'.join(k)]=report
    seeds={}
    for k,total in seed_totals.items():
        # Keep seed means plus prefix hand/object coordinate spread in mm.
        perdraw=total[:, :,1,0,:2,0]/np.maximum(total[:,:,1,0,:2,1],1)
        dd=np.stack(diversity[k]);seeds['/'.join(k)]=dict(seed_ids=list(NOISE_SEEDS[:total.shape[1]]) if k[0]=='flow8500' else ['deterministic'],
            prefix_EPE_mm_by_goal_draw_hand_object=perdraw.tolist(),
            mean_coordinate_std_mm_by_goal_hand_object=[[float(dd[:,g,:8,:126].mean()*1000),float(dd[:,g,:8,126:].mean()*1000)] for g in range(3)])
    expected=Counter(r['panel'] for r in rows);coverage={label:{p:sum(lookup[k]['panel']==p for k in rec) for p in expected} for label,rec in records.items()}
    complete=all(coverage.get(label,{})==dict(expected) for label in ('healthy4371','flow8500'))
    result=dict(status='COMPLETED' if complete else 'PARTIAL',work_version=VERSION,expected_windows=dict(expected),coverage=coverage,
        protocol='K-average sum/count per window,then pooled points;capture-cluster paired bootstrap500 seed188;not independent model seeds',
        results=results,noise_sampling=seeds,paired_goal_intervals=bootstrap_contrasts(records,lookup))
    write(out/'large-summary.json',result)
    lines=['# Large goal intervention audit', '', 'Status: '+result['status'], 'Coverage: '+json.dumps(coverage),'',
           '|Model/panel|N|Oracle hand/object mm|Planner hand/object mm|Perturbed hand/object mm|C F1 Oracle/Planner/Perturbed|',
           '|---|---:|---|---|---|---|']
    for key,r in results.items():
        if not key.endswith('/all/full'):continue
        values=[];f=[]
        for goal in GOALS:
            v=r['goals'][goal]['prefix8']['all'];values.append('/'.join('%.3f'%v[n]['mean'] for n in ('hand_epe_mm','object_epe_mm')))
            f.append('%.4f'%v['C_F1']['value'] if v['C_F1']['value'] is not None else 'NA')
        lines.append('|%s|%d|%s|%s|%s|%s|'%(key,r['windows'],*values,'/'.join(f)))
    (out/'large-summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
