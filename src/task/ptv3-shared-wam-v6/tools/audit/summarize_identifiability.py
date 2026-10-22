"""Rebuild the handoff tables and capture-paired intervals from audit JSON only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(p.read_text())
def mean(x,k):return x[k]['mean']


def paired_ci(a,b,part,key,other=None,seed=180):
    if len(a)!=len(b) or any(x['row']!=y['row'] for x,y in zip(a,b)):raise ValueError('paired input mismatch')
    groups=defaultdict(lambda:np.zeros(4,np.float64))
    for x,y in zip(a,b):
        left=x['metrics'][part][key];right=y['metrics'][part][other or key]
        groups[x['row']['capture']]+=np.asarray([left['sum'],left['count'],right['sum'],right['count']])
    values=np.stack(list(groups.values()));rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(values),np.ones(len(values))/len(values),size=1000);sample=weights@values
    known=(sample[:,1]>0)&(sample[:,3]>0)
    delta=sample[known,2]/sample[known,3]-sample[known,0]/sample[known,1]
    return np.percentile(delta,[2.5,97.5]).tolist() if len(delta) else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    panel=read(out/'panels.json');identity=read(out/'identifiability.json');loaded={};tables=[];geometry=[];contacts=[]
    for name,rows in panel['panels'].items():
        frame_counts={};times=np.arange(24)
        for part,lo,hi in [('prefix8',0,8),('middle8',8,16),('last8',16,24),('all24',0,24)]:
            frame_counts[part]=dict(windows=len(rows),captures=len(set(r['capture'] for r in rows)),frames=len(rows)*(hi-lo))
        for k,part in enumerate(('pre_contact','contact','post_release','unknown')):
            hit=[r for r in rows if k in r['phase']]
            frame_counts[part]=dict(windows=len(hit),captures=len(set(r['capture'] for r in hit)),frames=sum(r['phase'].count(k) for r in rows))
            for band,lo,hi in [('prefix8',0,8),('middle8',8,16),('last8',16,24)]:
                hit=[r for r in rows if k in r['phase'][lo:hi]]
                frame_counts[band+'/'+part]=dict(windows=len(hit),captures=len(set(r['capture'] for r in hit)),frames=sum(r['phase'][lo:hi].count(k) for r in rows))
        for model in ('healthy4371','flow8500'):
            path=out/(model+'-'+name+'.json')
            if path.exists():loaded[(model,name)]=read(path)
        if not all((m,name) in loaded for m in ('healthy4371','flow8500')):continue
        base=loaded[('healthy4371',name)];flow=loaded[('flow8500',name)]
        for part in frame_counts:
            bm=base['metrics'][part];fm=flow['metrics'][part]
            row=dict(panel=name,stratum=part,**frame_counts[part],healthy={},flow={},delta_flow_minus_healthy={},delta95_capture_bootstrap={})
            for key in ('hand_epe_mm','object_epe_mm','C_head_mae'):
                row['healthy'][key]=mean(bm,key);row['flow'][key]=mean(fm,key)
                row['delta_flow_minus_healthy'][key]=mean(fm,key)-mean(bm,key) if mean(bm,key) is not None else None
                if part=='prefix8':row['delta95_capture_bootstrap'][key]=paired_ci(base['records'],flow['records'],part,key)
            tables.append(row)
        for model,report in [('healthy4371',base),('flow8500',flow)]:
            for part in ('prefix8','all24'):
                metrics=report['metrics'][part];row=dict(model=model,panel=name,stratum=part,**frame_counts[part],metrics={})
                for key in ('contact_distance_mm','geometric_contact_mae','clearance_violation_mm','oriented_penetration_proxy_mm','oriented_inside_proxy_fraction','contact_recall3mm'):
                    left='pred_hand_GT_object/'+key;right='pred_hand_pred_object/'+key;ref='GT_hand_GT_object/'+key
                    lv=mean(metrics,left);rv=mean(metrics,right)
                    row['metrics'][key]=dict(GT_object=lv,predicted_object=rv,gap_pred_minus_GT=rv-lv if lv is not None else None,
                        GT_hand_GT_object=mean(metrics,ref),count=metrics[left]['count'],
                        gap95_capture_bootstrap=paired_ci(report['records'],report['records'],part,left,right))
                row['affine_fit_error_mm']=mean(metrics,'affine_fit_error_mm');row['object_dense_affine_epe_mm']=mean(metrics,'object_dense_affine_epe_mm')
                geometry.append(row)
    # Raw per-pair quantiles show whether dispersion means are dominated by a few
    # recordings; no clipping or silent target filtering is applied.
    quantiles={};presence_sensitivity={};bank_meta=read(out/'bank.json')['samples']
    for gate in (20,50,100):
        z=np.load(out/('pairs-%dmm.npz'%gate));v=z['variance_mm2']
        quantiles[str(gate)]=dict(pairs=len(v),per_pair_variance_mm2_p50=np.percentile(v,50,axis=0).tolist(),
            per_pair_variance_mm2_p90=np.percentile(v,90,axis=0).tolist(),per_pair_variance_mm2_max=v.max(0).tolist()) if len(v) else dict(pairs=0)
        i,j=z['pairs'].T;available=np.array([s['contact_available'] for s in bank_meta]);region=np.array([s['region'] for s in bank_meta])
        both=available[i]&available[j];same=both&(region[i]==region[j]);distance=z['state_goal_distance_m']
        presence_sensitivity[str(gate)]={name:dict(pairs=int(mask.sum()),variance_mm2=v[mask].mean(0).tolist() if mask.any() else None,
            raw_condition_distances_mm=(distance[mask].mean(0)*1000).tolist() if mask.any() else None)
            for name,mask in [('all',np.ones(len(i),bool)),('both_future_contact_available',both),('same_oracle_region',same)]}
    (out/'contact-presence-sensitivity.json').write_text(json.dumps(presence_sensitivity,indent=2)+'\n')
    result=dict(status='COMPLETED' if len(loaded)==2*len(panel['panels']) else 'PARTIAL',seed=178,bootstrap_seed=180,
        sample_counts={k:dict(windows=len(v),captures=len(set(r['capture'] for r in v))) for k,v in panel['panels'].items()},
        bank_samples=identity['samples'],variance=identity['gates'],pair_quantiles=quantiles,contact_presence_sensitivity=presence_sensitivity,prefix_and_phase=tables,self_consistency=geometry,
        commands=dict(recompute='python src/task/ptv3-shared-wam-v6/tools/audit/summarize_identifiability.py --output '+str(out.resolve())),
        caveats=['heldout_prefix_local_recipe is not certified lifetime-heldout because remote ancestor windows are unavailable',
        'conditional pair variance is mm^2,RMS spread not EPE;gate100mm is weak-match sensitivity',
        'penetration numbers are oriented point-cloud proxies,not mesh SDF;GT-hand/GT-object calibration is included',
        'confidence>0.25 contact mask includes soft proximity out to~17mm;GT reference distances need not be3mm',
        'contact/intent refinements use oracle realized labels;no causal intent tensor was added or fitted'])
    for key,filename in [('strict_group_variance','group-identifiability.json'),('watertight_mesh','mesh-penetration.json')]:
        if (out/filename).exists():result[key]=read(out/filename)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Existing-checkpoint WAM-v6 audit numerical handoff','','Seed178;capture bootstrap/permutation180;flow noise172;GRAB only.','',
        '|Panel|stratum|windows/captures|frames|healthy hand/object mm|flow hand/object mm|','|---|---|---|---:|---|---|']
    fmt=lambda v:'NA' if v is None else '%.3f'%v
    for row in tables:
        if row['stratum']=='unknown':continue
        lines.append('|%s|%s|%d/%d|%d|%s / %s|%s / %s|'%(row['panel'],row['stratum'],row['windows'],row['captures'],row['frames'],fmt(row['healthy']['hand_epe_mm']),fmt(row['healthy']['object_epe_mm']),fmt(row['flow']['hand_epe_mm']),fmt(row['flow']['object_epe_mm'])))
    lines+=['','## Conditional dispersion (mm²;order hand8,hand24,object8,object24)','','|gate|condition|pairs/captures|variance mm²|','|---|---|---|---|']
    for gate,report in identity['gates'].items():
        for name,row in report.get('stages',{}).items():
            if name=='observed_object_and_finger':continue
            values=row['variance_mm2'];lines.append('|%s|%s|%d/%d|%s|'%(gate,name,row['pairs'],row['captures'],' / '.join(fmt(v) for v in values.values()) if values else 'NA'))
    lines+=['','## Geometry prefix8: GT object vs predicted object','','|Panel|model|metric|n query/vertex frames|GT object|pred object|difference|GT hand+GT object|','|---|---|---|---:|---:|---:|---:|---:|']
    for row in geometry:
        if row['stratum']!='prefix8':continue
        for key,value in row['metrics'].items():
            lines.append('|%s|%s|%s|%d|%s|%s|%s|%s|'%(row['panel'],row['model'],key,value['count'],fmt(value['GT_object']),fmt(value['predicted_object']),fmt(value['gap_pred_minus_GT']),fmt(value['GT_hand_GT_object'])))
    lines+=['','All aggregate sums/counts and per-window values are in model-panel JSON;panel triples and source provenance in panels.json.','',*['- '+c for c in result['caveats']]]
    (out/'summary.md').write_text('\n'.join(lines)+'\n');print(result['status'],len(tables),'strata rows',len(geometry),'geometry rows')


if __name__=='__main__':main()
