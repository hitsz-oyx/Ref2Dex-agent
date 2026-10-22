"""Per-object source failures, actual joints and geometry from saved artifacts."""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-radius),min(1.,center+radius)]


def death(row):
    if row['final_success']:return 'success'
    if row['physical_abort']:return row['physical_abort']
    stages=row['stage_achieved']
    # This is a stage certificate, not proof that no physical contact occurred:
    # force-bearing grasps can fail the landmark/sample proximity subcriterion.
    if not stages['contact']:return 'stage_contact_not_certified'
    if not stages['lift']:return 'contact_without_lift'
    if not stages['hold']:return 'lift_without_continuous_hold'
    return 'held_off_goal' if row['final_hold'] else 'held_then_lost'


def summarize(rows):
    records=[]
    for row in rows:
        folder=Path(row['run'])/row['id'];audit=json.loads((folder/'native-audit.json').read_text())
        info=json.loads((folder/'consumer.json').read_text());bounds=audit['physical_joint_limits']
        with np.load(folder/'trajectory.npz') as z:
            t=len(z['action']);q=z['q'][-1,0];dq=z['dq'][-1,0]
            violations=row['physical_safety_final'].get('joint_violation_indices',[])
            joints=[dict(index=i,name=audit['native_dof_names'][i],q=float(q[i]),dq=float(dq[i]),
                lower=float(bounds['lower'][i]),upper=float(bounds['upper'][i]),
                excess_rad=float(max(bounds['lower'][i]-q[i],q[i]-bounds['upper'][i],0)),
                preceding_pd_target=float(z['pd_targets_actual'][-1,i]) if t else None) for i in violations]
            # Geometric evidence is measured, rather than inferred from a label.
            obj=z['objects'][-1,0,:3];palm=z['palm_pose'][-1,0,:3];tips=z['hand'][-1,0,[2,4,6,8,10]]
            geometry=dict(palm_object_origin_distance_m=float(np.linalg.norm(palm-obj)),
                fingertip_object_origin_distances_m=np.linalg.norm(tips-obj,axis=-1).tolist(),
                object_height_change_m=float(obj[2]-z['objects'][0,0,2]),
                final_goal_error_m=row['final_goal_error_m'],position_tolerance_m=row['geometry']['position_tolerance_m'],
                support_gap_m=row['physical_safety_final']['table_gap_m'])
            force=np.linalg.norm(z['contact_forces'][:,0],axis=-1)
            force_quantiles=np.quantile(force.max(-1),[.5,.9,1]).tolist()
        records.append(dict(id=row['id'],sequence=row['sequence'],object=row['object'],death=death(row),
            final_phase=info[-1]['phase'] if info else 'reset',controls=t,final_success=row['final_success'],
            longest_hold_s=row['longest_hold_s'],joints=joints,geometry=geometry,force_max_quantiles_n=force_quantiles,
            unloading_controls=sum(bool(i.get('source_ceiling',{}).get('unloading')) for i in info),
            source_admission_failures=row['source_admission_failures'],final_clipped_controls=row['final_clipped_controls']))
    objects={}
    for obj in sorted(set(r['object'] for r in records)):
        cases=[r for r in records if r['object']==obj];counts={}
        for case in cases:counts[case['death']]=counts.get(case['death'],0)+1
        objects[obj]=dict(n=len(cases),successes=sum(r['final_success'] for r in cases),causes=counts,cases=cases)
    n=len(records);k=sum(r['final_success'] for r in records)
    return dict(n=n,successes=k,wilson95=wilson(k,n),controls=sum(r['controls'] for r in records),
        joint_abort_episodes=sum(r['death']=='physical_joint_limit' for r in records),
        joint_events=sum(len(r['joints']) for r in records),per_object=objects,records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--manifest',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads(a.baseline.read_text())['rows'];baseline=[r for r in rows if r['arm']=='A0']
    if len(baseline)!=40:raise ValueError('exact original source n40 required')
    result=dict(baseline=summarize(baseline),new=None)
    if a.manifest:
        new=[]
        for m in a.manifest:
            manifest=json.loads(m.read_text())
            if manifest['status']!='COMPLETED':raise ValueError('unfinished worker retained; not a complete n40 result')
            new.extend(dict(r,run=str(m.parent.resolve())) for r in manifest['runs'])
        if len(new)!=40 or {r['trial'] for r in new}!={r['trial'] for r in baseline}:raise ValueError('same forty assigned cases required')
        original={r['trial']:r for r in baseline};initial_count=0
        for row in new:
            old=original[row['trial']]
            with np.load(Path(old['run'])/old['id']/'initial.npz') as x,np.load(Path(row['run'])/row['id']/'initial.npz') as y:
                if set(x.files)!=set(y.files) or any(not np.allclose(x[k],y[k],rtol=0,atol=2e-6) for k in x.files):raise ValueError('initial state changed '+row['trial'])
            initial_count+=1
        result['new']=summarize(new);b=result['baseline'];c=result['new']
        result.update(initials_matched=initial_count,target_passed=c['successes']>=32,
            fisher_two_sided=float(fisher_exact([[c['successes'],40-c['successes']],[b['successes'],40-b['successes']]])[1]),
            caveat='Same forty initials are a development panel; 20 objects x2 cases are clustered. Wilson/Fisher are descriptive conditional-panel statistics, not independent-object generalization.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','new')}));print('baseline',result['baseline']['successes'],'/40; new',None if not result['new'] else result['new']['successes'])


if __name__=='__main__':main()
