"""Independent replay of measured object tracking, with complete matrix CSV."""
import argparse,csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,RotationSpline
ROOT=Path(__file__).resolve().parents[5]


def matrix_pose(values):
    values=np.asarray(values);out=np.repeat(np.eye(4)[None],len(values),0)
    out[:,:3,3]=values[:,:3];out[:,:3,:3]=Rotation.from_quat(values[:,3:7]).as_matrix();return out


def angle(relative):
    return np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))


def audit(run):
    manifest=json.loads((run/'manifest.json').read_text())
    if manifest['status'] not in ('COMPLETED','STOPPED_P0'):raise ValueError('only finalized runs; do not audit partial files')
    expected=json.loads((run/'registered-matrix.json').read_text());results=manifest['results'];failures=[];rows=[];initials=[];references={};maximums=defaultdict(float);counts=Counter();by_arm=defaultdict(list)
    limits=manifest['native_audit']['physical_joint_limits'];lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper']);limited=np.asarray(limits['has_limits'],bool)
    offset=np.asarray(manifest['native_audit']['controller']['offset']);scale=np.asarray(manifest['native_audit']['controller']['scale']);active=[0,1,2,3,4,5,6,8,10,12,14,15]
    for result in results:
        folder=run/('case%03d'%result['index']);case=expected[result['index']]
        if any(result[k]!=v for k,v in case.items()):failures.append([result['index'],'registered case changed'])
        if result['tracked_controls']==0:
            rows.append(dict(case,index=result['index'],status=result['status']));continue
        with np.load(folder/'trajectory.npz') as z:states={k:z[k].copy() for k in z.files}
        with np.load(folder/'reference.npz') as z:knots=z['poses'].copy();times=z['times'].copy();local=z['hand_object'].copy()
        with np.load(folder/'controls.npz') as z:raw=z['raw_action'].copy();pd=z['pd_target'].copy()
        with np.load(folder/'initial.npz') as z:initials.append(z['q'].copy())
        tick_rows=json.loads((folder/'ticks.json').read_text());clock=np.asarray([x['time_s'] for x in tick_rows]);n=len(clock)
        if len(raw)+1!=n or not np.all(np.diff(states['tick'][:,0])==1):failures.append([result['index'],'post-state/control clock'])
        actual=matrix_pose(states['objects'][:,0]);palms=matrix_pose(states['palm_pose'][:,0]);position=CubicSpline(times,knots[:,:3,3],bc_type='clamped');rotation=RotationSpline(times,Rotation.from_matrix(knots[:,:3,:3]))
        reference_clock=np.minimum(clock,times[-1]);desired_pos=position(reference_clock);desired_rot=rotation(reference_clock).as_matrix();e=np.linalg.norm(actual[:,:3,3]-desired_pos,axis=-1);a=angle(desired_rot.transpose(0,2,1)@actual[:,:3,:3])
        relative=np.linalg.inv(palms)@actual;drift=np.linalg.norm(relative[:,:3,3]-local[:3,3],axis=-1);da=angle(local[:3,:3].T@relative[:,:3,:3])
        for key,recomputed in [('position_error_m',e),('orientation_error_rad',a),('drift_m',drift),('drift_rad',da)]:
            delta=float(np.max(np.abs(recomputed-np.asarray([x[key] for x in tick_rows]))));maximums[key]=max(maximums[key],delta)
            if delta>2e-7:failures.append([result['index'],key,delta])
        q=states['q'][:,0];margin=np.minimum(q-lo,hi-q);margin[:,~limited]=np.inf
        dense=np.linspace(0,times[-1],2001);vel=position(dense,1);acc=position(dense,2);speed=np.linalg.norm(vel,axis=-1);curvature=np.linalg.norm(np.cross(vel,acc),axis=-1)/np.maximum(speed,1e-12)**3
        meaningful=(speed>max(speed.max()*1e-3,1e-7))&(curvature>1e-5)
        radius=1/curvature[meaningful] if meaningful.any() else np.array([])
        v_peak=float(speed.max());acc_peak=float(np.linalg.norm(acc,axis=-1).max())
        omega_peak=float(np.linalg.norm(rotation(dense,1),axis=-1).max())
        commands=json.loads((folder/'commands.json').read_text())
        mount=Rotation.from_euler('XYZ',q[0,3:6]).as_matrix().T@palms[0,:3,:3]
        def nearby(rot,current):
            value=Rotation.from_matrix(rot@mount.T).as_euler('XYZ')
            choices=np.asarray([value,[value[0]+np.pi,np.pi-value[1],value[2]+np.pi]])
            choices+=2*np.pi*np.round((current-choices)/(2*np.pi))
            return choices[np.argmin(np.linalg.norm(choices-current,axis=-1))]
        translation_caps=angular_caps=0
        for t,command in enumerate(commands):
            target=np.asarray(command['wrist_target']);dq=states['dq'][t,0]
            angles=nearby(target[:3,:3],q[t,3:6])
            trans=.45*(target[:3,3]-palms[t,:3,3])-.012*dq[:3]
            angular=.4*(angles-q[t,3:6])-.008*dq[3:6]
            if case['feedforward']:
                trans+=np.asarray(command['velocity_ff_world'])/30
                advanced=Rotation.from_rotvec(np.asarray(command['angular_velocity_ff_world'])*1e-5).as_matrix()@target[:3,:3]
                angular+=(nearby(advanced,angles)-angles)/1e-5/30
            translation_caps+=int(np.any(np.abs(trans)>.025+1e-10));angular_caps+=int(np.any(np.abs(angular)>.12+1e-10))
        expanded=np.zeros((len(raw),18));expanded[:,active]=np.clip(raw,-1,1)
        expected_pd=(expanded+1)/2*scale+offset;expected_pd[:,:6]=q[:-1,:6]+offset[:6]+expanded[:,:6]*scale[:6]
        for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):expected_pd[:,dst]=expected_pd[:,src]*ratio
        pd_delta=float(np.max(np.abs(expected_pd-pd)));maximums['pd_mapping']=max(maximums['pd_mapping'],pd_delta)
        if pd_delta>2e-6:failures.append([result['index'],'PD mapping',pd_delta])
        key=(result['shape'],result['scale_m'],result['requested_mean_speed_m_s'])
        if key in references and not np.array_equal(knots,references[key]):failures.append([result['index'],'A/B/FF object reference mismatch'])
        references[key]=knots
        closure_valid=bool(result['closed'] and result['status']=='COMPLETED')
        closure_p=float(np.linalg.norm(actual[-1,:3,3]-knots[0,:3,3])) if closure_valid else None
        closure_a=float(np.rad2deg(angle(knots[0,:3,:3].T@actual[-1,:3,:3]))) if closure_valid else None
        min_index=np.unravel_index(np.argmin(margin),margin.shape);min_value=float(margin[min_index]);min_joint=int(min_index[1])
        def first(mask):
            ids=np.flatnonzero(mask);return float(clock[ids[0]]) if len(ids) else None
        contact=np.asarray([t['contact'] for t in tick_rows],bool)
        intact=contact&(drift<=.01)&(da<=np.deg2rad(10))
        since=None;longest=0.
        for time_value,valid in zip(clock,intact):
            if not valid:since=None
            else:
                if since is None:since=time_value
                longest=max(longest,float(time_value-since))
        firsts=dict(limit=first(np.any(margin<.02,axis=-1)),lag=first((e>.015)|(a>np.deg2rad(5))),
                    slip=None if result['stage']=='A' else first((drift>.01)|(da>np.deg2rad(10))),collision=first(np.asarray([bool(t['collision_pairs']) or t['safety']['reason']=='physical_table_penetration' for t in tick_rows])))
        causes=[(v,k) for k,v in firsts.items() if v is not None]
        if result['abort'] and result['abort']['kind']=='slip' and firsts['slip'] is None:causes.append((result['abort']['tick']/30,'slip'))
        primary=min(causes)[1] if causes else 'none'
        failure_class=None
        if not result['pass_tracking']:
            if result['abort']:failure_class=result['abort']['kind']
            elif min_value<.02:failure_class='limit_margin'
            elif drift.max()>.01 or da.max()>np.deg2rad(10):failure_class='slip'
            elif np.quantile(e,.95)>.015 or np.quantile(a,.95)>np.deg2rad(5) or closure_p is not None and (closure_p>.015 or closure_a>5):failure_class='lag'
            else:failure_class='other'
        independent_pass=bool(result['status']=='COMPLETED' and np.quantile(e,.95)<=.015 and np.quantile(a,.95)<=np.deg2rad(5) and drift.max()<=.01 and da.max()<=np.deg2rad(10) and min_value>=.02 and (not result['closed'] or closure_p is not None and closure_p<=.015 and closure_a<=5) and not any(t['collision_pairs'] for t in tick_rows))
        if independent_pass!=result['pass_tracking']:failures.append([result['index'],'independent success disagreement'])
        # Discrete paths with only 3 controls remain explicitly short, not long-hold evidence.
        record=dict(case,index=result['index'],status=result['status'],pass_tracking=result['pass_tracking'],
                    observed_s=float(clock[-1]),planned_s=float(times[-1]),controls=len(raw),
                    position_p50_mm=float(np.quantile(e,.5)*1000),position_p90_mm=float(np.quantile(e,.9)*1000),position_p95_mm=float(np.quantile(e,.95)*1000),position_max_mm=float(e.max()*1000),
                    orientation_p50_deg=float(np.rad2deg(np.quantile(a,.5))),orientation_p95_deg=float(np.rad2deg(np.quantile(a,.95))),orientation_max_deg=float(np.rad2deg(a.max())),
                    closure_valid=closure_valid,closure_mm=None if closure_p is None else closure_p*1000,closure_deg=closure_a,
                    slip_max_mm=float(drift.max()*1000),slip_max_deg=float(np.rad2deg(da.max())),
                    min_margin=min_value,min_margin_joint=min_joint,min_margin_joint_name=manifest['native_audit']['native_dof_names'][min_joint],
                    safety_abort=None if result['abort'] is None else result['abort']['kind'],clipped_controls=int(np.any(np.abs(raw)>1,axis=-1).sum()),
                    translation_servo_cap_controls=translation_caps,angular_servo_cap_controls=angular_caps,
                    collision_ticks=sum(bool(t['collision_pairs']) for t in tick_rows),primary_cause=primary,failure_class=failure_class,
                    first_limit_s=firsts['limit'],first_lag_s=firsts['lag'],first_slip_s=firsts['slip'],first_collision_s=firsts['collision'],
                    reference_peak_speed_m_s=v_peak,reference_peak_acceleration_m_s2=acc_peak,
                    reference_peak_angular_speed_rad_s=omega_peak,
                    reference_min_curvature_radius_m=None if not len(radius) else float(radius.min()),reference_median_curvature_radius_m=None if not len(radius) else float(np.median(radius)),
                    initial_wrist_speed_m_s=float(np.linalg.norm(states['dq'][0,0,:3])),initial_wrist_speed_rad_s=float(np.linalg.norm(states['dq'][0,0,3:6])))
        record.update(first_double_contact_loss_s=first(~contact),longest_force_coupled_relative_hold_s=None if result['stage']=='A' else longest)
        if not result['pass_tracking']:counts[(result['stage'],primary)]+=1
        rows.append(record);by_arm[(result['stage'],result['feedforward'])].append(record)
    if len(results)+len(manifest['unmeasured'])!=len(expected):failures.append(['incomplete matrix accounting'])
    groups={}
    for key,data in by_arm.items():
        groups[str(key)]=dict(cases=len(data),pass_count=sum(r['pass_tracking'] for r in data),complete_count=sum(r['status']=='COMPLETED' for r in data),
                             abort_causes=dict(Counter(r['safety_abort'] for r in data if r['safety_abort'])),
                             median_case_position_p95_mm=float(np.median([r['position_p95_mm'] for r in data])),median_case_orientation_p95_deg=float(np.median([r['orientation_p95_deg'] for r in data])),
                             maximum_slip_mm=max(r['slip_max_mm'] for r in data),maximum_slip_deg=max(r['slip_max_deg'] for r in data),
                             min_joint_margin=min(r['min_margin'] for r in data),clipped_controls=sum(r['clipped_controls'] for r in data),controls=sum(r['controls'] for r in data),
                             translation_servo_cap_controls=sum(r['translation_servo_cap_controls'] for r in data),angular_servo_cap_controls=sum(r['angular_servo_cap_controls'] for r in data))
    pairs=[]
    for key in references:
        for stage in ('A','B'):
            selected=[r for r in rows if r['shape']==key[0] and r['scale_m']==key[1] and r['requested_mean_speed_m_s']==key[2] and r['stage']==stage]
            if len(selected)==2:
                no=next(r for r in selected if not r['feedforward']);yes=next(r for r in selected if r['feedforward'])
                pairs.append(dict(shape=key[0],scale_m=key[1],mean_speed_m_s=key[2],stage=stage,
                                  both_complete=no['status']==yes['status']=='COMPLETED',position_p95_delta_mm=yes['position_p95_mm']-no['position_p95_mm'],orientation_p95_delta_deg=yes['orientation_p95_deg']-no['orientation_p95_deg'],
                                  off_pass=no['pass_tracking'],on_pass=yes['pass_tracking']))
    frozen_runner=run/'runtime-code/src/task/wam-rl-policy-v3/tools/run/run_object_tracking.py'
    force_holding="case['stage']=='B' and bool(state['contact_force_valid'][0])" in frozen_runner.read_text()
    summary=dict(status=manifest['status'],audit_failures=failures,audit_max_abs_differences=dict(maximums),cases=len(results),not_run=len(manifest['unmeasured']),
                 real_contact_existing_force_holding=force_holding,
                 physical_controls=manifest['controls'],groups=groups,failed_case_first_cause={str(k):v for k,v in counts.items()},
                 failed_case_constraint={str((stage,kind)):sum(r['stage']==stage and r.get('failure_class')==kind for r in rows) for stage in ('A','B') for kind in ('limit','limit_margin','lag','slip','collision','other')},
                 paired_ff_comparisons=pairs,reset_q_max_difference=float(np.abs(np.asarray(initials)-initials[0]).max()) if initials else None,
                 limitations=['A: imposed kinematic grasp, not welded/load-bearing dynamics.','B force-holding usage explicitly reported; no newly designed slip feedback.',
                              'Native contacts filtered at lambda>0.1; overlap field absent. Full negative contact-pair list not persisted; absence not independently certified.',
                              'Aborted path quantiles cover only observed prefix; closure is undefined for incomplete paths.','One known object/reset, one run per grid cell; no generalization/replicate confidence claim.'])
    (run/'audit-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (run/'matrix.csv').open('w') as f:
        fields=sorted({k for r in rows for k in r});writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    header='| 形状 | 尺度mm | 均速mm/s | A无FF P95 mm/deg | A有FF | B无FF | B有FF |\n|---|---:|---:|---|---|---|---|\n'
    markdown=[]
    for key in references:
        cells=[]
        for stage,ff in [('A',False),('A',True),('B',False),('B',True)]:
            r=next((r for r in rows if r['shape']==key[0] and r['scale_m']==key[1] and r['requested_mean_speed_m_s']==key[2] and r['stage']==stage and r['feedforward']==ff),None)
            if r is None:cells.append('未测');continue
            cells.append('{:.2f}/{:.2f} {}'.format(r['position_p95_mm'],r['orientation_p95_deg'],'通过' if r['pass_tracking'] else '中止:'+str(r['safety_abort']) if r['status']=='ABORT' else '未过阈值'))
        markdown.append('| {} | {:.0f} | {:.0f} | {} |'.format(key[0],key[1]*1000,key[2]*1000,' | '.join(cells)))
    (run/'matrix.md').write_text(header+'\n'.join(markdown)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='paired_ff_comparisons'},indent=2))
    return summary


def merge_stages(a_run,b_run):
    """Join qualified A/B only; keep rejected fixed-finger B in its own run."""
    a_rows=[r for r in csv.DictReader((a_run/'matrix.csv').open()) if r['stage']=='A']
    b_rows=[r for r in csv.DictReader((b_run/'matrix.csv').open()) if r['stage']=='B']
    if len(a_rows)!=90 or len(b_rows)!=90:raise ValueError('full 90+90 matrix required')
    def key(r):return tuple(r[k] for k in ('shape','scale_m','requested_mean_speed_m_s','feedforward'))
    cross=[];initial_differences=defaultdict(float);post_grasp_differences=defaultdict(float)
    for ar in a_rows:
        br=next(r for r in b_rows if key(r)==key(ar));af=a_run/('case%03d'%int(ar['index']));bf=b_run/('case%03d'%int(br['index']))
        with np.load(af/'reference.npz') as x,np.load(bf/'reference.npz') as y:
            if not np.array_equal(x['poses'],y['poses']) or not np.array_equal(x['times'],y['times']):cross.append(key(ar))
        for file,result in [('initial.npz',initial_differences),('trajectory.npz',post_grasp_differences)]:
            with np.load(af/file) as x,np.load(bf/file) as y:
                for field in ('q','objects'):
                    xv,yv=(x[field],y[field]) if file=='initial.npz' else (x[field][0],y[field][0])
                    result[field]=max(result[field],float(np.abs(xv-yv).max()))
    rows=[dict(r,source_run=str(a_run)) for r in a_rows]+[dict(r,source_run=str(b_run)) for r in b_rows]
    with (b_run/'qualified-matrix.csv').open('w') as f:
        fields=sorted({k for r in rows for k in r});writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    table='| 形状 | 尺度mm | 均速mm/s | A无FF P95 mm/deg | A有FF | B无FF | B有FF |\n|---|---:|---:|---|---|---|---|\n'
    grid=list(dict.fromkeys(tuple(r[k] for k in ('shape','scale_m','requested_mean_speed_m_s')) for r in a_rows))
    for shape,scale,speed in grid:
        cells=[]
        for stage,ff in [('A','False'),('A','True'),('B','False'),('B','True')]:
            r=next(r for r in rows if (r['shape'],r['scale_m'],r['requested_mean_speed_m_s'],r['stage'],r['feedforward'])==(shape,scale,speed,stage,ff))
            cells.append('{:.2f}/{:.2f} {}'.format(float(r['position_p95_mm']),float(r['orientation_p95_deg']),'通过*' if r['pass_tracking']=='True' else '中止:'+r['safety_abort'] if r['status']=='ABORT' else '未过阈值'))
        table+='| {} | {:.0f} | {:.0f} | {} |\n'.format(shape,float(scale)*1000,float(speed)*1000,' | '.join(cells))
    (b_run/'qualified-matrix.md').write_text(table)
    receipt=dict(reference_comparisons=len(a_rows),reference_mismatches=cross,initial_max_difference=dict(initial_differences),post_grasp_max_difference=dict(post_grasp_differences),
                 valid_A_run=str(a_run),valid_B_run=str(b_run),valid_cases=180,excluded_scope='r2 B is preserved fixed-finger instrumentation deviation, not qualified B')
    (b_run/'stage-join-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if cross:raise ValueError('cross-run reference mismatch')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--stage-a-run',type=Path);args=parser.parse_args();result=audit(args.run.resolve())
    if args.stage_a_run:print(json.dumps(merge_stages(args.stage_a_run.resolve(),args.run.resolve()),indent=2))
    sys.exit(bool(result['audit_failures']))
