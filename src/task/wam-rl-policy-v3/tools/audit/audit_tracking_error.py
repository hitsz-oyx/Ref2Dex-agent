"""CPU-only SE3 error decomposition on frozen traces; never runs physics."""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

ROOT=Path(__file__).resolve().parents[5]

def matrices(x):
    out=np.repeat(np.eye(4)[None],len(x),axis=0)
    out[:,:3,3]=x[:,:3];out[:,:3,:3]=Rotation.from_quat(x[:,3:7]).as_matrix()
    return out

def decompose(hand,obj,desired,relative):
    target=desired@np.linalg.inv(relative)
    proxy=hand@relative
    translation=hand[:,:3,3]-target[:,:3,3]
    lever=np.einsum('nij,j->ni',hand[:,:3,:3]-target[:,:3,:3],relative[:3,3])
    slip=obj[:,:3,3]-proxy[:,:3,3]
    error=obj[:,:3,3]-desired[:,:3,3]
    closure=error-translation-lever-slip
    return dict(hand_translation=translation,rotation_lever=lever,grasp=slip,object=error,
                closure=closure,target=target,proxy=proxy)

def stats(vector,orientation=None):
    vector=np.asarray(vector);norm=np.linalg.norm(vector,axis=-1);power=float((vector*vector).sum())
    out=dict(samples=len(vector),mean_xyz_mm=(vector.mean(0)*1000).tolist(),
             std_xyz_mm=(vector.std(0)*1000).tolist(),rms_mm=float(np.sqrt((norm*norm).mean())*1000),
             mean_norm_mm=float(norm.mean()*1000),p50_mm=float(np.quantile(norm,.5)*1000),
             p95_mm=float(np.quantile(norm,.95)*1000),max_mm=float(norm.max()*1000),
             vertical_power_fraction=float((vector[:,2]**2).sum()/power) if power else 0.,
             horizontal_rms_mm=float(np.sqrt((vector[:,:2]**2).sum(1).mean())*1000))
    if orientation is not None:
        out['orientation_p95_deg']=float(np.rad2deg(np.quantile(orientation,.95)))
        out['orientation_mean_deg']=float(np.rad2deg(np.mean(orientation)))
    return out

def angles(a,b):
    return Rotation.from_matrix(a.transpose(0,2,1)@b).magnitude()

def check_recorded(recorded,expected,tolerance,label):
    residual=float(np.max(np.abs(np.asarray(recorded)-np.asarray(expected))))
    assert residual<tolerance,label
    return residual

def audit_case(folder,steady=False,save_ticks=None):
    folder=folder.resolve()
    meta=json.loads((folder/'result.json').read_text())
    tick=json.loads((folder/'ticks.json').read_text())
    with np.load(folder/'reference.npz') as z:time=z['times'];poses=z['poses'];relative=z['hand_object']
    clock=np.asarray([t.get('reference_time_s',min(t['time_s'],time[-1])) for t in tick])
    position=CubicSpline(time,poses[:,:3,3],bc_type='clamped')
    rotation=RotationSpline(time,Rotation.from_matrix(poses[:,:3,:3]))
    desired=np.repeat(np.eye(4)[None],len(clock),axis=0);desired[:,:3,3]=position(clock);desired[:,:3,:3]=rotation(clock).as_matrix()
    with np.load(folder/'trajectory.npz') as z:
        hand=matrices(z['palm_pose'][:,0]);obj=matrices(z['objects'][:,0]);q=z['q'][:,0];dq=z['dq'][:,0]
        object_force=np.linalg.norm(z['object_contact_force'][:,0],axis=-1)
        finger_force=np.linalg.norm(z['contact_forces'][:,0],axis=-1)
    components=decompose(hand,obj,desired,relative);target=components['target']
    map_closure=target@relative-desired
    hand_angle=angles(target[:,:3,:3],hand[:,:3,:3]);obj_angle=angles(desired[:,:3,:3],obj[:,:3,:3])
    with np.load(folder/'controls.npz') as z:pd=z['pd_target'];raw=z['raw_action']
    commands=json.loads((folder/'commands.json').read_text())
    command_targets=np.asarray([c['wrist_target'] for c in commands])
    assert len(pd)==len(clock)-1
    check_recorded(command_targets,target[:-1],1e-10,'wrong reference frame/time')
    assert np.max(np.abs(components['closure']))<1e-10,'SE3 decomposition inconsistent'
    assert np.max(np.abs(map_closure))<1e-10,'reference round trip broken'
    check_recorded(np.linalg.norm(components['object'],axis=1),[t['position_error_m'] for t in tick],1e-9,'object scorer time/pose mismatch')
    check_recorded(obj_angle,[t['orientation_error_rad'] for t in tick],1e-9,'SO3 scorer mismatch')
    object_omega=np.einsum('nij,nj->ni',desired[:,:3,:3],rotation(clock,1))
    expected_velocity=position(clock,1)+np.cross(object_omega,target[:,:3,3]-desired[:,:3,3])
    velocity_residual=check_recorded([c['velocity_ff_world'] for c in commands],expected_velocity[:-1],1e-9,'missing/wrong angular lever arm')
    mount=Rotation.from_euler('XYZ',q[0,3:6]).as_matrix().T@hand[0,:3,:3]
    predicted_R=Rotation.from_euler('XYZ',q[:,3:6]).as_matrix()@mount
    fk_error=angles(predicted_R,hand[:,:3,:3])
    used=clock>7.0+1e-10 if steady else np.ones(len(clock),bool)
    assert used.sum()>0
    row=dict(path=str(folder.relative_to(ROOT)),stage=meta['stage'],feedforward=meta['feedforward'],
             repeat=meta.get('repeat'),shape=meta['shape'],scale_m=meta['scale_m'],speed=meta['requested_mean_speed_m_s'],
             duration_s=float(clock[-1]),status=meta['status'],controls=len(pd),
             hand=stats(components['hand_translation'][used],hand_angle[used]),
             object=stats(components['object'][used],obj_angle[used]),
             lever=stats(components['rotation_lever'][used]),grasp=stats(components['grasp'][used]),
             map_roundtrip_max=float(np.max(np.abs(map_closure))),
             decomposition_max_m=float(np.max(np.abs(components['closure']))),
             command_target_max=float(np.max(np.abs(command_targets-target[:-1]))),
             palm_q_position_max_m=float(np.max(np.abs(hand[:,:3,3]-q[:,:3]))),
             wrist_fk_orientation_max_deg=float(np.rad2deg(fk_error.max())),
             command_velocity_max_m_s=velocity_residual,
             object_force_max_N=float(object_force[used].max()),
             object_force_after_first_max_N=float(object_force[1:].max()),
             finger_force_max_N=float(finger_force[used].max()),
             contact_ticks=int(sum(t['contact'] for t,u in zip(tick,used) if u)),
             collision_ticks=int(sum(bool(t['collision_pairs']) for t,u in zip(tick,used) if u)))
    # Recompute PD-consumer increments, including Euler wrapping as production does.
    # Native PD is incremental about q at t, not the complete Cartesian reference.
    actual_delta=pd[:,:3]-q[:-1,:3]
    desired_p_error=target[:-1,:3,3]-hand[:-1,:3,3]
    damping=-.012*dq[:-1,:3]
    ff=np.array([c['velocity_ff_world'] for c in commands])/30 if meta['feedforward'] else np.zeros_like(damping)
    expected=np.clip(.45*desired_p_error+damping+ff,-.025,.025)
    pd_error=float(np.abs(actual_delta-expected).max())
    assert pd_error<2e-6,'incremental PD consumption mismatch'
    row['pd_increment_replay_max_m']=pd_error
    cuse=used[:-1];fd=np.diff(hand[:,:3,3],axis=0)*30
    row['native_dq_xyz_mean_m_s']=dq[:-1,:3][cuse].mean(0).tolist()
    row['finite_difference_xyz_mean_m_s']=fd[cuse].mean(0).tolist()
    row['native_dq_vs_fd_rms_m_s']=float(np.sqrt(((dq[:-1,:3][cuse]-fd[cuse])**2).sum(1).mean()))
    row['pd_target_minus_q_mean_mm']=(actual_delta[cuse].mean(0)*1000).tolist()
    row['damping_mean_mm']=(damping[cuse].mean(0)*1000).tolist()
    row['position_gain_term_mean_mm']=(.45*desired_p_error[cuse].mean(0)*1000).tolist()
    row['pd_cap_ticks']=int((np.abs(expected)>=.025-1e-8).any(1).sum())
    row['actual_mean_speed_m_s']=float(meta.get('arc_length_m',0.)/clock[-1])
    if not steady:
        row['actual_mean_speed_m_s']=float(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1).sum()/clock[-1])
    # Mapping derivative independently checked by differentiating the hand pose.
    sample=np.linspace(time[0]+1e-4,time[-1]-1e-4,101);eps=1e-6
    R=rotation(sample).as_matrix();omega=np.einsum('nij,nj->ni',R,rotation(sample,1))
    hand_offset=-relative[:3,:3].T@relative[:3,3]
    lever_world=np.einsum('nij,j->ni',R,hand_offset)
    analytic=position(sample,1)+np.cross(omega,lever_world)
    def hp(t):return position(t)+np.einsum('nij,j->ni',rotation(t).as_matrix(),hand_offset)
    fd_v=(hp(sample+eps)-hp(sample-eps))/(2*eps)
    row['hand_velocity_fd_max_m_s']=float(np.max(np.abs(analytic-fd_v)))
    row['omitting_lever_velocity_max_m_s']=float(np.max(np.linalg.norm(np.cross(omega,lever_world),axis=1)))
    if save_ticks:
        np.savez_compressed(save_ticks,clock=clock,used=used,**{k:components[k] for k in ('object','hand_translation','rotation_lever','grasp','closure')},hand_angle=hand_angle,object_angle=obj_angle)
    return row

def fit(points):
    x=np.asarray([p[0] for p in points]);y=np.asarray([p[1] for p in points]);beta=np.polyfit(x,y,1);pred=np.polyval(beta,x)
    total=((y-y.mean())**2).sum()
    return dict(n=len(x),points=np.c_[x,y].tolist(),intercept_mm=float(beta[1]),slope_mm_per_m_s=float(beta[0]),r_squared=float(1-((y-pred)**2).sum()/total) if total else None)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--static-run',type=Path);parser.add_argument('--paired-run',type=Path)
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for stage,suffix in [('A','r2'),('B','r4')]:
        run=ROOT/'outputs/wam-rl-policy-v3'/('object-trajectory-tracking-20261023-'+suffix)
        manifest=json.loads((run/'manifest.json').read_text())
        for meta in manifest['results']:
            if meta['stage']!=stage:continue
            rows.append(audit_case(run/('case%03d'%meta['index']),save_ticks=out/('dynamic-%s%03d-tick-decomposition.npz'%(stage,meta['index']))))
    summary={};group=defaultdict(list)
    for r in rows:group[str((r['stage'],r['feedforward']))].append(r)
    for key,rs in group.items():
        summary[key]=dict(cases=len(rs),**{kind+'_p95_median_mm':float(np.median([r[kind]['p95_mm'] for r in rs])) for kind in ('object','hand','lever','grasp')},
            vertical_fraction_median=float(np.median([r['object']['vertical_power_fraction'] for r in rs])),
            object_rms_median_mm=float(np.median([r['object']['rms_mm'] for r in rs])),
            hand_orientation_p95_median_deg=float(np.median([r['hand']['orientation_p95_deg'] for r in rs])))
    static=[]
    if a.static_run:
        m=json.loads((a.static_run/'manifest.json').read_text());assert m['status']=='COMPLETED',m['status']
        for r in m['results']:
            static.append(audit_case(a.static_run/('case%03d'%r['index']),True,out/('static%03d-tick-decomposition.npz'%r['index'])))
    paired=[];pairing=[]
    if a.paired_run:
        m=json.loads((a.paired_run/'manifest.json').read_text());assert m['status']=='COMPLETED',m['status']
        for r in m['results']:
            paired.append(audit_case(a.paired_run/('case%03d'%r['index']),True,out/('paired%03d-tick-decomposition.npz'%r['index'])))
        for i in range(0,len(paired),2):
            before=a.paired_run/('case%03d'%i);after=a.paired_run/('case%03d'%(i+1))
            with np.load(before/'trajectory.npz') as z,np.load(after/'trajectory.npz') as y:
                mismatch={k:float(np.abs(z[k][-1]-y[k][0]).max()) for k in ('q','dq','objects','palm_pose')}
                assert all(v==0. for v in mismatch.values()),'paired load-switch state changed'
            with np.load(before/'reference.npz') as z,np.load(after/'reference.npz') as y:
                assert all(np.array_equal(z[k],y[k]) for k in ('times','poses','hand_object')),'paired reference/transform changed'
            h=json.loads((after/'handoff.json').read_text())
            with np.load(before/'controls.npz') as z:
                assert np.array_equal(z['pd_target'][-1],np.asarray(h['last_actual_pd'])),'paired finger preload changed'
            pairing.append(dict(pair=i//2,state_mismatch=mismatch,reference_exact=True,preload_exact=True,
                                loaded_p95_mm=paired[i]['object']['p95_mm'],unloaded_p95_mm=paired[i+1]['object']['p95_mm'],
                                signed_load_effect_xyz_mm=(np.asarray(paired[i]['object']['mean_xyz_mm'])-paired[i+1]['object']['mean_xyz_mm']).tolist()))
    regressions=[]
    for stage in ('A','B'):
        for ff in (False,True):
            for scale in (.02,.05,.10):
                selected=[r for r in rows if r['shape']=='straight' and r['stage']==stage and r['feedforward']==ff and r['scale_m']==scale]
                points=[(r['actual_mean_speed_m_s'],r['object']['p95_mm']) for r in selected]
                regressions.append(dict(stage=stage,feedforward=ff,scale_m=scale,metric='dynamic_object_position_p95_mm',**fit(points)))
                if scale==.10 and static:
                    sr=[r for r in static if r['stage']==stage]
                    # Mean of three fixed-pose repeats is the pre-registered v=0 datum.
                    z=float(np.mean([r['object']['p95_mm'] for r in sr]))
                    regressions.append(dict(stage=stage,feedforward=ff,scale_m=scale,metric='v0_tail_p95_plus_dynamic_full_p95_mm',zero_from='unloaded static A, not legacy loaded A' if stage=='A' else 'real static B',**fit([(0.,z)]+points)))
    matched_regressions=[];matched_initial=None
    if paired:
        # Post hoc contract sensitivity: first B is the only zero-speed state
        # proven bitwise identical to the legacy dynamic reference acquisition.
        old=ROOT/'outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r4/case012'
        new=a.paired_run/'case000'
        with np.load(old/'trajectory.npz') as z,np.load(new/'trajectory.npz') as y:
            matched_initial={k:float(np.abs(z[k][0]-y[k][0]).max()) for k in ('q','dq','objects','palm_pose')}
            assert all(v==0 for v in matched_initial.values()),'legacy/static first B initial state differs'
        with np.load(old/'reference.npz') as z,np.load(new/'reference.npz') as y:
            assert np.array_equal(z['hand_object'],y['hand_object'])
        for ff in (False,True):
            rs=[r for r in rows if r['stage']=='B' and r['feedforward']==ff and r['shape']=='straight' and r['scale_m']==.10]
            matched_regressions.append(dict(stage='B',feedforward=ff,scope='post hoc matched-initial sensitivity; static FF derivative zero',**fit([(0.,paired[0]['object']['p95_mm'])]+[(r['actual_mean_speed_m_s'],r['object']['p95_mm']) for r in rs])))
    load_validation=[]
    legacy_native=json.loads((ROOT/'outputs/wam-rl-policy-v3/object-trajectory-tracking-20261023-r4/manifest.json').read_text())['native_audit']
    for run,rs in ((a.static_run,static),(a.paired_run,paired)):
        if run is None:continue
        native=json.loads((run/'physics/manifest.json').read_text())
        assert native['native_pd_properties']==legacy_native['native_pd_properties'],'PD gains/properties changed'
        for change in native['tracking_load_probe_changes']:
            expected_flags=[x|1 if change['enabled'] else x for x in native['tracking_original_object_flags']]
            expected_filters=[0x7fffffff if change['enabled'] else x for x in native['tracking_original_object_filters']]
            assert change['flags']==expected_flags and change['filters']==expected_filters,'load flag/filter readback failed'
        unloaded=[r for r in rs if r['stage']=='A']
        assert all(r['object_force_after_first_max_N']==0 and r['finger_force_max_N']==0 for r in unloaded),'unloaded probe still carrying contact force'
        load_validation.append(dict(run=str(run),unloaded_cases=len(unloaded),object_force_after_first_max_N=0,pd_properties_exact_legacy=True,readback_changes=len(native['tracking_load_probe_changes'])))
    all_rows=rows+static+paired
    result=dict(dynamic_rows=rows,group_summary=summary,static_rows=static,paired_rows=paired,pairing=pairing,regressions=regressions,load_validation=load_validation,
                matched_initial_mismatch=matched_initial,matched_regressions=matched_regressions,
                largest_map_roundtrip=max(r['map_roundtrip_max'] for r in all_rows),
                largest_decomposition_residual_m=max(r['decomposition_max_m'] for r in all_rows),
                largest_command_reference_residual=max(r['command_target_max'] for r in all_rows),
                largest_pd_replay_residual_m=max(r['pd_increment_replay_max_m'] for r in all_rows),
                largest_derivative_residual_m_s=max(r['hand_velocity_fd_max_m_s'] for r in all_rows))
    (out/'decomposition.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (out/'dynamic-cells.csv').open('w') as f:
        fields=['path','stage','feedforward','shape','scale_m','speed','actual_mean_speed_m_s','object_p95_mm','hand_p95_mm','lever_p95_mm','grasp_p95_mm','vertical_fraction']
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for r in rows:
            writer.writerow(dict({k:r[k] for k in fields[:7]},**{kind+'_p95_mm':r[kind]['p95_mm'] for kind in ('object','hand','lever','grasp')},vertical_fraction=r['object']['vertical_power_fraction']))
    print(json.dumps({k:v for k,v in result.items() if k not in ('dynamic_rows','static_rows','paired_rows')},indent=2))

if __name__=='__main__':main()
