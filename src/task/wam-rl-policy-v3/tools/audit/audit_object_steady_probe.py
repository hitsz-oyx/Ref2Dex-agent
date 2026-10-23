"""Object-reference scoring and bounded-correction replay; no new physics."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

ROOT=Path(__file__).resolve().parents[5]


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def case(folder, baseline, steady, original_audit, decomposition):
    result=json.loads((folder/'result.json').read_text());ticks=json.loads((folder/'ticks.json').read_text());commands=json.loads((folder/'commands.json').read_text())
    manifest=json.loads((folder.parent/'manifest.json').read_text());limits=manifest['native_audit']['physical_joint_limits']
    with np.load(folder/'trajectory.npz') as z:s={k:z[k] for k in z.files}
    with np.load(folder/'reference.npz') as z:times=z['times'];poses=z['poses'];relative=z['hand_object']
    with np.load(folder/'controls.npz') as z:pd=z['pd_target'];raw=z['raw_action']
    hand=original_audit.matrix_pose(s['palm_pose'][:,0]);objects=original_audit.matrix_pose(s['objects'][:,0])
    clock=np.asarray([t['reference_time_s'] for t in ticks]);position=CubicSpline(times,poses[:,:3,3],bc_type='clamped');rotation=RotationSpline(times,Rotation.from_matrix(poses[:,:3,:3]))
    desired=np.repeat(np.eye(4)[None],len(clock),0);desired[:,:3,3]=position(clock);desired[:,:3,:3]=rotation(clock).as_matrix()
    components=decomposition.decompose(hand,objects,desired,relative)
    angles=Rotation.from_matrix(desired[:,:3,:3].transpose(0,2,1)@objects[:,:3,:3]).magnitude()
    used=clock>7.+1e-10 if steady else np.ones(len(clock),bool)
    assert used.any()
    object_stats=decomposition.stats(components['object'][used],angles[used])
    # Score only the ORIGINAL trajectory. Correction never changes this target.
    np.testing.assert_allclose(np.linalg.norm(components['object'],axis=1),[t['position_error_m'] for t in ticks],rtol=0,atol=1e-9)
    largest_mapping=largest_pd=largest_ff=0.;reasons=Counter()
    for i,c in enumerate(commands):
        correction=c['object_steady_correction'];reasons[correction['reason']]+=1
        p=np.asarray(correction['position_bias_m']);r=np.asarray(correction['rotation_bias_rad'])
        assert np.linalg.norm(p)<=.04+1e-12 and np.linalg.norm(r)<=.14+1e-12
        expected=desired[i].copy()
        if correction['accepted']:
            expected[:3,3]+=p;expected[:3,:3]=Rotation.from_rotvec(r).as_matrix()@expected[:3,:3]
        expected=expected@np.linalg.inv(relative)
        largest_mapping=max(largest_mapping,float(np.abs(expected-np.asarray(c['wrist_target'])).max()))
        np.testing.assert_allclose(c['original_wrist_target'],desired[i]@np.linalg.inv(relative),rtol=0,atol=1e-10)
        target=np.asarray(c['wrist_target']);velocity=np.asarray(c['velocity_ff_world'])
        delta=np.clip(.45*(target[:3,3]-hand[i,:3,3])-.012*s['dq'][i,0,:3]+(velocity/30 if result['feedforward'] else 0),-.025,.025)
        largest_pd=max(largest_pd,float(np.abs(pd[i,:3]-s['q'][i,0,:3]-delta).max()))
        def mapped(t):
            obj=np.eye(4);obj[:3,3]=position(t);obj[:3,:3]=rotation(t).as_matrix()
            if correction['accepted']:
                obj[:3,3]+=p;obj[:3,:3]=Rotation.from_rotvec(r).as_matrix()@obj[:3,:3]
            return obj@np.linalg.inv(relative)
        eps=1e-6;derivative=(mapped(clock[i]+eps)[:3,3]-mapped(clock[i]-eps)[:3,3])/(2*eps)
        largest_ff=max(largest_ff,float(np.abs(derivative-velocity).max()))
    assert largest_mapping<1e-10 and largest_pd<2e-6 and largest_ff<2e-5
    q=s['q'][:,0];mask=np.asarray(limits['has_limits'],bool);lo=np.asarray(limits['lower']);hi=np.asarray(limits['upper'])
    margins=np.minimum(q-lo,hi-q);margins[:,~mask]=np.inf;at=np.unravel_index(np.argmin(margins),margins.shape)
    outside=mask&((q<lo)|(q>hi));near=mask&~outside&(margins<=.10);badpd=mask&((pd<lo)|(pd>hi))
    with np.load(baseline/'trajectory.npz') as z:pairing={k:float(np.abs(z[k][0]-s[k][0]).max()) for k in ('q','dq','objects','palm_pose')}
    with np.load(baseline/'reference.npz') as z:
        reference_exact=all(np.array_equal(z[k],v) for k,v in [('times',times),('poses',poses),('hand_object',relative)])
    old=decomposition.audit_case(baseline,steady)
    old_manifest=json.loads((baseline.parent/'manifest.json').read_text())
    assert manifest['native_audit']['native_pd_properties']==old_manifest['native_audit']['native_pd_properties']
    return dict(path=str(folder),baseline=str(baseline),steady=steady,object=object_stats,baseline_object=old['object'],
                hand=decomposition.stats(components['hand_translation'][used]),grasp=decomposition.stats(components['grasp'][used]),
                controls=len(pd),status=result['status'],abort=result['abort'],pass_tracking=result['pass_tracking'],
                feedforward=result['feedforward'],initial_mismatch=pairing,reference_exact=reference_exact,
                nominal_minimum_margin=float(margins[at]),minimum_joint=int(at[1]),
                minimum_joint_name=manifest['native_audit']['native_dof_names'][int(at[1])],
                continuous_wrist_min_max_rad=np.c_[q[:,3:6].min(0),q[:,3:6].max(0)].tolist(),
                true_outside_state_ticks=int(outside.any(1).sum()),near_inside_state_ticks=int(near.any(1).sum()),
                command_outside_state_inside_controls=int((badpd&~outside[:-1]).any(1).sum()),
                correction_reasons=dict(reasons),mapping_max=largest_mapping,pd_replay_max_m=largest_pd,ff_derivative_max_m_s=largest_ff,
                clipped_controls=int((np.abs(raw)>1).any(1).sum()),collision_ticks=result['collision_ticks'],
                drift_max_m=result['drift_max_m'],drift_max_deg=result['drift_max_deg'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    audit=load('original_tracking_audit',Path(__file__).with_name('audit_object_tracking.py'))
    decomposition=load('tracking_decomposition',Path(__file__).with_name('audit_tracking_error.py'))
    ledger=json.loads((a.run/'execution.json').read_text())
    if ledger['status']!='COMPLETED':raise ValueError('not a completed physical probe: '+ledger['status'])
    baseline=ROOT/'outputs/wam-rl-policy-v3';rows=[];checks=[]
    run=a.run/'static';checks.append(audit.audit(run))
    for i in range(3):rows.append(case(run/('case%03d'%i),baseline/'tracking-error-decomposition-20261023-r1/paired-static'/('case%03d'%(i*2)),True,audit,decomposition))
    for i in range(12,18):
        run=a.run/('dynamic-%03d'%i);checks.append(audit.audit(run))
        rows.append(case(run/'case000',baseline/'object-trajectory-tracking-20261023-r4'/('case%03d'%i),False,audit,decomposition))
    assert all(not x['audit_failures'] for x in checks), 'independent raw-state audit failed'
    fits=[]
    for ff in (False,True):
        dynamic=[r for r in rows if not r['steady'] and r['feedforward']==ff]
        for field in ('object','baseline_object'):
            zero=float(np.mean([r[field]['p95_mm'] for r in rows if r['steady']]))
            values=[(0.,zero)]+[(v,r[field]['p95_mm']) for v,r in zip((.03,.10,.25),dynamic)]
            fits.append(dict(feedforward=ff,condition=field,**decomposition.fit(values)))
    summary=dict(status='AUDITED',rows=rows,fits=fits,existing_independent_audit=checks,
                 scope='fixed endpoint and six frozen straight trajectories; not the full 90-cell grid',
                 self_collision='unobservable: native self-collision filtering remains disabled')
    (a.run/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
