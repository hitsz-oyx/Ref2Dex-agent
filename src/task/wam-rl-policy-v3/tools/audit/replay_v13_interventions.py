"""Independent output-field selection and physical control replay; stored R, no neural inference."""
import argparse,json,os,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.worker.resolve();frozen=run/'runtime-code'
    os.environ.update(REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen));sys.path[:0]=[str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    from wam_rl_policy_v3.pose_consumer import pose7,reduce_to_wrist_plan,grasp_intent_from_retargeter
    from wam_rl_policy_v3.grasp_repair import RepairedGraspConsumer,support_plane
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
    from wam_rl_policy_v3.measured_execution import online_support
    from wam_rl_policy_v3.stage_evaluation import PalmSurface
    from wam_rl_policy_v3.evaluation_sources import freeze_template,template_intent
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];m=json.loads((run/'manifest.json').read_text());reg=json.loads((run/'registration.json').read_text());results=[];mutant_killed=False
    with np.load(ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz') as z:template=freeze_template({k:z[k] for k in z.files},30,50)
    surface=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')
    if m['learning_updates']!=0:raise ValueError('frozen inference only')
    for r in m['runs']:
        folder=run/r['id'];audit=json.loads((folder/'native-audit.json').read_text());infos=json.loads((folder/'consumer.json').read_text());offset=np.asarray(audit['controller']['offset'],np.float32);scale=np.asarray(audit['controller']['scale'],np.float32)
        with np.load(folder/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files if k in ('q','dq','hand','objects','contact_forces','contact_force_valid','table_pose','object_contact_force','palm_pose','forecast_hand','tau_valid','adapter_raw_action','raw_action','tau')}
        source=Path(next(c['source'] for c in catalog if c['sequence']==r['sequence']))
        with np.load(source/'canonical.npz') as z:points=z['points'].copy()
        with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/r['sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
        job=next(j for j in reg['jobs'] if j['id']==r['trial'])
        with np.load(job['reference']) as z:reference={k:z[k].copy() for k in z.files}
        rotations=np.stack([pose7(v)[:3,:3] for v in reference['objects'][:,0]]);local=np.einsum('tki,tij->tkj',reference['hand'][:,0]-reference['objects'][:,0,None,:3],rotations);cursor=0
        forces=np.linalg.norm(reference['contact_forces'][:,0],axis=-1);contact=int(np.flatnonzero(np.asarray(reference['contact_force_valid']).reshape(len(forces),-1)[:,0]&((forces>=.25).sum(-1)>=2))[0]);rigid=[0,1,3,5,7,9]
        phase_cap=contact;full_native_phase=bool(r['arm']=='A4' and 'gt_phase_cap_frame' in infos[0])
        if full_native_phase:
            source_observer=FeedbackObserver(points);phase_cap=None
            for st,so in enumerate(reference['objects'][:,0]):
                sf=source_observer.update(pose7(so),reference['hand'][st,0,NATIVE_TO_WAM],so[7:13],st/30,contact_forces=reference['contact_forces'][st,0],force_valid=bool(reference['contact_force_valid'][st,0]),workspace_ok=True)
                if sf.force_valid and sf.contact_observed and sf.grasp_established:phase_cap=st;break
            if phase_cap is None:raise ValueError('GT reference lacks measured acquisition')
        c=RepairedGraspConsumer(data['q'][0,0],pose7(data['palm_pose'][0,0]),offset,scale,plane=support_plane(data['table_pose'][0,0],bounds),palm_vertices=surface.vertices[::32],table_geometry=(pose7(data['table_pose'][0,0]),bounds),object_points=points);observer=FeedbackObserver(points);command_error=keypose_error=future_error=tau_error=0.;phase_errors=0;deadline=None
        if r['arm']=='TIME':c.predicted_close_at=float('inf')
        for tick in range(r['controls']):
            state={k:data[k][tick] for k in ('q','dq','hand','objects','contact_forces','contact_force_valid','table_pose','object_contact_force','palm_pose')};obj=state['objects'][0];pose=pose7(obj);palm=pose7(state['palm_pose'][0]);hand=state['hand'][0];force=state['contact_forces'][0]
            feedback=observer.update(pose,hand[NATIVE_TO_WAM],obj[7:13],tick/30,contact_forces=force,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds),workspace_ok=True)
            future=data['forecast_hand'][tick];has_future=bool(np.any(future));plan=reduce_to_wrist_plan(hand,palm,future,pose,points) if has_future else None;valid=bool(data['tau_valid'][tick])
            expected_tau=future[:8]-hand if has_future else np.zeros((8,11,3));tau_error=max(tau_error,float(np.abs(data['tau'][tick]-expected_tau).max()))
            if r['arm'] in ('A3','A4') and valid:
                if r['arm']=='A3':indices=tick+np.arange(1,25)
                else:
                    lh=(hand-pose[:3,3])@pose[:3,:3];errors=np.mean(np.linalg.norm(local[cursor:phase_cap+1]-lh,axis=-1),axis=-1) if full_native_phase else np.mean(np.linalg.norm(local[cursor:contact+1,rigid]-lh[rigid],axis=-1),axis=-1);cursor+=int(np.argmin(errors));indices=np.minimum(cursor+np.arange(1,25),len(local)-1)
                expected=local[indices]@pose[:3,:3].T+pose[:3,3];future_error=max(future_error,float(np.abs(expected-future).max()))
            if full_native_phase and infos[tick]['gt_phase_cap_frame']!=phase_cap:raise ValueError('GT acquisition cap mismatch')
            if r['arm']=='A4' and (cursor!=infos[tick]['gt_cursor'] or contact!=infos[tick]['gt_contact_frame']):raise ValueError('closed-loop GT source phase mismatch')
            intent=grasp_intent_from_retargeter(state['q'][0],palm,pose,np.asarray(infos[tick]['postgrasp_object_pose']),np.clip(data['adapter_raw_action'][tick],-1,1),offset,scale,plan=plan,tau_valid=valid)
            if r['arm'] in ('WRIST','FINGER','TIME','T0'):
                fixed=template_intent(template,state['q'][0],palm,pose,intent.postgrasp_object_pose,c.phase)
                wrist=intent if r['arm']=='WRIST' else fixed;fingers=intent if r['arm']=='FINGER' else fixed
                intent=replace(intent,approach_pose_object=wrist.approach_pose_object.copy(),grasp_pose_object=wrist.grasp_pose_object.copy(),approach_joint_angles=wrist.approach_joint_angles.copy(),preshape_q=fingers.preshape_q.copy(),closure_q=fingers.closure_q.copy(),mode='component-'+r['arm'])
            if r['arm']=='TIME':
                if deadline is None and plan is not None and plan.contact_index is not None:deadline=(tick+plan.contact_index+1)/30;c.predicted_close_at=deadline
                logged=infos[tick]['predicted_close_at_s']
                if (deadline is None)!=(logged is None) or (deadline is not None and abs(deadline-logged)>1e-10):raise ValueError('predicted timing latch mismatch')
            error=float(np.abs(intent.grasp_pose_object-np.asarray(infos[tick]['grasp_candidate_object'])).max());keypose_error=max(keypose_error,error)
            if not mutant_killed:
                altered=np.asarray(infos[tick]['grasp_candidate_object']).copy();altered[0,3]+=.01;mutant_killed=bool(np.abs(intent.grasp_pose_object-altered).max()>1e-6)
            if not np.array_equal(data['adapter_raw_action'][tick],np.asarray(infos[tick]['unmodified_R_action'],dtype=np.float32)):raise ValueError('R output logging mismatch')
            e=observer.last_evidence;raw,info=c.step(hand=hand,q=state['q'][0],dq=state['dq'][0],palm=palm,object_pose=pose,object_velocity=obj[7:13],forces=force,feedback=feedback,near_fingers=e['near_fingers'],fingertip_gap_m=e['fingertip_gap_m'],intent=intent,now=tick/30)
            command_error=max(command_error,float(np.abs(raw-data['raw_action'][tick]).max()));phase_errors+=int(info['phase']!=infos[tick]['phase'])
        nonzero=bool(np.any(data['adapter_raw_action']));match=bool(command_error<2e-5 and keypose_error<1e-6 and future_error<2e-6 and tau_error<2e-6 and phase_errors==0 and r['actor_calls']==r['controls'] and nonzero)
        results.append(dict(id=r['id'],controls=r['controls'],command_max_error=command_error,keypose_max_error=keypose_error,future_max_error=future_error,tau_max_error=tau_error,phase_errors=phase_errors,R_called_each_control=r['actor_calls']==r['controls'],R_nonzero=nonzero,match=match))
    out=dict(status='PASS' if results and all(r['match'] for r in results) and mutant_killed else 'FAIL',n=len(results),controls=sum(r['controls'] for r in results),keypose_metadata_mutant_killed=mutant_killed,records=results,scope='Independent component field selection and GT local-future reconstruction; replay frozen physical consumer with recorded R. CPU only, no neural inference; does not validate WAM model quality.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'}));return int(out['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
