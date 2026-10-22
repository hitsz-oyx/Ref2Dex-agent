"""Replay actual v10 control calls with v13 opt-in hook absent; no neural inference."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run=a.run.resolve();frozen=run/'g4-resume2/runtime-code'
    os.environ.update(REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen));sys.path[:0]=[str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    import wam_rl_policy_v3
    spec=importlib.util.spec_from_file_location('wam_rl_policy_v3.pose_consumer',ROOT/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/pose_consumer.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from wam_rl_policy_v3.pose_consumer import pose7,reduce_to_wrist_plan,grasp_intent_from_retargeter
    from wam_rl_policy_v3.grasp_repair import RepairedGraspConsumer,support_plane
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
    from wam_rl_policy_v3.measured_execution import online_support
    from wam_rl_policy_v3.stage_evaluation import PalmSurface
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];rows={}
    for gpu in (4,5,6):
        for r in json.loads((run/f'g{gpu}-resume2/manifest.json').read_text())['runs']:rows[r['id']]=(r,run/f'g{gpu}-resume2'/r['id'])
    # Fixed trace cases cover full successful hold, zero tau, recovery-close,
    # recovery-approach and below-slab reentry. Never select by replay error.
    selected=['p000-A1','p000-A2','p003-A1','p015-A1','p077-A1'];results=[]
    surface=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')
    for label in selected:
        r,folder=rows[label];audit=json.loads((folder/'native-audit.json').read_text());infos=json.loads((folder/'consumer.json').read_text());offset=np.asarray(audit['controller']['offset'],np.float32);scale=np.asarray(audit['controller']['scale'],np.float32)
        with np.load(folder/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files if k in ('q','dq','hand','objects','contact_forces','contact_force_valid','table_pose','object_contact_force','palm_pose','forecast_hand','tau_valid','adapter_raw_action','raw_action')}
        source=Path(next(c['source'] for c in catalog if c['sequence']==r['sequence']))
        with np.load(source/'canonical.npz') as z:points=z['points'].copy()
        with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/r['sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
        c=RepairedGraspConsumer(data['q'][0,0],pose7(data['palm_pose'][0,0]),offset,scale,plane=support_plane(data['table_pose'][0,0],bounds),palm_vertices=surface.vertices[::32],table_geometry=(pose7(data['table_pose'][0,0]),bounds),object_points=points);observer=FeedbackObserver(points);error=0.;phase_errors=0
        for tick in range(r['controls']):
            state={k:data[k][tick] for k in ('q','dq','hand','objects','contact_forces','contact_force_valid','table_pose','object_contact_force','palm_pose')};obj=state['objects'][0];pose=pose7(obj);palm=pose7(state['palm_pose'][0]);hand=state['hand'][0];forces=state['contact_forces'][0]
            feedback=observer.update(pose,hand[NATIVE_TO_WAM],obj[7:13],tick/30,contact_forces=forces,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds),workspace_ok=True)
            future=data['forecast_hand'][tick];plan=reduce_to_wrist_plan(hand,palm,future,pose,points) if np.any(future) else None
            intent=grasp_intent_from_retargeter(state['q'][0],palm,pose,np.asarray(infos[tick]['postgrasp_object_pose']),np.clip(data['adapter_raw_action'][tick],-1,1),offset,scale,plan=plan,tau_valid=bool(data['tau_valid'][tick]));e=observer.last_evidence
            raw,info=c.step(hand=hand,q=state['q'][0],dq=state['dq'][0],palm=palm,object_pose=pose,object_velocity=obj[7:13],forces=forces,feedback=feedback,near_fingers=e['near_fingers'],fingertip_gap_m=e['fingertip_gap_m'],intent=intent,now=tick/30)
            error=max(error,float(np.abs(raw-data['raw_action'][tick]).max()));phase_errors+=int(info['phase']!=infos[tick]['phase'])
        results.append(dict(id=label,controls=r['controls'],raw_command_max_error=error,phase_errors=phase_errors,match=bool(error<2e-5 and phase_errors==0)))
    out=dict(status='PASS' if all(r['match'] for r in results) else 'FAIL',records=results,controls=sum(r['controls'] for r in results),scope='CPU physical-state/consumer replay with stored R outputs; v13 pose module, frozen v10 repair/observer. No neural inference, no claim about timing-only branch.');a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));return int(out['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
