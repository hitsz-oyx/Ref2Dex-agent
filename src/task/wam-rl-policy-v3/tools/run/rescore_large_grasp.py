"""Independent CPU replay of v9 raw states; no stage high-water scores as success."""
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
FIELDS=('q','dq','objects','hand','physical','tick','episodes','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','palm_pose')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-mutant',action='store_true');a=p.parse_args();run=a.worker.resolve();frozen=run/'runtime-code'
    os.environ.update(REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen))
    sys.path[:0]=[str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/trajectory-policy/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    from wam_rl_policy_v3.stage_evaluation import PalmSurface
    from wam_rl_policy_v3.pose_consumer import pose7
    from wam_rl_policy_v3.task_feedback import FeedbackObserver
    from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
    from wam_rl_policy_v3.measured_execution import online_support
    from wam_rl_policy_v3.physical_safety import physical_safety
    from wam_rl_policy_v3.measured_reward_bridge import reward_features
    from wam_rl_policy_v3.tau_replay_contract import pd_targets
    from ptv3_shared_wam_v6.staged_progress import StagedProgress,StagedDenseReward
    m=json.loads((run/'manifest.json').read_text());reg=json.loads((run/'registration.json').read_text());catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records'];cache={}
    palm=PalmSurface.from_urdf(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf');records=[];mutant_killed=False
    for r in m['runs']:
        folder=run/r['id'];audit=json.loads((folder/'native-audit.json').read_text())
        if r['sequence'] not in cache:
            c=next(c for c in catalog if c['sequence']==r['sequence'])
            with np.load(Path(c['source'])/'canonical.npz') as z:points=z['points'].copy()
            with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/r['sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
            cache[r['sequence']]=(points,bounds)
        points,bounds=cache[r['sequence']]
        with np.load(folder/'trajectory.npz') as z:data={k:z[k].copy() for k in FIELDS};actions=z['action'].copy();actual_pd=z['pd_targets_actual'].copy();raw=z['raw_action'].copy();adapter=z['adapter_raw_action'].copy()
        ticks=data['tick'][:,0]
        if len(actions)+1!=len(ticks) or not np.array_equal(ticks,np.arange(len(ticks))):raise ValueError('saved clock mismatch '+r['id'])
        geometry=None
        if 'geometry' in r:
            from wam_rl_policy_v3.grasp_geometry import GraspGeometry
            geometry=GraspGeometry.from_points(points)
        observer=FeedbackObserver(points);stages=StagedProgress(data['objects'][0,0,2],**({} if geometry is None else dict(geometry=geometry)));goal=pose7(data['objects'][0,0]);goal[2,3]+=.08;decision=SimpleNamespace(local_goal_pose=goal,terminal_goal_pose=goal)
        ever=False;any_abort=False;rewarder=None;reward_values=[];reward_parts=[]
        for t in range(len(ticks)):
            state={k:v[t] for k,v in data.items()};obj=state['objects'][0];pose=pose7(obj);hand=state['hand'][0,NATIVE_TO_WAM];force=state['contact_forces'][0]
            safety=physical_safety(state,0,points,bounds,audit['physical_joint_limits']);any_abort|=safety['abort']
            feedback=observer.update(pose,hand,obj[7:13],float(ticks[t])/30,contact_forces=force,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds),workspace_ok=not safety['abort'])
            features=reward_features(pose,hand,points,decision,obj[7:13],force);features=palm.features(features,pose7(state['palm_pose'][0]),pose,points)
            stage=stages.update(obj[2],float(ticks[t])/30,features,feedback,force,release_authorized=False,physical_abort=safety['abort'],**({} if geometry is None else dict(approach_tip_gap_m=geometry.paired_tip_gap(hand,pose,points))))
            error=float(np.linalg.norm(pose[:3,3]-goal[:3,3]));
            if 'ppo_update' in r:
                if t==0:rewarder=StagedDenseReward(stages)
                else:
                    d=SimpleNamespace(task_complete=bool(t==reg['horizon_controls'] and stage['current']['hold'] and error<=geometry.position_tolerance_m and not safety['abort']),failure_reason='physical' if safety['abort'] else None,terminated=bool(safety['abort']))
                    value,parts,_=rewarder.step(stages,d,actions[t-1],terminal=bool(safety['abort'] or t==reg['horizon_controls']),failed=bool(safety['abort']))
                    reward_values.append(value);reward_parts.append(parts)
            ever|=bool(stage['current']['hold'] and error<=(.015 if geometry is None else geometry.position_tolerance_m) and not safety['abort'])
        full=bool(ticks[-1]==reg['horizon_controls']);final=bool(full and not any_abort and stage['current']['hold'] and error<=(.015 if geometry is None else geometry.position_tolerance_m))
        hold=bool(full and not any_abort and stage['current']['hold']);goal15=bool(full and not any_abort and error<=.015)
        expected_pd=pd_targets(actions,data['q'][:-1,0],np.asarray(audit['controller']['offset']),np.asarray(audit['controller']['scale']));pd_error=float(np.max(np.abs(expected_pd-actual_pd))) if len(actions) else 0.
        clip=int((np.abs(raw-actions)>1e-6).any(-1).sum());upstream=int((np.abs(adapter)>1+1e-6).any(-1).sum())
        match=bool(final==r['final_success'] and ever==r['ever_success'] and hold==r['final_hold'] and goal15==r['goal_within15mm'] and stage['achieved']==r['stage_achieved'] and stage['current']==r['stage_final'] and abs(stages.longest_hold_s-r['longest_hold_s'])<1e-8 and abs(error-r['final_goal_error_m'])<1e-8 and clip==r['final_clipped_controls'] and upstream==r['adapter_clipped_controls'] and pd_error<2e-5)
        reward_match=None
        if 'ppo_update' in r:
            published_parts=json.loads((folder/'reward-parts.json').read_text())
            with np.load(folder/'ppo-rollout.npz') as z:saved_rewards=z['rewards'].copy()
            reward_match=bool(len(published_parts)==len(reward_parts)==len(actions) and np.allclose(saved_rewards,reward_values,atol=1e-9,rtol=0) and all(set(p)==set(q) and all(abs(p[k]-q[k])<1e-9 for k in p) for p,q in zip(published_parts,reward_parts)))
            match=bool(match and reward_match)
        if a.verify_mutant and not mutant_killed:
            # Flip the publication label in memory; raw states are untouched.
            corrupted=dict(r,final_success=not r['final_success']);mutant_killed=bool(final!=corrupted['final_success'])
        records.append(dict(id=r['id'],match=match,final_success=final,ever_success=ever,longest_hold_s=stages.longest_hold_s,pd_max_error=pd_error,reward_match=reward_match))
    report=dict(status='PASS' if all(r['match'] for r in records) and (not a.verify_mutant or mutant_killed) else 'FAIL',n=len(records),matched=sum(r['match'] for r in records),publication_label_mutant_killed=mutant_killed,records=records,scope='raw physical trajectory replay and separate endpoint/PD calculation; same frozen S2c/observer, no external contact ground truth')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}));return int(report['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
