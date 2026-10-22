"""Independently score S2d saved states and challenge the live-object GT mapping."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/audit')]
from audit_first_tau_loss import load, physical_rows
from wam_rl_policy_v3.tau_replay_contract import bouts, pd_targets, require_saved_clock
from wam_rl_policy_v3.inspire_contract import expand_action
from measure_s2d_controls import object_anchored_targets, object_anchored_tau


def pose(value):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(value[3:7]).as_matrix()
    matrix[:3, 3] = value[:3]
    return matrix


def challenge(hand, objects):
    """Counterfactual hand/object changes must have different causal effects."""
    t = 40
    live_obj = objects[t].copy()
    live_hand = hand[t].copy()
    target = object_anchored_targets(hand, objects, t, live_obj)
    tau, _ = object_anchored_tau(hand, objects, t, live_hand, live_obj)
    perturbed_hand = live_hand + np.array([.17, -.11, .09])
    other_tau, _ = object_anchored_tau(hand, objects, t, perturbed_hand, live_obj)
    reconstruction_error = float(np.abs(other_tau+perturbed_hand-target).max())
    # Correct delta changes oppositely to measured hand while absolute goals stay fixed.
    delta_error = float(np.abs((other_tau-tau)+(perturbed_hand-live_hand)).max())
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.2, -.4, .7]).as_matrix()
    transform[:3, 3] = [.4, -.2, .3]
    changed_pose = transform @ pose(live_obj)
    changed_obj = np.r_[changed_pose[:3, 3], Rotation.from_matrix(changed_pose[:3, :3]).as_quat()]
    moved = object_anchored_targets(hand, objects, t, changed_obj)
    expected = target @ transform[:3, :3].T + transform[:3, 3]
    object_equivariance_error = float(np.abs(moved-expected).max())
    aligned_error = float(np.abs(target-hand[t+np.arange(1,9)]).max())
    # A fake tau=0/live-hand target must fail the independent reference-target oracle.
    fake_goal_error = float(np.abs(live_hand[None]-target).max())
    result = dict(hand_perturbation_reconstructed_target_max_error=reconstruction_error,
                  hand_perturbation_delta_max_error=delta_error,
                  object_SE3_equivariance_max_error=object_equivariance_error,
                  aligned_reference_future_max_error=aligned_error,
                  fake_live_hand_target_error_m=fake_goal_error,
                  fake_live_hand_target_rejected=fake_goal_error>1e-4)
    if max(reconstruction_error, delta_error)>1e-6 or max(object_equivariance_error, aligned_error)>1e-9 or fake_goal_error<=1e-4:
        raise ValueError('GT counterfactual failed: '+str(result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    m=json.loads((args.run/'manifest.json').read_text())
    if m['status']!='COMPLETED' or len(m['runs'])!=10:
        raise ValueError('completed 2x5 run required')
    original=json.loads((Path(m['original_run'])/'manifest.json').read_text())
    points=load(Path(original['source_pd_reference'])/'canonical.npz')['points']
    bounds=load(Path(m['asset_root'])/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/m['source_sequence']/'admission.npz')['table_bounds']
    ref=load(Path(m['complete_reference']))
    hand=ref['hand'][:,0]; objects=ref['objects'][:,0]
    offset=np.asarray(m['native_initial_audit']['controller']['offset'])
    scale=np.asarray(m['native_initial_audit']['controller']['scale'])
    records=[]
    for online in m['runs']:
        d=load(args.run/online['label']/'trajectory.npz')
        require_saved_clock(d)
        rows=physical_rows(d,points,bounds,m['native_initial_audit']['physical_joint_limits'])
        spans=bouts([r['held'] for r in rows])
        longest=max(((hi-lo-1)/30 for lo,hi in spans),default=0.)
        ever=any(hi-lo>=31 for lo,hi in spans)
        final=bool(spans and spans[-1][1]==len(rows) and spans[-1][1]-spans[-1][0]>=31)
        height=d['objects'][:,0,2]-d['objects'][0,0,2]
        reached=np.flatnonzero(height>=.05)
        drop=next((i for i in range(int(reached[0])+1,len(height)) if height[i]<.05),None) if len(reached) else None
        held_drop=next((r['tick'] for r in rows[spans[0][0]+1:] if 'lift_below5cm' in r['failed_predicates']),None) if spans else None
        n=len(d['action'])
        pd_error=float(np.abs(pd_targets(d['action'],d['q'][:n,0],offset,scale)-d['pd_targets_actual']).max())
        clip_count=int((np.abs(d['raw_action']-d['action'])>1e-6).any(axis=-1).sum())
        clips_error=float(np.abs(np.clip(d['raw_action'],-1,1)-d['action']).max())
        expanded_error=float(np.abs(expand_action(d['action'])-d['expanded_action']).max())
        packed_error=float(np.abs(d['observation'][:,421:685]-d['tau'].reshape(len(rows),-1)).max())
        valid_ok=np.array_equal(d['observation'][:,685],d['tau_valid'].astype(np.float32))
        episodes_ok=np.array_equal(d['episodes_input'],d['episodes'][:n,0]) and (np.diff(d['episodes'][:,0])==0).all()
        gt_error=0.; gt_valid_ticks=[]; relative_correction=[]
        if online['chain']=='c-live-gt':
            # Independent homogeneous-matrix reconstruction, not runner helper reuse.
            for t in range(len(rows)):
                live=d['objects'][t,0]
                # Exact production availability (also checks CAD) is read separately.
                import importlib.util
                if t==0:
                    spec=importlib.util.spec_from_file_location('s2d_domain',args.run/'runtime-code/src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/execution_contract.py')
                    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                from wam_rl_policy_v3.inspire_contract import NATIVE_TO_WAM
                available=bool(t>=3 and module.encoder_domain_available(points,d['hand'][t,0,NATIVE_TO_WAM],pose(live)))
                expected_valid=bool(available and t+8<len(hand))
                if bool(d['tau_valid'][t])!=expected_valid:raise ValueError('GT validity mismatch')
                expected=np.zeros((8,11,3),np.float32)
                if expected_valid:
                    transform=pose(live) @ np.linalg.inv(pose(objects[t]))
                    target=hand[t+np.arange(1,9)] @ transform[:3,:3].T+transform[:3,3]
                    expected=(target-d['hand'][t,0]).astype(np.float32)
                    gt_valid_ticks.append(t)
                    relative_correction.append(float(np.linalg.norm(expected-(hand[t+np.arange(1,9)]-hand[t]),axis=-1).mean()))
                gt_error=max(gt_error,float(np.abs(expected-d['tau'][t]).max()))
        safety=next((r for r in rows if any(v not in ('lift_below5cm','grasp_not_established','grasp_lost','force_invalid','fewer_than_two_force_contacts','table_supported') for v in r['failed_predicates'])),None)
        if (ever,final)!=(online['ever_success'],online['final_success']) or abs(longest-online['longest_hold_elapsed_s'])>1e-9:
            raise ValueError('independent physical verdict mismatch')
        if clip_count!=online['clipped_controls'] or pd_error>2e-5 or gt_error>1e-6 or max(clips_error,expanded_error,packed_error)>0 or not valid_ok or not episodes_ok:
            raise ValueError('saved action/tau/PD contract mismatch')
        records.append(dict(**online,independent_longest_hold_s=longest,first_lift5cm_tick=int(reached[0]) if len(reached) else None,
            first_height_failure_tick=drop,first_height_failure_after_held_tick=held_drop,
            first_height_failure_detail=None if drop is None else rows[drop],
            height_failure_note='null means never reached5cm, or no subsequent height loss; see first_lift5cm_tick',
            hold_intervals=[dict(first_tick=lo,last_tick=hi-1,elapsed_s=(hi-lo-1)/30) for lo,hi in spans],
            first_safety_state=safety,pd_error=pd_error,gt_reconstruction_max_error=gt_error,
            gt_valid_state_ticks=gt_valid_ticks,mean_gt_correction_m=float(np.mean(relative_correction)) if relative_correction else None))
    chains={}
    for name in ('a-original-wam','c-live-gt'):
        r=sorted([r for r in records if r['chain']==name],key=lambda r:r['block'])
        if [v['block'] for v in r]!=list(range(5)):raise ValueError('five blocks required')
        chains[name]=dict(n=5,longest_hold_s=[v['independent_longest_hold_s'] for v in r],
            ever_successes=sum(v['ever_success'] for v in r),final_successes=sum(v['final_success'] for v in r),
            clipped_controls=sum(v['clipped_controls'] for v in r),controls=sum(v['controls'] for v in r),
            safety_aborts=sum(v['physical_safety_abort'] is not None for v in r),
            first_height_failure_ticks=[v['first_height_failure_tick'] for v in r],
            first_lift5cm_ticks=[v['first_lift5cm_tick'] for v in r])
    result=dict(status='PASS',run_id=m['run_id'],experiment_id=m['experiment_id'],git_commit=m['git_commit'],
        environment=m['environment'],protocol=m['protocol'],elapsed_s=m['elapsed_s'],environment_controls=m['environment_controls'],
        learning_updates=m['learning_updates'],maximum_actual_pd_error=m['maximum_pd_reconstruction_error'],
        maximum_initial_error=max(v for r in records for v in r['old_reference_initial_max_errors'].values()),
        counterexamples=challenge(hand,objects),chains=chains,records=records)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
