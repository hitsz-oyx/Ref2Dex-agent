"""P2 necessary conditions on the six saved production source-PD traces."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3';sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.task_spec import TaskSpec
from wam_rl_policy_v3.task_scheduler import TaskScheduler
from wam_rl_policy_v3.terminal_support_contract import terminal_support_compatibility
from wam_rl_policy_v3.tau_replay_contract import bouts


def errors(poses,goal,points):
    position=np.linalg.norm(poses[:,:3,3]-goal[:3,3],axis=-1)
    angle=Rotation.from_matrix(poses[:,:3,:3].transpose(0,2,1)@goal[:3,:3]).magnitude()
    actual=points[None]@poses[:,:3,:3].transpose(0,2,1)+poses[:,None,:3,3]
    target=points@goal[:3,:3].T+goal[:3,3]
    mean_points=np.linalg.norm(actual-target[None],axis=-1).mean(-1)
    return position,angle,mean_points,(position<=.008)&(angle<=.12)&(mean_points<=.008)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);manifest=json.loads((a.run/'manifest.json').read_text());decl=Path(manifest['source_manifest']);rows={r['sequence']:r for r in json.loads(decl.read_text())['records']}
    previous=json.loads((a.run/'scoring-final/scores.json').read_text());scores={r['sequence']:r for r in previous['records']};traces=json.loads((a.run/'scoring-final/task-traces.json').read_text());results=[]
    with np.load(a.run/'trajectory.npz') as captured:
        for i,name in enumerate(manifest['sequence_order']):
            row=rows[name];side=decl.parent/row['sidecar'];spec=TaskSpec.from_packet(json.loads((side/'raw-task.json').read_text()))
            with np.load(Path(row['source'])/'canonical.npz') as z:points=z['points'].copy()
            with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
            objects=captured['objects'][:,i];poses=np.repeat(np.eye(4)[None],len(objects),0);poses[:,:3,:3]=Rotation.from_quat(objects[:,3:7]).as_matrix();poses[:,:3,3]=objects[:,:3]
            table=captured['table_pose'][:,i];table_delta=float(np.abs(table-table[:1]).max());proof=terminal_support_compatibility(spec,points,table[0],bounds)
            if table_delta>1e-6:proof.update(geometrically_disproved=False,scope='moving table: fixed-table proof not applicable')
            abort=manifest['aborts'][i];end=len(objects) if abort is None else abort['tick'];eligible=np.arange(len(objects))<end
            pos,ang,point,reached=errors(poses,spec.object_path[-1],points)
            feedback=traces[name]['feedback'];terminal_conditions=reached&np.array([r['released'] and r['supported'] and r['stable'] for r in feedback])&eligible
            terminal_bouts=bouts(terminal_conditions);longest=max(((hi-lo-1)/30 for lo,hi in terminal_bouts),default=0.)
            # A pending ordered milestone can be geometrically visited yet not
            # certified in order. Keep those facts separate.
            terminal=scores[name]['after_terminal'];scheduler=TaskScheduler(spec,points);pointer=terminal['pointer'] if terminal else scheduler.path.pointer
            pending_index=int(scheduler.path.milestones[min(pointer+1,len(scheduler.path.milestones)-1)])
            ppos,pang,ppoint,pending=errors(poses,spec.object_path[pending_index],points)
            releases=[event for event in spec.events if event.kind.value=='RELEASE'];release_frame=releases[-1].source_frame if releases else None
            release_reached=None
            if release_frame is not None:
                k=int(np.flatnonzero(spec.source_frames==release_frame)[0]);release_reached=bool((errors(poses,spec.object_path[k],points)[3]&eligible).any())
            first_grasp=next((r['tick'] for r in feedback if r['grasp'] and r['tick']<end),None)
            released_after=next((r['tick'] for r in feedback if first_grasp is not None and first_grasp<r['tick']<end and r['released']),None)
            record=dict(sequence=name,source_split=row['split'],raw_terminal_mode=spec.terminal_mode.value,raw_require_support=spec.require_support,
                geometry=proof,table_pose_max_change=table_delta,physical_or_control_abort=abort,full_source_executed=scores[name]['full_source_commands_executed'],
                real_post_source_controls_observed=scores[name]['observed_suffix_controls'] if abort is None else None,
                eligible_source_observed_states=end,full_task_success=scores[name]['after_full_task_success'],terminal_failure=None if terminal is None else terminal['failure'],
                maximum_path_progress=scores[name]['maximum_path_progress']['after'],pending_source_frame=int(spec.source_frames[pending_index]),pending_pose_visited_anytime=bool((pending&eligible).any()),
                pending_pose_min_position_error_m=float(ppos[eligible].min()),goal_visited_states=int((reached&eligible).sum()),goal_min_position_error_m=float(pos[eligible].min()),
                goal_min_rotation_error_rad=float(ang[eligible].min()),goal_min_mean_point_error_m=float(point[eligible].min()),
                independent_goal_release_support_stability_longest_s=longest,first_grasp_frame=first_grasp,first_release_after_grasp_frame=released_after,
                declared_release_frame=release_frame,declared_release_pose_visited_anytime=release_reached,
                annotation_recommendation='annotation_invalid: terminal_goal_incompatible_with_required_table_support' if proof['geometrically_disproved'] else 'Not geometrically disproved; intent and physical executability still unqualified')
            results.append(record)
    invalid={r['sequence'] for r in results if r['geometry']['geometrically_disproved']};aborted={r['sequence'] for r in results if r['physical_or_control_abort']};unexplained={r['sequence'] for r in results}-invalid-aborted
    summary=dict(status='COMPLETED',records=results,counts=dict(all_cases=len(results),successes=0,physical_aborts=len(aborted),terminal_support_contradictions=len(invalid),abort_contradiction_overlap=len(aborted&invalid),
        complete_source_and_suffix_without_abort=sum(r['full_source_executed'] and r['physical_or_control_abort'] is None for r in results),neither_abort_nor_disproved=len(unexplained)),
        remaining_cases=sorted(unexplained),necessary_conditions=['Independently declare intended endpoint/support object; do not infer complete-task intent from final measured contact loss.',
        'Explicitly mark incompatible declarations annotation_invalid; retain every original record and report rejected sequences, never relabel UNKNOWN to HOLD.',
        'Traverse ordered milestones under the actual readiness/recovery contract, reaching release pose before authorized release.',
        'At actual endpoint meet8mm translation/mean-point and.12rad rotation, released+stable+measured table support for1 elapsed second.',
        'Observe real post-endpoint dynamics; respect physical DOF safety and qualify source-PD and GT-tau executor separately.'],
        limits=['This is a proof about current fixed-table success predicate, not universal physical impossibility.','Not-disproved is not executable or intent-valid.','Original packets/labels/gates are not rewritten by this audit.','No new complete-task success denominator certified; all6 descriptive outcomes retained.'])
    (out/'release-reachability.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(status=summary['status'],counts=summary['counts'],records=[{k:r[k] for k in ('sequence','geometry','goal_visited_states','goal_min_position_error_m','pending_source_frame','pending_pose_visited_anytime','independent_goal_release_support_stability_longest_s')} for r in results])))


if __name__=='__main__':main()
