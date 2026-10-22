"""Full outcome and budget ledger of the v7 interface implementation run."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();run=args.run.resolve();manifest=json.loads((run/'manifest.json').read_text())
    report=dict(run_id=manifest['run_id'],status=manifest['status'],git_commit=manifest['git_commit'],
        environment_controls_all_attempts=manifest['environment_controls'],elapsed_s_all_attempts=manifest.get('elapsed_s'),
        learning_updates=manifest['learning_updates'],chains={},maximum_actual_pd_error=manifest['maximum_pd_reconstruction_error'])
    formal_controls=0
    for name in ('a-original-wam','b-grasp-intent-wam'):
        part=sorted([r for r in manifest['runs'] if r['chain']==name],key=lambda r:r['block'])
        if len({r['block'] for r in part})!=len(part):raise ValueError('duplicate evaluation block')
        formal_controls+=sum(r['controls'] for r in part)
        report['chains'][name]=dict(n=len(part),blocks=[r['block'] for r in part],
            longest_hold_s=[r['longest_hold_elapsed_s'] for r in part],
            ever_successes=sum(r['ever_success'] for r in part),final_successes=sum(r['final_success'] for r in part),
            controls=sum(r['controls'] for r in part),final_output_clipped_controls=sum(r['clipped_controls'] for r in part),
            joint_limit_episodes=sum(r['physical_safety_abort']=='physical_joint_limit' for r in part),
            other_abort_episodes=sum(r['physical_safety_abort'] not in (None,'physical_joint_limit') for r in part),
            final_object_goal_errors_m=[r['final_object_goal_error_m'] for r in part],
            measured_grasp_handoffs=sum(r['phase_switch_time_s'] is not None for r in part),
            phase_controls={stage:sum(r['phase_counts'][stage] for r in part) for stage in ('approach','close','lift','hold')},
            maximum_initial_error=max([e for r in part for e in r['old_reference_initial_max_errors'].values()]+[0.]))
    report['formal_controls']=formal_controls
    report['engineering_only_controls']=manifest['environment_controls']-formal_controls
    report['maximum_actual_pd_error_all_attempts']=report['maximum_actual_pd_error']
    report['prior_versions']=[];path=manifest.get('engineering_attempts_source');seen=set();total_current=manifest.get('current_environment_controls',manifest['environment_controls'])
    while path:
        previous_path=Path(path).resolve()
        if str(previous_path) in seen:raise ValueError('cyclic run ledger')
        seen.add(str(previous_path));previous=json.loads((previous_path/'manifest.json').read_text())
        own=previous.get('current_environment_controls',previous['environment_controls']);total_current+=own
        report['maximum_actual_pd_error_all_attempts']=max(report['maximum_actual_pd_error_all_attempts'],previous['maximum_pd_reconstruction_error'])
        part=[r for r in previous['runs'] if r['chain']=='b-grasp-intent-wam']
        report['prior_versions'].append(dict(run=str(previous_path),status=previous['status'],git_commit=previous['git_commit'],
            own_controls=own,prototype_n=len(part),prototype_hold_s=[r['longest_hold_elapsed_s'] for r in part],
            prototype_successes=sum(r['ever_success'] for r in part),partial_episode=previous.get('partial_episode_saved')))
        path=previous.get('engineering_attempts_source')
    if total_current!=manifest['environment_controls'] or report['engineering_only_controls']<0:
        raise ValueError('budget ledger mismatch')
    b=report['chains']['b-grasp-intent-wam'];report['target_met']=b['n']==5 and b['ever_successes']>=4
    report['complete_n5_comparison']=all(c['n']==5 for c in report['chains'].values())
    report['limitations']=['Repeated seen single initial state, not independent tasks/seeds.',
        'Unchanged completed baseline carried across process resets; visible initial fields checked, hidden physics state not certified equal.',
        'Success means the original 5cm/force/coupled/unsupported elapsed1s witness, not full RELEASE or exact8cm goal.']
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(text)
    print(text,end='')


if __name__=='__main__':main()
