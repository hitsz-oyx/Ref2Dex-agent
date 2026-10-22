"""Print all five outcomes of each chain in the fixed pose-consumption batch."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();m=json.loads((a.run/'manifest.json').read_text())
    if m['status']!='COMPLETED' or len(m['runs'])!=15:
        raise ValueError('completed 15-case batch required')
    summary=dict(status=m['status'],run_id=m['run_id'],git_commit=m['git_commit'],
        environment_controls=m['environment_controls'],elapsed_s=m['elapsed_s'],
        wam_seconds=m['wam_seconds'],learning_updates=m['learning_updates'],chains={},
        maximum_actual_pd_error=m['maximum_pd_reconstruction_error'],
        maximum_initial_error=max(v for r in m['runs'] for v in r['old_reference_initial_max_errors'].values()))
    for name in ('a-original-wam','b-pose-wam','c-live-gt'):
        rows=sorted([r for r in m['runs'] if r['chain']==name],key=lambda r:r['block'])
        if [r['block'] for r in rows]!=list(range(5)):
            raise ValueError('all five blocks required: '+name)
        info=[v for r in rows for v in json.loads((a.run/r['label']/'consumer.json').read_text())]
        summary['chains'][name]=dict(n=5,longest_hold_s=[r['longest_hold_elapsed_s'] for r in rows],
            ever_successes=sum(r['ever_success'] for r in rows),final_successes=sum(r['final_success'] for r in rows),
            controls=sum(r['controls'] for r in rows),final_output_clipped_controls=sum(r['clipped_controls'] for r in rows),
            safety_aborts=sum(r['physical_safety_abort'] is not None for r in rows),
            wam_calls=sum(r['wam_calls'] for r in rows),contact_switches=sum(r['phase_switch_time_s'] is not None for r in rows),
            stage_contact_ever_episodes=sum(r['final_stage']['achieved']['contact'] for r in rows),
            approach_mode_controls=sum(v['plan_mode']=='approach' for v in info),
            contact_window_controls=sum(v['plan_mode']=='contact_window' for v in info))
    a_score=summary['chains']['a-original-wam']['ever_successes']
    b_score=summary['chains']['b-pose-wam']['ever_successes']
    c_score=summary['chains']['c-live-gt']['ever_successes']
    summary['pose_target_met']=b_score>=4
    summary['consumption_improvement_screen']=b_score>=4 and b_score-a_score>=2
    summary['reference_quality_priority_screen']=c_score-b_score>=2
    text=json.dumps(summary,indent=2)+'\n'
    if a.output:
        with a.output.open('x') as stream:stream.write(text)
    print(text,end='')


if __name__=='__main__':main()
