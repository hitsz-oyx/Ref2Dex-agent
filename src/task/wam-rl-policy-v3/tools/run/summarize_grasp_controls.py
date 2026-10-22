"""V8 complete denominators, saved-state contiguous scoring and declared budgets."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.evaluation_sources import continuous_hold


def summarize(paths):
    groups={};all_runs=[];manifest_rows=[]
    for path in paths:
        manifest=json.loads((path/'manifest.json').read_text())
        manifest_rows.append(dict(path=str(path),status=manifest['status'],git_commit=manifest['git_commit'],
            environment_controls=manifest['environment_controls'],elapsed_s=manifest.get('elapsed_s'),maximum_controls=manifest['maximum_environment_controls'],
            pd_max_error=manifest['maximum_pd_reconstruction_error'],reference_qualification=manifest.get('reference_qualification'),error=manifest.get('error')))
        for r in manifest['runs']:
            row=dict(r,run=str(path),sequence=manifest['source_sequence']);all_runs.append(row)
            key=manifest['protocol'].split('.')[-1]+':'+r['chain'];groups.setdefault(key,[]).append(row)
    result={}
    for key,rows in groups.items():
        result[key]=dict(n=len(rows),ever=sum(r['ever_success'] for r in rows),final=sum(r['final_success'] for r in rows),
            longest_hold_s=[r['longest_hold_elapsed_s'] for r in rows],controls=[r['controls'] for r in rows],
            final_output_clipped=[r['clipped_controls'] for r in rows],adapter_clipped=[r['adapter_clipped_controls'] for r in rows],
            physical_abort=[r['physical_safety_abort'] for r in rows],wam_calls=[r['wam_calls'] for r in rows],actor_calls=[r['actor_calls'] for r in rows],
            goal_error_mm=[1000*r['final_object_goal_error_m'] for r in rows],score_recalculation_matches=all(r['ever_success']==r['independent_continuous_score']['ever_success'] and r['final_success']==r['independent_continuous_score']['final_success'] for r in rows))
    return dict(groups=result,runs=all_runs,manifests=manifest_rows,environment_controls=sum(r['environment_controls'] for r in manifest_rows),execution_seconds=sum(r['elapsed_s'] or 0 for r in manifest_rows),learning_updates=0,
        limitations=['n=5 is local engineering evidence, not powered proof of a universal tau benefit.','New initial states exclude v7 tuning panel; objects may be seen in training.','Template uses one disclosed successful robot demonstration; it is not a matched-data learning comparison.','Independent continuous-time calculation shares the contact observer, not independent contact ground truth.','Success is >=5cm held continuously1s, not exact8cm target or full RELEASE.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.run);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))


if __name__=='__main__':main()
