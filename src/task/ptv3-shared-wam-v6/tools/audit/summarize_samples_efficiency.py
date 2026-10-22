"""Report paired common-state learning curves without fabricating missing points."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def compare_initials(reference, candidate):
    with np.load(reference) as left, np.load(candidate) as right:
        if set(left.files)!=set(right.files):return False
        return all(np.array_equal(left[key],right[key]) for key in left.files)


def metrics(points, budget=1048576, threshold=.5):
    points=sorted(points,key=lambda row:row['environment_steps'])
    x=np.array([row['environment_steps'] for row in points])
    y=np.array([row['task_successes']/row['n'] for row in points])
    reaches=[int(x[i]) for i in range(len(x)-1) if y[i]>=threshold and y[i+1]>=threshold]
    complete=bool(len(x) and x[0]==0 and x[-1]==budget and len(x)==5)
    fixed=next((row for row in points if row['environment_steps']==budget),None)
    return dict(n_checkpoints=len(points),threshold_steps=reaches[0] if reaches else None,
        threshold_censored_at=int(x[-1]) if len(x) and not reaches else None,
        normalized_auc=float(np.trapz(y,x)/budget) if complete else None,
        complete_budget=complete,fixed_successes=fixed['task_successes'] if fixed else None,
        fixed_n=fixed['n'] if fixed else None,
        fixed_failures=dict(collections.Counter(row['failure_reason'] for row in fixed['rows'])) if fixed else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    root=a.output.resolve();groups=collections.defaultdict(list);reference=None;checks=[]
    for path in sorted(root.glob('*-eval*/policy/evaluation.json')):
        row=json.loads(path.read_text());name=path.parents[1].name
        seed=int(name.split('-seed')[1].split('-eval')[0]);initial=path.parent/'initial-state.npz'
        if reference is None:reference=initial
        matches=compare_initials(reference,initial);checks.append(dict(path=str(initial),matches=matches))
        if matches:groups[(row['arm'],seed)].append(row)
    runs=[dict(arm=arm,seed=seed,**metrics(points)) for (arm,seed),points in sorted(groups.items())]
    valid=checks and all(row['matches'] for row in checks)
    complete=valid and len(runs)==9 and all(row['complete_budget'] for row in runs)
    result=dict(status='COMPLETE' if complete else 'INCOMPLETE' if valid else 'UNPAIRED_OR_NO_RESULTS',
        common_initial_state_checks=checks,runs=runs,missing_results='Unfinished seed/arm/checkpoint is missing, never a zero success rate')
    costs=[]
    for path in sorted(root.glob('*/interaction-costs.json')):
        costs.append(dict(path=str(path),**json.loads(path.read_text())))
    result['interaction_costs']=costs
    result['total_train_physical_transitions']=sum(row['actual_physical_transitions'] for row in costs if not row['evaluation'])
    result['total_eval_physical_transitions']=sum(row['actual_physical_transitions'] for row in costs if row['evaluation'])
    (root/'sample-efficiency-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if groups:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axis=plt.subplots(figsize=(7,4.2))
        for arm,color in [('A','#3b6fb6'),('B','#cd7e29'),('C','#438d52')]:
            checkpoints=collections.defaultdict(list)
            for (key,seed),points in groups.items():
                if key!=arm:continue
                ordered=sorted(points,key=lambda row:row['environment_steps'])
                axis.plot([p['environment_steps'] for p in ordered],[p['task_successes']/p['n'] for p in ordered],color=color,alpha=.18)
                for point in points:checkpoints[point['environment_steps']].append(point['task_successes']/point['n'])
            if checkpoints:
                x=sorted(checkpoints);y=np.array([np.mean(checkpoints[step]) for step in x]);sd=np.array([np.std(checkpoints[step]) for step in x])
                axis.plot(x,y,'o-',color=color,label=arm)
                axis.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color=color,alpha=.13)
        axis.axhline(.5,color='grey',ls='--',lw=1);axis.set(xlabel='Training environment transitions',ylabel='Task success fraction',ylim=(0,1),title='Grasp/lift: individual seeds and mean ± SD')
        axis.legend();fig.tight_layout();fig.savefig(root/'learning-curves.png',dpi=180);fig.savefig(root/'learning-curves.svg');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
