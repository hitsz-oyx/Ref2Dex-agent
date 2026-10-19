"""Audit initial states and summarize a prepared mixed-object checkpoint panel."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import numpy as np


METRICS=('native_success','ever_held60','final_held60','drop_after_hold',
         'final_task_success','final_stable_success','workspace_failure')


def initial_difference(left,right):
    differences={}
    with np.load(left/'initial-state.npz') as a,np.load(right/'initial-state.npz') as b:
        if set(a.files)!=set(b.files):raise ValueError('Initial state schema changed')
        n=len(a['observation']);matched=np.ones(n,bool)
        for state in (a,b):
            if state['root'].shape!=(n*4,13):raise ValueError('Native four-actor root layout required')
            if not np.array_equal(state['root'].reshape(n,4,13)[:,1,:3],state['pose'][:,:3,3]):
                raise ValueError('Object actor/environment root mapping differs')
        for key in a.files:
            same_shape=a[key].shape==b[key].shape
            equal=same_shape and np.array_equal(a[key],b[key])
            per_env=(a[key]==b[key]).reshape(n,-1).all(1) if same_shape else np.zeros(n,bool)
            matched &= per_env
            differences[key]=dict(equal=equal,max_abs_difference=float(np.max(np.abs(a[key]-b[key]))) if same_shape else None,
                                  differing_envs=np.flatnonzero(~per_env).tolist())
    left_objects=json.loads((left/'env-objects.json').read_text());right_objects=json.loads((right/'env-objects.json').read_text())
    objects_equal=left_objects==right_objects
    matched &= np.array([a==b for a,b in zip(left_objects,right_objects)])
    return dict(matched=objects_equal and all(d['equal'] for d in differences.values()),
                objects_equal=objects_equal,fields=differences,matched_envs=np.flatnonzero(matched).tolist())


def cluster_interval(values,objects,rng):
    groups=sorted(set(objects))
    counts=np.array([sum(o==g for o in objects) for g in groups])
    sums=np.array([values[np.array(objects)==g].sum() for g in groups])
    draws=rng.integers(len(groups),size=(10000,len(groups)))
    return (np.quantile(sums[draws].sum(1)/counts[draws].sum(1),[.025,.975])*100).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();manifest=json.loads((out/'process.json').read_text())
    if manifest['status']!='COMPLETED':raise ValueError('Complete panel required')
    paths={};reports={};objects={}
    for job in manifest['checkpoints']:
        panel=job.get('panel',0);iteration=job['iteration']
        folder=out/(('panel-%d-'%panel if 'panel' in job else '')+'step-%d'%iteration)
        report=json.loads((folder/'evaluation.json').read_text());records=report['records']
        if report['optimizer_steps'] or report['iteration']!=iteration or len(records)!=manifest['envs']:
            raise ValueError('Evaluation identity or update count changed')
        if sorted(r['env'] for r in records)!=list(range(manifest['envs'])):raise ValueError('First episode membership changed')
        paths[panel,iteration]=folder;reports[panel,iteration]=report
        rows=json.loads((folder/'env-objects.json').read_text())
        objects[panel,iteration]=['%s@%g'%(r['object_code'],r['scale']) for r in rows]
    panels=sorted({p for p,_ in paths});iterations=sorted({i for _,i in paths})
    if len(paths)!=len(panels)*len(iterations):raise ValueError('Incomplete checkpoint/panel grid')
    rng=np.random.default_rng(manifest['seed']);audit={};summary={};common={}
    for panel in panels:
        admitted=set(range(manifest['envs']))
        for iteration in iterations:
            audit['%d:%d'%(panel,iteration)]=initial_difference(paths[panel,iterations[0]],paths[panel,iteration])
            admitted &= set(audit['%d:%d'%(panel,iteration)]['matched_envs'])
        common[panel]=sorted(admitted)
    for iteration in iterations:
        records=[];codes=[];nominal=[]
        for panel in panels:
            report=reports[panel,iteration];nominal+=report['records']
            if [r['env'] for r in report['records']]!=list(range(manifest['envs'])):raise ValueError('Environment order differs')
            records += [report['records'][i] for i in common[panel]];codes += [objects[panel,iteration][i] for i in common[panel]]
        if not records:raise ValueError('No common initial scenes; cannot assess checkpoint trend')
        summary[str(iteration)]=dict(episodes=len(records),objects=len(set(codes)),scenes_per_object=dict(Counter(codes)),
            counts={k:sum(bool(r[k]) for r in records) for k in METRICS},mean_return=float(np.mean([r['reward'] for r in records])),
            nominal_episodes=len(nominal),nominal_counts={k:sum(bool(r[k]) for r in nominal) for k in METRICS},
            all_initials_match_first=all(audit['%d:%d'%(p,iteration)]['matched'] for p in panels))
    pairs={}
    for left,right in [(i,iterations[-1]) for i in iterations[:-1]]:
        matches={str(p):initial_difference(paths[p,left],paths[p,right]) for p in panels}
        admitted=[p for p in panels if common[p]]
        result=dict(initial_audit=matches,matched_panels=admitted,selection='same all-checkpoint common initial scenes',metrics={})
        if admitted:
            codes=[objects[p,left][i] for p in admitted for i in common[p]]
            result.update(episodes=len(codes),objects=len(set(codes)))
            for key in ('native_success','final_task_success','final_stable_success'):
                a=np.array([reports[p,left]['records'][i][key] for p in admitted for i in common[p]],dtype=int)
                b=np.array([reports[p,right]['records'][i][key] for p in admitted for i in common[p]],dtype=int)
                result['metrics'][key]=dict(delta_pp=float((b-a).mean()*100),
                    wins=int(((b==1)&(a==0)).sum()),losses=int(((b==0)&(a==1)).sum()),
                    cluster_bootstrap95_pp=cluster_interval(b-a,codes,rng))
        pairs['%d->%d'%(left,right)]=result
    primary=[pairs.get('8500->10000'),pairs.get('9500->10000')]
    first=summary[str(iterations[0])]
    sufficient=all(primary) and first['episodes']>=.9*first['nominal_episodes'] and first['objects']==manifest.get('objects',first['objects'])
    plateau=sufficient and all(abs(p['metrics'][k]['delta_pp'])<=3 for p in primary for k in ('native_success','final_stable_success'))
    still_allows5pp=not sufficient or any(p['metrics'][k]['cluster_bootstrap95_pp'][1]>=5 for p in primary for k in ('native_success','final_stable_success'))
    result=dict(status='COMPLETED',scope=manifest['scope'],checkpoints=summary,pairs=pairs,
        common_initial_scenes={str(p):common[p] for p in panels},sufficient_common_coverage=sufficient,
        analysis_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[5],text=True).strip(),
        observed_plateau_signal=bool(plateau),
        five_pp_gain_not_excluded=bool(still_allows5pp),
        uncertainty='10000 object-cluster bootstrap resamples, descriptive single-seed TRAIN panel; not a global convergence proof')
    (out/'initial-state-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for key,label in [('native_success','Ever native success'),('final_stable_success','Final stable hold')]:
        rates=[summary[str(i)]['counts'][key]/summary[str(i)]['episodes']*100 for i in iterations]
        ax.plot(iterations,rates,'o-',label=label)
    ax.set(xlabel='PPO cumulative iteration',ylabel='First-episode rate (%)',ylim=(0,100),
        title='%d TRAIN objects; %d scenes/checkpoint'%(summary[str(iterations[0])]['objects'],summary[str(iterations[0])]['episodes']))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'checkpoint-trend.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
