"""Review every paired Shadow trajectory and export an offline 3D log viewer."""
import argparse
import base64
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]


def runs(mask):
    longest=tail=0;prefix=[]
    for active in mask:
        tail=tail+1 if active else 0;longest=max(longest,tail);prefix.append(tail)
    return longest,tail,prefix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or ROOT/'outputs/wam-rl-policy' not in a.output.resolve().parents:
        raise ValueError('Fresh Task output required')
    a.output.mkdir(parents=True)
    scratch=ROOT/'tmp'/a.output.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(TMPDIR=str(scratch),MPLCONFIGDIR=str(scratch/'matplotlib'))
    import numpy as np
    cache=ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1'
    lookup={r['object_code']:r['geometry'] for r in json.loads((cache/'manifest.json').read_text())['records']}
    geometry={};rows=[];cases=[];packets={}
    for seed in (216,233,234):
        identities=None
        for role in ('bc','ppo'):
            source=a.run/('eval-seed%d'%seed)/role
            ids=json.loads((source/'environment-objects.json').read_text())
            if identities is not None and ids!=identities:raise ValueError('Paired object assignments differ')
            identities=ids
            with np.load(source/'last-episode.npz') as z:
                data={k:z[k].copy() for k in ('hand','object','goal','action','base','intended')}
            n=len(ids)
            if (data['hand'].shape!=(200,n,11,3) or data['object'].shape!=(200,n,7)
                    or data['goal'].shape!=(200,n,3) or any(not np.isfinite(v).all() for v in data.values())):
                raise ValueError('Finite native200-frame hand11/object7/goal3 logs required')
            distance=np.linalg.norm(data['object'][...,:3]-data['goal'],axis=-1)
            lift=data['object'][...,2]-data['object'][0,:,2]
            packed=np.concatenate((data['object'],data['hand'].reshape(len(lift),len(ids),33),data['goal']),-1).astype('<f4')
            name='%d_%s'%(seed,role)
            packets[name]=dict(shape=list(packed.shape),data=base64.b64encode(packed.tobytes()).decode())
            for i,identity in enumerate(ids):
                code=identity['object_code'];short=code.split('/')[-1].split('-')[0]
                if code not in geometry:
                    with np.load(lookup[code]) as f:
                        if str(f['object_code'].item())!=code or abs(float(f['scale'].item())-identity['scale'])>1e-7:
                            raise ValueError('Visualization CAD identity/scale must match simulation')
                        keep=np.linalg.norm(f['normals'],axis=-1)>1e-6;points=f['surface'][keep]
                    geometry[code]=points[np.linspace(0,len(points)-1,min(384,len(points)),dtype=int)].tolist()
                held=(distance[:,i]<=.05)&(lift[:,i]>=.1)
                best,tail,prefix=runs(held)
                if lift[:,i].max()<.1:outcome='never_lift10cm'
                elif best==0:outcome='lifted_without_qualified_goal'
                elif best<60:outcome='goal_hold_shorter_than60'
                elif tail>=60:outcome='held60_through_endpoint'
                else:outcome='held60_earlier_endpoint_tail_short'
                row=dict(seed=seed,env=i,role=role,object=short,object_code=code,split=identity['split'],
                    max_lift_m=float(lift[:,i].max()),final_lift_m=float(lift[-1,i]),
                    final_object_height_m=float(data['object'][-1,i,2]),
                    reached5cm=bool((distance[:,i]<=.05).any()),longest_hold_frames=best,endpoint_hold_frames=tail,
                    final_goal_distance_m=float(distance[-1,i]),held60=best>=60,held60_at_end=tail>=60,
                    action_change_rms=float(np.sqrt(np.mean((data['action'][:,i]-data['base'][:,i])**2))),outcome=outcome)
                rows.append(row)
                if role=='bc':cases.append(dict(seed=seed,env=i,object=short,object_code=code,split=identity['split']))
                packets[name].setdefault('metrics',[]).append(dict(row,prefix=prefix))
    aggregates=[]
    for code in sorted(geometry):
        for role in ('bc','ppo'):
            selected=[r for r in rows if r['object_code']==code and r['role']==role]
            aggregates.append(dict(object=selected[0]['object'],object_code=code,split=selected[0]['split'],role=role,
                episodes=len(selected),lift10=sum(r['max_lift_m']>=.1 for r in selected),
                reached5cm=sum(r['reached5cm'] for r in selected),held60=sum(r['held60'] for r in selected),
                held60_at_end=sum(r['held60_at_end'] for r in selected),
                outcomes={key:sum(r['outcome']==key for r in selected) for key in sorted({r['outcome'] for r in selected})}))
    paired=defaultdict(dict)
    for row in rows:paired[(row['seed'],row['env'])][row['role']]=row
    transitions={}
    for metric in ('held60','held60_at_end'):
        transitions[metric]={name:sum(bool(v['bc'][metric])==bc and bool(v['ppo'][metric])==ppo for v in paired.values())
            for name,bc,ppo in [('both',True,True),('bc_only',True,False),('ppo_only',False,True),('neither',False,False)]}
    report=dict(status='UNCLEAR',objects=aggregates,cases=rows,paired_transitions=transitions,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TASK,text=True).strip(),
        input_run=str(a.run.resolve()),run_id=a.output.name,model_compute=False,
        scope='All48 stored paired trajectories per policy; full hand/object state logs, no new training or simulation',
        metric='60 consecutive60Hz states with object-goal distance<=5cm AND lift>=10cm',
        endpoint='Endpoint tail is right-censored; a short final tail does not prove loss of grasp',
        visualization='Measured11 hand landmarks and transformed object CAD surface points; not simulator camera rendering or contact certification')
    (a.output/'behavior.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=dict(cases=cases,geometry=geometry,packets=packets)
    template=(TASK/'tools/audit/shadow_viewer.html').read_text()
    (a.output/'replay.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,separators=(',',':'))))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    labels=['Lift >=10cm','Reached5cm','Held60','Held60 at end']
    for ax,code in zip(axes.flat,sorted(geometry)):
        for role,offset in [('bc',-.18),('ppo',.18)]:
            item=next(r for r in aggregates if r['object_code']==code and r['role']==role)
            ax.bar(np.arange(4)+offset,[item[k] for k in ('lift10','reached5cm','held60','held60_at_end')],width=.35,label=role)
        ax.set_xticks(np.arange(4));ax.set_xticklabels(labels,rotation=10,fontsize=8)
        ax.set(title=item['object']+' / '+item['split'].upper(),ylim=(0,12.7),ylabel='Observed trajectories /12')
        ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Shadow BC vs final PPO: all three reset panels, position proxies')
    fig.tight_layout();fig.savefig(a.output/'objects.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(objects=aggregates,output=str(a.output),html_bytes=(a.output/'replay.html').stat().st_size)))


if __name__=='__main__':main()
