"""V9 counts/Wilson intervals/Fisher tests, always separated by subset."""
import argparse,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[5];sys.path.insert(0,str(ROOT/'src/task/wam-rl-policy-v3/src'))
from wam_rl_policy_v3.binary_statistics import wilson,compare


def summarize(reg,manifests):
    rows=[];seen=set();workers=[]
    for path in manifests:
        m=json.loads(path.read_text());workers.append(dict(path=str(path),status=m['status'],controls=m['environment_controls'],elapsed_s=m.get('elapsed_s'),active=m.get('active_episode'),error=m.get('error')))
        if m['protocol']!=reg['protocol']:raise ValueError('mixed scoring protocols')
        for r in m['runs']:
            if r['id'] in seen:raise ValueError('duplicate episode, probably carried results counted twice')
            seen.add(r['id']);rows.append(dict(r,run=str(path.parent)))
    groups={};comparisons={};retraining={}
    for subset in reg['subsets']:
        groups[subset]={};comparisons[subset]={};index={}
        for arm in reg['arms']:
            r=sorted([r for r in rows if r['subset']==subset and r['arm']==arm],key=lambda r:r['trial']);n=len(r);k=sum(x['final_success'] for x in r);index[arm]={x['trial']:x for x in r}
            if n>40:raise ValueError('oversampled arm')
            errors=np.array([x['final_goal_error_m'] for x in r]);survived=[x for x in r if x['controls']==reg['horizon_controls'] and x['physical_abort'] is None]
            stage={}
            for phase in ('approach','contact','lift','hold','release'):
                if phase=='release':stage[phase]=dict(applicable=False,n=0,ever=None,final=None);continue
                stage[phase]=dict(applicable=True,n=n,ever=sum(x['stage_achieved'][phase] for x in r),final=sum(x['stage_final'][phase] for x in r))
            groups[subset][arm]=dict(n=n,k=k,wilson95=wilson(k,n) if n else None,complete=n==40,ever_precise_hold=sum(x['ever_success'] for x in r),final_hold=sum(x['final_hold'] for x in r),stages=stage,
                initial_families=len({x['sequence'] for x in r}),goal15_k=sum(x['goal_within15mm'] for x in r),goal15_n=n,goal15_wilson95=wilson(sum(x['goal_within15mm'] for x in r),n) if n else None,
                terminal_goal_error_mm=[float(v*1000) for v in errors],terminal_error_quantiles_mm=np.quantile(errors*1000,[0,.25,.5,.75,.9,1]).tolist() if n else None,
                full_horizon_survivors=len(survived),fixed_horizon_error_mm=[x['final_goal_error_m']*1000 for x in survived],
                reach_filter=[x.get('reach_filter_controls',0) for x in r],recovery=[x.get('recovery_controls',0) for x in r],goal_servo_saturated=[x.get('goal_servo_saturated_controls',0) for x in r],actor_calls=[x.get('actor_calls',0) for x in r],encoder_covered=[x.get('encoder_covered_controls',0) for x in r],
                hold_s=[x['longest_hold_s'] for x in r],controls=[x['controls'] for x in r],adapter_clip=[x['adapter_clipped_controls'] for x in r],final_clip=[x['final_clipped_controls'] for x in r],physical_abort=[x['physical_abort'] for x in r])
        for a,b in itertools.combinations(reg['arms'],2):
            ga,gb=groups[subset][a],groups[subset][b]
            if ga['n'] and gb['n']:
                c=compare(ga['k'],ga['n'],gb['k'],gb['n']);c['claim_eligible']=ga['complete'] and gb['complete']
                if not c['claim_eligible']:c['registered_a_superior']=False
                paired=sorted(set(index[a])&set(index[b]));ab=sum(index[a][t]['final_success'] and not index[b][t]['final_success'] for t in paired);ba=sum(index[b][t]['final_success'] and not index[a][t]['final_success'] for t in paired)
                c['paired_supplement']=dict(n=len(paired),a_only=ab,b_only=ba,mcnemar_exact_p=float(binomtest(ab,ab+ba,p=.5).pvalue) if ab+ba else 1.)
                comparisons[subset][a+'_vs_'+b]=c
        a1,a4=groups[subset]['A1'],groups[subset].get('A4');retraining[subset]=bool(a4 and a1['complete'] and a4['complete'] and a4['k']>=32 and a1['k']<=8)
    complete=all(groups[s][a]['complete'] for s in reg['subsets'] for a in reg['arms'])
    return dict(status='COMPLETED' if complete else 'INCOMPLETE',protocol=reg['protocol'],groups=groups,comparisons=comparisons,retraining_recommendation=retraining,
        robust_tau_contribution=bool(complete and all(comparisons[s]['A1_vs_A2']['registered_a_superior'] for s in reg['subsets'])),
        workers=workers,completed_episodes=len(rows),environment_controls=sum(w['controls'] for w in workers),rows=rows,
        scope=reg.get('scope','Wilson and Fisher describe fixed-panel episode trials; repeated families are not independent objects. Unseen means no interface tuning, not unseen in WAM/BC pretraining. No automatic retraining.'),
        primary=reg['primary'],error_quantiles_order=[0,.25,.5,.75,.9,1])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--manifest',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(json.loads(a.registration.read_text()),a.manifest);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','groups','comparisons')},indent=2))


if __name__=='__main__':main()
