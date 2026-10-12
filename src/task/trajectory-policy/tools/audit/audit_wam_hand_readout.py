"""Independent own-future labels, metrics, draws and offline gate reconstruction."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'tools/audit'))
from audit_wam_ref2_shadow import write
from audit_competent_pd import ACTIVE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);a=p.parse_args()
    out=a.fit.resolve();m=json.loads((out/'manifest.json').read_text());r=json.loads((out/'result.json').read_text())
    if m['status']!='COMPLETED' or r['status']!='COMPLETED':raise ValueError('completed matched run required')
    data={}
    for name,path in [('nominal',m['source'])]+[('recovery%d'%i,v) for i,v in enumerate(m['recovery_sources'])]:
        with np.load(Path(path)/'trajectory.npz',allow_pickle=False) as f:data[name]={k:f[k] for k in f.files}
    controller=json.loads((Path(m['source'])/'manifest.json').read_text())['native_controller']
    scale=np.asarray(controller['scale'])[ACTIVE].copy();scale[6:]/=2
    maxima={};computed={}
    with np.load(out/'predictions.npz',allow_pickle=False) as f:
        for arm in ('old','anchored'):
            computed[arm]={}
            if any(r['arms'][arm][key]!=0 for key in ('frozen_world_action_max_change','buffer_max_change','R_max_change')):
                raise ValueError('frozen model drift')
            phases=('initial','final')+(('gate_zero',) if m.get('readout_mode') and arm=='anchored' else ())
            for phase in phases:
                reported=r['arms'][arm]['gate_zero_ablation'] if phase=='gate_zero' else r['arms'][arm][phase]
                computed[arm][phase]={}
                for group in ('nominal','startup','recovery0_TRAIN','recovery1_TRAIN','recovery2_TRAIN'):
                    prefix=arm+'_'+phase+'_'+group+'_';row_ids=f[prefix+'row_ids'];true_tau=[];true_action=[]
                    for index in row_ids:
                        kind=str(f['rows_kind'][index]);tick=int(f['rows_tick'][index]);role=int(f['rows_role'][index]);d=data[kind]
                        if kind.startswith('recovery') and (role!=3 or not d['expert_applied'][tick:tick+8,1].all()
                                                           or d['expert_clipped'][tick:tick+8,1].any()):
                            raise ValueError('own actual expert future/control not qualified')
                        delta=d['hand'][tick+np.arange(1,25),role]-d['hand'][tick,role]
                        local=np.matmul(delta,d['obj'][tick,role,:3,:3]).reshape(24,33)
                        true_tau.append(local);true_action.append(d['applied'][tick,role][ACTIVE])
                    maxima[prefix+'GT_tau']=float(np.abs(f[prefix+'tau_true']-np.stack(true_tau)).max())
                    maxima[prefix+'actual_first_action']=float(np.abs(f[prefix+'action0_true']-np.stack(true_action)).max())
                    error=np.linalg.norm((f[prefix+'tau']-f[prefix+'tau_true']).reshape(-1,24,11,3),axis=-1)*1000
                    pd=(f[prefix+'action0']-f[prefix+'action0_true'])*scale
                    metrics=dict(h1_mm=float(error[:,0].mean()),first8_mm=float(error[:,:8].mean()),all24_mm=float(error.mean()),
                                 first_PD_xyz_p95_mm=float(np.percentile(np.linalg.norm(pd[:,:3],axis=-1),95)*1000),
                                 clipping_rate=float(f[prefix+'raw_clipped'].mean()))
                    for key,value in metrics.items():maxima[prefix+key]=abs(value-reported[group][key])
                    computed[arm][phase][group]=metrics
                response=np.linalg.norm((f[arm+'_'+phase+'_goal_plus_tau']-f[arm+'_'+phase+'_goal_minus_tau']).reshape(-1,24,11,3),axis=-1)*1000
                value=float(response.mean());maxima[arm+'_'+phase+'_goal_response']=abs(value-reported['goal_sensitivity_mm'])
                computed[arm][phase]['goal_sensitivity_mm']=value
        if m.get('readout_mode'):
            for key in f.files:
                if key.startswith('anchored_initial_'):
                    if not np.array_equal(f[key],f[key.replace('anchored_initial_', 'old_initial_',1)]):
                        raise ValueError('initial behavior is not bitwise preserved')
            gradient=r['arms']['anchored']['gradient']
            if gradient['initial_alpha']!=0 or gradient['initial_behavior_max_difference']!=0 or gradient['gate_grad_step1']<=0:
                raise ValueError('gate initialization/gradient failed')
            if m['maximum_updates_per_arm']>1 and (gradient['projection_grad_step2']<=0 or gradient['final_alpha']==0):
                raise ValueError('projection never received trained gate gradient')
            if any(r['arms'][arm]['frozen_parameter_max_change']!=0 for arm in ('old','anchored')):
                raise ValueError('frozen PTv3/goal/raw adapter changed')
        if not np.array_equal(f['old_draws'],f['anchored_draws']):raise ValueError('unmatched draw matrices')
        expected=[];rng=np.random.default_rng(296)
        pools=[np.flatnonzero((f['rows_kind']==('recovery%d'%i))&(f['rows_role']==3)) for i in range(3)]
        for step in range(m['maximum_updates_per_arm']):
            expected.append(np.concatenate([rng.choice(np.arange(256),16),rng.choice(np.arange(256,8304),16),
                                            *[rng.choice(pool,count) for pool,count in zip(pools,(11,11,10))]]))
        if not np.array_equal(f['old_draws'],expected):raise ValueError('fixed sampling stream differs')
    checks={}
    for group in ('nominal','startup','recovery0_TRAIN','recovery1_TRAIN','recovery2_TRAIN'):
        old,new=[computed[arm]['final'][group] for arm in ('old','anchored')]
        for key in ('h1_mm','first8_mm'):checks[group+'_'+key]=new[key]<=1.1*old[key]+.05
        checks[group+'_control']=new['first_PD_xyz_p95_mm']<=1.1*old['first_PD_xyz_p95_mm']+.1
        checks[group+'_clipping']=new['clipping_rate']<=old['clipping_rate']+.01
    checks['goal_sensitivity']=computed['anchored']['final']['goal_sensitivity_mm']>=max(.001,.5*computed['old']['final']['goal_sensitivity_mm'])
    if checks!=r['checks']:raise ValueError('predeclared gate reconstruction differs')
    if m.get('readout_mode'):
        useful={group+'_'+key:computed['anchored']['final'][group][key]<=.95*computed['old']['final'][group][key]
                for group in ('nominal','startup','recovery0_TRAIN','recovery1_TRAIN','recovery2_TRAIN')
                for key in ('h1_mm','first8_mm','first_PD_xyz_p95_mm')}
        expected_gate=all(checks.values()) and m['maximum_updates_per_arm']>1
        if useful!=r['useful_panels'] or r['B_offline_gate']!=expected_gate or r['B_useful_signal']!=(expected_gate and any(useful.values())):
            raise ValueError('usefulness/non-regression classification differs')
        if r['C_started']:raise ValueError('C must remain paused')
    report=dict(pass_=all(v<=1e-5 for v in maxima.values()),maxima=maxima,matched_sampling=True,
                own_actual_expert_future=True,frozen_R_world_action_buffers=True,checks=checks,B_offline_gate=r['B_offline_gate'])
    report['pass']=report.pop('pass_');write(out/'root_hand_readout_audit.json',report)
    print(json.dumps(dict(pass_=report['pass'],checks=len(maxima),maximum_error=max(maxima.values()),B_offline_gate=r['B_offline_gate'])))
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
