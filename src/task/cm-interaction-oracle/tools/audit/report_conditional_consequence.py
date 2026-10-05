#!/usr/bin/env python3
"""Export every signed arm vector, amplitude comparison and standalone ref9 plot."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
from intervention import PER_FINGER_ARM_NAMES
from conditional_consequence import contrast_scores

E_NAMES=('dx_m','dy_m','dz_m','rotation_x_rad','rotation_y_rad','rotation_z_rad',
    'delta_vx_m_s','delta_vy_m_s','delta_vz_m_s','delta_wx_rad_s','delta_wy_rad_s','delta_wz_rad_s')
I_NAMES=tuple(f'{body}_log1p_force_norm_N' for body in ('index','middle','pinky','ring','thumb'))+tuple(f'object_log1p_abs_force_{axis}_N' for axis in ('x','y','z'))+tuple(f'{body}_surface_distance_m' for body in ('index','middle','pinky','ring','thumb'))+('global_contact_proxy',)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); out=args.run_dir
    if (out/'per_arm_contrasts.json').exists(): raise FileExistsError(out/'per_arm_contrasts.json')
    d=torch.load(out/'diagnostic.pt',weights_only=False,map_location='cpu'); r=json.loads((out/'result.json').read_text())
    records=json.loads((out/'fit_records.json').read_text()); scale=d['normalizers']['z_scale'].numpy(); rows=[]
    lines=['# Ref9 per-arm GT and predicted signed contrasts','',
        'GT=current-state-adjusted factual randomized arm-minus-zero contrasts in166test windows. Pred=sameH mean model candidate arm-minus-zero; no individual GT counterfactual labels. All26axes and factual prediction contrasts remain in JSON.','',
        '| Arm | GT dz mm | Pred dz mm | GT own logforce | Pred own logforce | GT own distance mm | Pred own distance mm | I correlation | I amplitude ratio |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for j,name in enumerate(PER_FINGER_ARM_NAMES[1:]):
        body=min(j//2,4); gt=d['GT_contrasts'][j]; predicted=d['candidate_contrasts']['Ha'][j]
        metrics={key:contrast_scores(gt[None],predicted[None],scale,axes) for key,axes in (('E',np.arange(12)),('I',np.arange(12,26)))}
        rows.append(dict(arm=j+1,name=name,GT_E12=gt[:12].tolist(),GT_I14=gt[12:].tolist(),
            candidate_predictions={key:dict(E12=value[j,:12].tolist(),I14=value[j,12:].tolist()) for key,value in d['candidate_contrasts'].items()},
            factual_predictions={key:dict(E12=value[j,:12].tolist(),I14=value[j,12:].tolist()) for key,value in d['factual_contrasts'].items()},metrics=metrics))
        corr=metrics['I']['correlation']
        lines.append(f'| {name} | {gt[2]*1000:+.2f} | {predicted[2]*1000:+.2f} | {gt[12+body]:+.3f} | {predicted[12+body]:+.3f} | {gt[20+body]*1000:+.2f} | {predicted[20+body]*1000:+.2f} | {corr:.3f} | {metrics["I"]["amplitude_ratio"]:.3f} |')
    payload=dict(E_axes=E_NAMES,I_axes=I_NAMES,arm_contrasts=rows,
        crosshalf={key:dict(contrast_status=r['crosshalf'][key]['contrast_status'],
            model_candidate_E12=value[:,:12].tolist(),model_candidate_I14=value[:,12:].tolist(),
            GT_adjusted_contrasts=None if r['crosshalf'][key]['rank']<14 else d['crosshalf_GT'][key].tolist()) for key,value in d['crosshalf_candidates'].items()},
        limits='Physical proxy units, noisy marginal observed contrasts; not same-state causal mediation or load/slip identity')
    (out/'per_arm_contrasts.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'per_arm_contrasts.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    names=('H','Ha','Shuffled','Ha_test_shuffled'); x=np.arange(4)
    axs[0,0].bar(x-.17,[r['predictor_metrics'][n]['E_mse'] for n in names],width=.34,label='E12')
    axs[0,0].bar(x+.17,[r['predictor_metrics'][n]['I_mse'] for n in names],width=.34,label='I14')
    axs[0,0].set_xticks(x,names,rotation=15); axs[0,0].legend(); axs[0,0].set_title('Held-out consequence error (lower better)')
    diag=json.loads((out/'anomaly_diagnostics.json').read_text())
    axs[0,0].axhline(diag['fixed_baselines']['train_mean']['I_mse'],linestyle='--',color='grey',label='mean I')
    gt=d['GT_contrasts'][:,12:]/scale[12:]
    for name in ('Ha','Shuffled'):
        pred=d['candidate_contrasts'][name][:,12:]/scale[12:]
        axs[0,1].scatter(gt.flatten(),pred.flatten(),label=name,alpha=.4,s=12)
    axs[0,1].plot([-2,2],[-2,2],'k--',linewidth=.5);axs[0,1].legend();axs[0,1].set_xlabel('GT adjusted I contrast');axs[0,1].set_ylabel('SameH model I contrast');axs[0,1].set_title('All signed196 arm/axis entries')
    tasks=('H','Ha','GT_HEI','P_H','P_Ha','P_Shuffled','Ha_P_Ha')
    axs[0,2].barh(tasks,[r['downstream_metrics'][n]['primary_mse'] for n in tasks]);axs[0,2].invert_yaxis();axs[0,2].set_title('OOF-trained task scorer: test primary MSE')
    for name in ('full_H','full_Ha','full_Shuffled'):
        axs[1,0].plot(records[name]['train_loss'],label=name)
    axs[1,0].set_yscale('log');axs[1,0].set_title('Fixed1500 predictor training epochs');axs[1,0].legend();axs[1,0].set_xlabel('optimizer update')
    comps=('P_Ha_vs_Ha','Ha_P_Ha_vs_Ha','P_Ha_vs_P_H','P_Ha_vs_P_Shuffled')
    c=[r['downstream_comparisons'][n]['primary'] for n in comps]
    v=np.array([n['gain'] for n in c])*100; lo=np.array([n['lower95'] for n in c])*100; hi=np.array([n['upper95'] for n in c])*100
    axs[1,1].errorbar(v,np.arange(len(c)),xerr=(v-lo,hi-v),fmt='o',capsize=4)
    axs[1,1].set_yticks(np.arange(len(c)),comps,fontsize=8);axs[1,1].axvline(0,color='grey');axs[1,1].set_xlabel('Fixed-fit env-bootstrap gain (%)');axs[1,1].set_title('95% intervals: extra value unresolved')
    original=r['contrasts']; centered=diag['centered_contrast_sensitivity']
    vals=[original[n]['I']['correlation'] for n in ('Ha','Shuffled')]; cv=[centered[n]['correlation'] for n in ('Ha','Shuffled')]
    axs[1,2].bar(np.arange(2)-.17,vals,width=.34,label='arm-minus-zero')
    axs[1,2].bar(np.arange(2)+.17,cv,width=.34,label='remove common offset')
    axs[1,2].set_xticks(np.arange(2),('Ha','Shuffled'));axs[1,2].legend();axs[1,2].set_title('Contrast correlation sensitivity');axs[1,2].axhline(0,color='grey',linewidth=.6)
    fig.suptitle(f'ref9 conditional consequence: UNCLEAR; A/B/C fail, R={r["oracle_retention"]["R"]:.3f}\nAction sensitivity exists, but current fit does not preserve reliable extra task value')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/'conditional_consequence.png',dpi=160);plt.close(fig)
    print('Exported all14 arms E12/I14 and standalone figure; no fit.')


if __name__=='__main__': main()
