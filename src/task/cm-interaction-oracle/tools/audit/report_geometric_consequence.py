#!/usr/bin/env python3
"""Export all signed nominal-action consequences and a shareable probe figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--support-run',type=Path,required=True)
    args=parser.parse_args()
    destinations=[args.run_dir/name for name in ('per_arm_contrasts.json','per_arm_contrasts.md','geometric_consequence.png')]
    if any(path.exists() for path in destinations): raise FileExistsError('report already exists')
    d=torch.load(args.run_dir/'diagnostic.pt',map_location='cpu',weights_only=False)
    r=json.loads((args.run_dir/'result.json').read_text())
    s=json.loads((args.support_run/'result.json').read_text())
    arms=('index_plus','index_minus','middle_plus','middle_minus','pinky_plus','pinky_minus',
          'ring_plus','ring_minus','thumb_yaw_plus','thumb_yaw_minus','thumb_pitch_plus','thumb_pitch_minus','synergy_plus','synergy_minus')
    axes=['dx_m','dy_m','dz_m','world_rotvec_x_rad','world_rotvec_y_rad','world_rotvec_z_rad',
        'dvx_m_s','dvy_m_s','dvz_m_s','dwx_rad_s','dwy_rad_s','dwz_rad_s']
    axes += ['log1p_force_'+f for f in ('index','middle','pinky','ring','thumb')]
    axes += ['log1p_abs_object_force_'+a for a in ('x','y','z')]
    axes += ['distance_'+f+'_m' for f in ('index','middle','pinky','ring','thumb')]+['global_proxy_fraction']
    contrasts={name:(value[:,1:]-value[:,:1]).mean(0) for name,value in d['candidate_arrays'].items()}
    contrasts['GT']=d['GT_contrasts']
    payload=dict(axes=axes,axis_scale=d['target_scale'].tolist(),
        reference='All14 arms relative to zero intervention. Predictions are same-state candidate means; GT is current-adjusted factual OLS, not paired counterfactual ground truth.',
        arms=[dict(arm=name,raw={key:value[j].tolist() for key,value in contrasts.items()},
            normalized={key:(value[j]/d['target_scale'].numpy()).tolist() for key,value in contrasts.items()}) for j,name in enumerate(arms)])
    destinations[0].write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# Signed arm contrasts','',payload['reference'],'',
        '| Arm | GT I RMS | Flow I RMS | Joint I RMS | Flow I cosine |','| --- | --- | --- | --- | --- |']
    for j,name in enumerate(arms):
        g=contrasts['GT'][j,12:]/d['target_scale'].numpy()[12:]
        f=contrasts['Flow'][j,12:]/d['target_scale'].numpy()[12:]
        q=contrasts['Joint'][j,12:]/d['target_scale'].numpy()[12:]
        cos=float(g@f/max(np.linalg.norm(g)*np.linalg.norm(f),1e-12))
        lines.append(f'| {name} | {np.sqrt(np.mean(g*g)):.4f} | {np.sqrt(np.mean(f*f)):.4f} | {np.sqrt(np.mean(q*q)):.4f} | {cos:.4f} |')
    lines+=['','All26 signed raw and normalized axes for all models are retained in per_arm_contrasts.json.']
    destinations[1].write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    names=['TrainMean','Persistence','State','Arm','Joint','Flow','FlowShuffled']
    values=[r['predictor_metrics'][name]['I_mse'] for name in names]
    axs[0].barh(names,values,color=['#777777']*2+['#4f89a8']*3+['#bb6755']*2)
    axs[0].set_xlabel('Held-environment normalized I MSE (lower is better)')
    axs[0].set_title('Frozen primary representation screen')
    matrix=np.array([[r['predictor_metrics']['Flow']['I_mse'],s['metrics']['FlowAdditive']['I_mse']],
        [s['metrics']['FlowPhysicalProducts']['I_mse'],s['metrics']['FlowPhysicalAdditive']['I_mse']]])
    axs[1].imshow(np.log10(matrix),cmap='YlOrRd',vmin=0,vmax=1.2)
    axs[1].set_xticks([0,1],['State × action products','Additive only'])
    axs[1].set_yticks([0,1],['Per-PC standardization','Fixed 20 mm scale'])
    for row in range(2):
        for col in range(2):axs[1].text(col,row,f'{matrix[row,col]:.3f}',ha='center',va='center',fontsize=15)
    axs[1].set_title('Post-result fixed support diagnosis: I MSE')
    task_names=['H','GT','P_State','P_Flow','Flow','Flow_P_Flow']
    axs[2].barh(task_names,[r['downstream_metrics'][name]['primary_mse'] for name in task_names],color='#6b9184')
    axs[2].set_xlabel('Normalized primary task MSE (lower is better)')
    axs[2].set_title('OOF consequences vs matched direct controls')
    fig.suptitle('Ref10 exploratory probes: nominal endpoint point-flow; no trained-policy claim',fontsize=13)
    fig.tight_layout();fig.savefig(destinations[2],dpi=160);plt.close(fig)
    print('All14 signed arm vectors and standalone figure exported.')

if __name__=='__main__':main()
