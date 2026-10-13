#!/usr/bin/env python3
"""Recompute saved wrist-probe metrics and draw the fixed comparison panels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();root=a.run
    manifest=json.loads((root/'manifest.json').read_text())
    result=json.loads((root/'result.json').read_text())
    if result['status']!='COMPLETED' or result['engineering_only']:
        raise ValueError('completed scientific probe required')
    step=manifest['updates_per_arm']
    labels=['initial','A_step%d'%step,'B_step%d'%step,'B_blank','B_shift16']
    data={label:dict(np.load(root/(label+'.npz'),allow_pickle=False)) for label in labels}
    metrics={label:json.loads((root/(label+'.json')).read_text()) for label in labels}
    keys=list(metrics['initial'])
    errors=[]
    for label in labels:
        for name in keys:
            d=data[label];target=d[name+'_target'];valid=d[name+'_valid']
            for field in ('target','valid'):
                if not np.array_equal(d[name+'_'+field],data['initial'][name+'_'+field]):
                    raise ValueError('evaluation target/mask mismatch')
            e=np.linalg.norm(d[name+'_tau'].astype(np.float64)-target.astype(np.float64),axis=-1)*1000
            for field,count in (('h1_mm',1),('first8_mm',8),('all24_mm',24)):
                v=valid[:,:count]
                rows=(e[:,:count]*v).sum((1,2,3))/np.maximum(v.sum((1,2,3)),1)
                errors.append(float(np.max(np.abs(rows-d[name+'_'+field]))))
                errors.append(abs(float(rows.mean())-metrics[label][name][field]))
                if not np.isfinite(rows).all():raise ValueError('nonfinite saved predictions')
    if max(errors)>1e-3:raise ValueError('independent metric mismatch')
    b=data['B_step%d'%step]
    sensitivity={}
    for label in ('B_blank','B_shift16'):
        sensitivity[label]={}
        for name in keys:
            valid=b[name+'_valid']
            delta=np.linalg.norm((data[label][name+'_tau']-b[name+'_tau']).astype(np.float64),axis=-1)*1000
            sensitivity[label][name]=float((delta*valid).sum()/max(valid.sum(),1))
            if name in ('oakink2','grab','arctic') and not np.array_equal(data[label][name+'_tau'],b[name+'_tau']):
                raise ValueError('masked human prediction changed under goal intervention')
    # Same physical queries, separately encoded at batch1 and batch16. Quantify
    # batching sensitivity rather than counting them as independent evidence.
    batching={}
    for label in labels:
        batching[label]={}
        for i in range(3):
            exact='recovery%d_exact_C_TRAIN'%i;contact='recovery%d_contact_TRAIN'%i
            d=data[label];valid=d[exact+'_valid'][0]
            if not np.array_equal(d[exact+'_target'][0],d[contact+'_target'][0]):
                raise ValueError('duplicated query target mismatch')
            e=np.linalg.norm(d[exact+'_tau'][0]-d[contact+'_tau'][0],axis=-1)*1000
            batching[label][exact]=dict(mean_valid_prediction_difference_mm=float(e[valid].mean()),
                first8_scalar_difference_mm=float(d[exact+'_first8_mm'][0]-d[contact+'_first8_mm'][0]))
    aa=metrics['A_step%d'%step];bb=metrics['B_step%d'%step]
    exact=['recovery%d_exact_C_TRAIN'%i for i in range(3)]
    primary={label:float(np.mean([metrics[label][n]['first8_mm'] for n in exact])) for label in labels}
    for arm in ('A','B'):
        if abs(primary[arm+'_step%d'%step]-result['primary_'+arm+'_mm'])>1e-8:
            raise ValueError('primary aggregation mismatch')
    batch16_primary={label:float(np.mean([data[label]['recovery%d_contact_TRAIN_first8_mm'%i][0]
        for i in range(3)])) for label in labels}
    receipt=dict(pass_=True,max_metric_difference_mm=max(errors),target_and_mask_equal=True,
        human_mask_exact=True,primary_mm=primary,reference_prediction_change_mm=sensitivity,
        batch16_same_query_primary_mm=batch16_primary,batching_sensitivity=batching,
        cpu_reason='small saved-array arithmetic and plotting, no model inference',
        scope='independent output arithmetic and masking audit; no new heldout or physical evidence')
    (root/'independent_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for ax,panel,title in ((axes[0],exact,'Exact C takeover queries (TRAIN, n=3)'),
            (axes[1],['recovery%d_contact_TRAIN'%i for i in range(3)],'Contact16: 1 C + 15 expert queries (TRAIN)'),
            (axes[2],['nominal','startup'],'Nominal development states')):
        x=np.arange(len(panel))
        for offset,label,name in ((-.25,'initial','Parent'),(0,'A_step%d'%step,'A: no wrist goal'),(.25,'B_step%d'%step,'B: planned wrist')):
            ax.bar(x+offset,[metrics[label][n]['first8_mm'] for n in panel],.24,label=name)
        ax.set(xticks=x,xticklabels=['t32','t40','t48'] if len(panel)==3 else panel,ylabel='First8 hand EPE (mm)',title=title)
        ax.legend(fontsize=7)
    fig.suptitle('Fixed 1000-update comparison; offline fit is not grasp success')
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
