"""Recompute held-object metrics and the predeclared readout qualification gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    output=a.run/'audit'
    if output.exists():raise FileExistsError(output)
    output.mkdir()
    d=torch.load(a.run/'features/features.pt',map_location='cpu',weights_only=False)
    results=json.loads((a.run/'fit/results.json').read_text())
    manifest=json.loads((a.run/'fit/manifest.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['steps']==1000
    fit=d['fit'];target=d['targets'];rows={}
    code=np.array([d['objects'][i]['object_code'] for i in d['object_id'].tolist()])
    assert set(code[fit]).isdisjoint(set(code[~fit]))
    for arm,result in results.items():
        packet=torch.load(a.run/'fit'/(arm+'.pt'),map_location='cpu',weights_only=False)
        pred=packet['prediction'];scale=packet['normalization']['target_scale']
        assert torch.isfinite(pred).all() and pred.shape==target.shape
        per_object={}
        for name in sorted(set(code[~fit])):
            selected=torch.from_numpy(code==name)
            error=pred[selected]-target[selected]
            per_object[name]=float(error[:,0].square().mean())
        macro=float(np.mean(list(per_object.values())))
        physics=float(((pred[~fit,1:]-target[~fit,1:])/scale[1:]).square().mean())
        stored=result['metrics']['heldout_objects']
        np.testing.assert_allclose(macro,stored['return_macro_mse'],rtol=1e-5)
        np.testing.assert_allclose(physics,stored['physics_normalized_mse'],rtol=1e-5)
        rows[arm]=dict(heldout_return_macro_mse=macro,physics_normalized_mse=physics,
            fit_return_mse=result['metrics']['fit']['return_macro_mse'],
            parameters=result['parameters'],per_object=per_object)
    decisions={}
    for arm in ['wam-query-aux','wam-query-forecast-aux']:
        r=rows[arm];raw=rows['raw-query-aux'];mean=rows['wam-mean-aux']
        gain_raw=1-r['heldout_return_macro_mse']/raw['heldout_return_macro_mse']
        gain_mean=1-r['heldout_return_macro_mse']/mean['heldout_return_macro_mse']
        physical_ratio=r['physics_normalized_mse']/mean['physics_normalized_mse']
        passed=gain_raw>=.1 and gain_mean>=.1 and physical_ratio<=1.05
        decisions[arm]=dict(gain_over_raw=gain_raw,gain_over_mean=gain_mean,
            physics_ratio_to_mean=physical_ratio,gate_pass=passed,
            object_wins_over_raw=sum(v<raw['per_object'][k] for k,v in r['per_object'].items()),
            object_wins_over_mean=sum(v<mean['per_object'][k] for k,v in r['per_object'].items()))
    audit=dict(status='PASS',phase='Probe',conclusion='PROMISING' if any(d['gate_pass'] for d in decisions.values()) else 'UNPROMISING',
        scope='Fixed offline task-readout qualification only; not WAM validity or policy improvement',
        samples=len(fit),fit_samples=int(fit.sum()),heldout_samples=int((~fit).sum()),
        groups=len(set(code[fit])),heldout_groups=len(set(code[~fit])),rows=rows,decisions=decisions,
        clock_baseline=manifest['clock_baseline']['heldout_objects']['return_macro_mse'])
    (output/'summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Raw query\n+ physics','WAM mean\n+ physics','WAM query\nreturn only','WAM query\n+ physics','WAM query\n+ forecast + physics']
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    values=[r['heldout_return_macro_mse'] for r in rows.values()]
    axes[0].bar(range(len(values)),values,color=['#888888','#7793b3','#b8b1d0','#3c7fa4','#2d8f76'])
    axes[0].axhline(audit['clock_baseline'],color='gray',linestyle=':',label='TRAIN clock baseline')
    axes[0].set_xticks(range(5),labels,fontsize=8)
    axes[0].set_ylabel('Held-object return MSE (lower is better)')
    axes[0].legend(fontsize=8)
    for arm,label in zip(results,labels):
        history=results[arm]['history']
        axes[1].plot([r['step'] for r in history],
            [r['metrics']['heldout_objects']['return_macro_mse'] for r in history],label=label.replace('\n',' '))
    axes[1].set_xlabel('Readout gradient steps')
    axes[1].set_ylabel('Held-object return MSE')
    axes[1].legend(fontsize=7)
    fig.suptitle('TRAIN-internal qualification: 24 fit / 8 held objects, one seed, final endpoint')
    fig.tight_layout()
    fig.savefig(output/'readout.png',dpi=180)
    fig.savefig(output/'readout.pdf')
    print(json.dumps({k:v for k,v in audit.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
