"""Merge the two native-current workers, enforcing the matched design."""
import argparse,json
from pathlib import Path
import numpy as np


def summarize(out):
    arms=('baseline','native_current')
    manifests={a:json.loads((out/a/'manifest.json').read_text()) for a in arms}
    results={};curves={}
    for arm in arms:
        directory=out/arm
        m=manifests[arm]
        assert m['status']=='COMPLETED'
        for k in ('parent_step','parent_checkpoint','source','updates','batch','learning_rate','seeds','windows','R_normalizers'):
            assert m[k]==manifests['baseline'][k],(arm,k)
        for name in ('panels.npz','extra_train_stats.npz'):
            candidate=np.load(directory/name);base=np.load(out/'baseline'/name)
            assert set(candidate.files)==set(base.files)
            for k in base.files: np.testing.assert_array_equal(candidate[k],base[k])
        assert json.loads((directory/'panel_membership.json').read_text())==json.loads((out/'baseline'/'panel_membership.json').read_text())
        r=json.loads((directory/'result.json').read_text())['arms'][arm]
        curve=json.loads((directory/(arm+'-curve.json')).read_text())
        assert r['selected']==min(curve,key=lambda v:(v['panels']['val']['xyz_p95_mm'],v['panels']['val']['xyz_mean_mm']))
        assert r['initial']==curve[0] and r['final']==curve[-1]
        assert [v['step'] for v in curve]==list(range(0,m['updates']+1,500))
        results[arm]=r;curves[arm]=curve
    for split in ('train','val'):
        for k in ('xyz_mean_mm','xyz_p95_mm','normalized_all8_L1','clipping'):
            np.testing.assert_allclose(results[arms[0]]['initial']['panels'][split][k],results[arms[1]]['initial']['panels'][split][k],rtol=1e-5,atol=1e-5)
    base=results['baseline']['selected']['panels']['val'];native=results['native_current']['selected']['panels']['val']
    ratios={k:native[k]/base[k] for k in ('xyz_mean_mm','xyz_p95_mm')}
    promising=all(v<=.85 for v in ratios.values()) and native['clipping']<.01
    payload=dict(status='COMPLETED',conclusion='PROMISING' if promising else 'UNCLEAR',arms=results,ratios=ratios,
                 matched_checks='same parent, extra TRAIN stats, panels and sequence/tick identities, initial functions, hyperparameters; original selection retained',
                 scope='single-seed GT-tau offline R-only native delta0 information diagnostic; no predicted-tau/grasping claim')
    path=out/'result.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(payload,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,k,title in zip(axes,('xyz_mean_mm','xyz_p95_mm'),('Mean','95th percentile')):
        for arm in arms: ax.plot([r['step'] for r in curves[arm]],[r['panels']['val'][k] for r in curves[arm]],marker='o',label=arm)
        ax.set(xlabel='Matched R updates',ylabel='First PD wrist XYZ error (mm)',title=title);ax.grid(alpha=.25)
    axes[0].legend();fig.suptitle('Native DExplore current inputs; GT tau, fixed new dev512')
    fig.savefig(out/'native-current.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(conclusion=payload['conclusion'],ratios=ratios,selected={a:results[a]['selected']['panels']['val'] for a in arms})))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    summarize(p.parse_args().output)
