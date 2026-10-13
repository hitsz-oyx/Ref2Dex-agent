"""Verify paired streams, census weighting, first-batch isolation and saved metrics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
TASK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(TASK/'tools/run'),str(TASK/'tools/audit')]
from train_three_domain import atomic_json
from oakink_wm.multisource import MixedWindows,mixed_indices
from audit_native_curation import masks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run
    m=json.loads((out/'manifest.json').read_text());r=json.loads((out/'result.json').read_text())
    if m['status']!='COMPLETED' or r['status']!='COMPLETED':raise ValueError('completed paired run required')
    base=MixedWindows(Path(m['data']),'train');indices=np.load(out/'train_indices.npy')
    if not np.array_equal(indices,mixed_indices(base,m['updates']*m['batch'],m['draw_seed'])):raise ValueError('original sampler stream differs')
    with np.load(Path(m['census'])/'oakink2.npz') as f:
        flags=np.zeros(len(indices),bool);oak=indices<base.offsets[1]
        flags[oak]=masks(f['metrics'][indices[oak]])['static_goal_far_moving_other_near']
    with np.load(out/'weight_contract.npz') as f:
        if not np.array_equal(flags,f['train_mismatch']):raise ValueError('actor weight selection differs from independent census')
        geometry=f['val_geometry']
    va=MixedWindows(Path(m['data']),'val');vi=np.load(out/'validation_indices.npy');sid=np.searchsorted(va.offsets,vi,side='right')-1
    groups={name:sid==i for i,name in enumerate(('oakink2','grab','arctic'))}
    vm=masks(geometry)['static_goal_far_moving_other_near']&(sid==0)
    groups.update(oakink2_mismatch=vm,oakink2_aligned_proxy=(sid==0)&~vm,oakink2_target_interaction=(sid==0)&(np.minimum(geometry[:,4],geometry[:,5])<=50))
    for arm in ('initial','original','downweight'):
        with np.load(out/(arm+'_val.npz')) as f:
            for name,mask in groups.items():
                if r['reports'][arm][name]['n']!=int(mask.sum()):raise ValueError('panel count differs')
                for k in f.files:
                    if mask.any() and abs(float(f[k][mask].mean())-r['reports'][arm][name][k])>1e-6:raise ValueError('metric differs')
    orig=json.loads((out/'original_contract.json').read_text());new=json.loads((out/'downweight_contract.json').read_text())
    if any(orig[k]!=new[k] for k in ('batch_digest','rng_digest','world','unweighted_actor')):raise ValueError('initial pairing not exact')
    if any(v!=0 for v in new['prediction_max_difference'].values()):raise ValueError('initial predictions differ')
    for arm in ('original','downweight'):
        curve=json.loads((out/(arm+'_curve.json')).read_text())['curve']
        if curve[-1]['step']!=m['updates']:raise ValueError('paired updates incomplete')
    report=dict(status='PASS',sample_count=len(indices),marked_windows=int(flags.sum()),marked_fraction=float(flags.mean()),same_original_sampler=True,initial_input_rng_prediction_world_exact=True,World_supervision_unchanged=True,validation_groups={k:int(v.sum()) for k,v in groups.items()},scope='saved stream/first-batch/output audit; not an independent second seed or causal grasp test')
    atomic_json(out/'root_weight_audit.json',report);print(json.dumps(report))

if __name__=='__main__':main()
