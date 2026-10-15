"""Compare recorded hand forecasts against motion-scale baselines; NumPy only."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np


def measures(hand,tick,raw_stride):
    future=hand[tick+np.arange(1,25)*raw_stride]
    delta=future-hand[tick]
    velocity=(hand[tick]-hand[max(tick-raw_stride,0)])*30
    cv=np.arange(1,25,dtype=np.float64)[:,None,None]/30*velocity[None]
    return dict(zero_mm=np.linalg.norm(delta,axis=-1)*1000,
        constant_velocity_mm=np.linalg.norm(delta-cv,axis=-1)*1000,
        wrist_motion_mm=np.linalg.norm(delta[:,0],axis=-1)*1000,
        relative_to_wrist_motion_mm=np.linalg.norm(delta-delta[:,:1],axis=-1)*1000)


def aggregate(samples,model_ade):
    summaries={}
    for key in samples[0]:
        values=np.stack([s[key] for s in samples])
        summaries[key]=dict(mean=float(values.mean()),p95=float(np.percentile(values,95)),
            final_horizon_mean=float(values[:,-1].mean()))
    summaries['model_hand_ADE_mm']=model_ade
    summaries['model_to_zero_ratio']=model_ade/summaries['zero_mm']['mean']
    summaries['model_to_constant_velocity_ratio']=model_ade/summaries['constant_velocity_mm']['mean']
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=Path(__file__).resolve().parents[5]
    probe=root/'outputs/ptv3-shared-wam-v2/inspire-action-representation-20261015-r1'
    old=root/'outputs/ptv3-shared-wam-v2/shared-actor-shared-20261015-r1'
    workspace=json.loads((probe/'train-geometry/encoder_workspace_admission.json').read_text())
    assert workspace==json.loads((old/'encoder_workspace_admission.json').read_text())
    ids=np.load(old/'validation.npy')
    assert np.array_equal(ids,np.load(probe/'train-geometry/validation.npy'))
    source=root/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1'
    corpus=json.loads((source/'manifest.json').read_text())
    lookup={r['sequence']:r for r in corpus['records']}
    keys=[(r['sequence'],int(t)) for r in workspace['val']['records'] for t in r['admitted_ticks']]
    assert len(keys)==workspace['val']['retained_windows']==10376
    @lru_cache(maxsize=96)
    def inspire_hand(name):
        with np.load(source/lookup[name]['path']/'trajectory.npz') as z:return z['hand'].copy()
    inspire=[measures(inspire_hand(keys[int(i)][0]),keys[int(i)][1],1) for i in ids]
    curve=json.loads((old/'curve.json').read_text())
    model=next(r for r in curve if r['step']==7000)['physical']['hand_all24_mm']['mean']
    result=dict(scope='Fixed prior DEV panels; same24 horizons at30Hz, own native landmarks; no cross-hand matched-task or causal claim',
        cpu_only=True,neural_inference=False,panels={},identities={})
    result['panels']['inspire_dev256']=aggregate(inspire,model)
    result['identities']['inspire_dev256']=[dict(sequence=keys[int(i)][0],tick=keys[int(i)][1]) for i in ids]
    run=root/'outputs/shadow-finetune/shadow-startup-sampling-20261015-r1'
    controls=json.loads((run/'controls.json').read_text())
    cache=Path(controls['validation_cache'])
    records=[r for r in json.loads((cache/'manifest.json').read_text())['records'] if r['split']=='dev']
    keys=[(i,int(t)) for i,r in enumerate(records) for t in r['ticks']]
    @lru_cache(maxsize=96)
    def shadow_hand(index):
        with np.load(cache/records[index]['path']) as z:return z['hand'].copy()
    physical=json.loads((run/'summary/shared-physical-readout.json').read_text())
    for panel,key,reference in [('shadow_common_dev64','validation_ids','common_dev'),
                                ('shadow_startup_dev64','startup_validation_ids','startup_dev')]:
        selected=[keys[int(i)] for i in controls[key]]
        samples=[measures(shadow_hand(i),tick,2) for i,tick in selected]
        model=physical['arms']['startup'][reference]['hand_all24_mm']['mean']
        result['panels'][panel]=aggregate(samples,model)
        result['identities'][panel]=[dict(record=i,tick=tick,object_code=records[i]['object_code']) for i,tick in selected]
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['panels'],indent=2))


if __name__=='__main__':main()
