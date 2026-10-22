#!/usr/bin/env python3
"""CPU resampling of persisted paired errors; no model inference or training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(TASK/'src'), str(TASK/'tools/run')]
from score_tau_panel_trivial_baselines import paired_summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    old=json.loads(a.reference.read_text())
    with np.load(a.records, allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    masks=dict(overall=data['frame_valid'],
        first_8_frames=data['frame_valid'] & (np.arange(24)[None,:]<8),
        contact_window=data['frame_valid'] & data['contact_frame'])
    for phase in sorted(set(data['phase'])):
        masks['phase/'+phase]=data['frame_valid'] & (data['phase'][:,None]==phase)
    subsets={}
    for name,mask in masks.items():
        seed=(173000+len(name.split('/')[1])) if name.startswith('phase/') else 172000+len(name)
        result=paired_summary(data['b1_error_mm'],data['b0a_error_mm'],mask,seed,sequence_ids=data['sequence_id'])
        previous=old['paired_frame_analysis']['subsets'][name]
        assert result['frame_count']==previous['frame_count']
        assert abs(result['win_rate_excluding_ties']-previous['win_rate_excluding_ties'])<1e-8
        frame=result['paired_difference_median_frame_bootstrap']['ci95_mm']
        assert np.allclose(frame,previous['paired_difference_median_bootstrap']['ci95_mm'],atol=1e-5,rtol=0)
        cluster=result['paired_difference_median_bootstrap']['ci95_mm']
        result['frame_only_significance']=bool(frame[1]<0 and cluster[0]<=0<=cluster[1])
        result['b1_advantage_supported_by_both_ci']=bool(frame[1]<0 and cluster[1]<0)
        subsets[name]=result
    result=dict(schema='ref2dex.tau-cluster-ci.v1',run_id='tau-contract-20261023-r1',
        records=str(a.records),rows=len(data['sequence_id']),mother_sequences=len(np.unique(data['sequence_id'])),
        primary='mother-sequence cluster bootstrap; 2000 replicates per subset',
        comparison='frame bootstrap retained with historical seeds and exact statistic',
        estimand='pooled paired frame median B1 minus B0a; all windows/objects/frames of a mother travel together',
        scope='fixed TRAIN anchor, historical goal and future-GT event conditions; not causal deployment evidence',
        subsets=subsets,training_updates=0)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:dict(frames=v['frame_count'],frame=v['paired_difference_median_frame_bootstrap']['ci95_mm'],
        cluster=v['paired_difference_median_bootstrap']['ci95_mm'],
        observed_clusters=v['paired_difference_median_bootstrap']['clusters_with_values'],
        frame_only_significance=v['frame_only_significance']) for k,v in subsets.items()},indent=2))


if __name__=='__main__': main()
