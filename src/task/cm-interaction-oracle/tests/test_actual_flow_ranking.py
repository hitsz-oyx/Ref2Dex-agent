import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from actual_flow_ranking import anchor_folds, partition, rows_for, readout_input, ranking, bootstrap_difference


def test_whole_anchor_split_across_offsets_and_candidates():
    keys = [(271,a,0,t) for a in range(16) for t in (0,8,16)]
    folds = anchor_folds(keys,4,271)
    for a in range(16):
        assert len(set(folds[a*3:a*3+3])) == 1
    for f in range(4):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        partition(keys,tr,te)
        assert not set(rows_for(tr,'cpu').tolist()) & set(rows_for(te,'cpu').tolist())
    with pytest.raises(ValueError,match='leakage'):
        partition(keys,[0],[1])


def test_bottleneck_has_no_direct_flow_and_h_is_candidate_constant():
    h = torch.ones(7,2); flow = torch.arange(7.).unsqueeze(1); ei = torch.zeros(7,26)
    b = readout_input(h,flow,ei,'Bottleneck')
    assert torch.equal(b,readout_input(h,flow+100,ei,'Bottleneck'))
    assert not torch.equal(readout_input(h,flow,ei,'Direct'),readout_input(h,flow+100,ei,'Direct'))
    y = torch.zeros(7,8); y[1,7] = 1
    out = ranking(y, torch.zeros_like(y))
    assert out['pairwise_accuracy'] == .5 and out['strict_pairs'] == 6
    assert out['mean_top1_regret'] == 1


def test_cluster_bootstrap_excludes_ties_and_counts_informative_anchors():
    y = torch.zeros(14,8); y[1,7] = 1
    out = bootstrap_difference(y,y,torch.zeros_like(y),[(271,0,0,0),(271,1,0,0)])
    assert out['difference'] == .5
    assert out['strict_pair_anchors'] == 1
    assert out['valid_bootstrap_fraction'] < 1
