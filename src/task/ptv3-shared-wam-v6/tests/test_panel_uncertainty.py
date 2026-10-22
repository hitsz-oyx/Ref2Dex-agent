import numpy as np
from ptv3_shared_wam_v6.panel_uncertainty import cluster_median_ci


def test_cluster_median_matches_explicit_whole_mother_resampling():
    values=np.array([-2., -1., 0., 10.])
    mothers=np.array([4, 4, 4, 9])
    result=cluster_median_ci(values,mothers,cluster_universe=[4,9],seed=170)
    rng=np.random.default_rng(170)
    medians=[]
    for _ in range(2000):
        chosen=rng.integers(2,size=2)
        medians.append(np.median(np.concatenate([values[mothers==[4,9][i]] for i in chosen])))
    assert result['median_mm']==np.median(values)
    assert result['median_mm']!=np.median([-1.,10.])
    assert np.array_equal(result['ci95_mm'],np.percentile(medians,[2.5,97.5]))


def test_repeated_frames_do_not_create_independent_cluster_evidence():
    values=np.r_[np.full(100,-1.),np.full(10,.5)]
    mothers=np.r_[np.full(100,4),np.full(10,9)]
    result=cluster_median_ci(values,mothers,cluster_universe=[4,9],seed=170)
    assert result['ci95_mm'][0]<0<result['ci95_mm'][1]
    rng=np.random.default_rng(170)
    naive=np.median(values[rng.integers(len(values),size=(2000,len(values)))],axis=1)
    assert np.percentile(naive,97.5)<0
