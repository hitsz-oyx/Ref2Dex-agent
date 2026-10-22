import importlib.util
from pathlib import Path
import numpy as np
import torch

spec=importlib.util.spec_from_file_location('identifiability_audit',Path(__file__).resolve().parents[1]/'tools/audit/audit_identifiability.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_disjoint_panel_excludes_overlap_and_caps_capture_count():
    rows=[dict(capture='a',tick=t,phase_group=t%3) for t in range(3,130)]
    picked=audit.disjoint_pick(rows,96,np.random.default_rng(178),max_per_capture=3)
    assert len(picked)<=3
    for i,row in enumerate(picked):
        assert all(abs(row['tick']-other['tick'])>=28 for other in picked[:i])


def test_pair_variance_trace_has_one_half_factor_and_splits_horizon():
    h=torch.zeros(2,24,2,21,3);s=torch.zeros(2,24,32,3)
    h[1,:8,...,0]=.002;h[1,8:,...,0]=.004;s[1,...,0]=.006
    result=audit.pair_variances(h,s,torch.ones(2,2,dtype=torch.bool),np.array([[0,1]]))[0]
    np.testing.assert_allclose(result,[2.,6.,18.,18.],rtol=1e-5)


def test_h4_history_reconstruction_preserves_encoded_dynamics():
    x=np.arange(4,dtype=np.float32)[:,None,None]*np.ones((4,2,3),np.float32)
    f=np.concatenate(((x[3]-x[2])*30,(x[3]-2*x[2]+x[1])*900,x[3]-x[0]),-1)
    np.testing.assert_allclose(audit.reconstruct_history(x[3],f),x)
