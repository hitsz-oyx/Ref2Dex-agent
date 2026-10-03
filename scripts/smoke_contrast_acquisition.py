"""Tiny CPU algebra smoke, not scientific evidence."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.task.CmResidual.contrast_acquisition import scores
rng=np.random.default_rng(4560);p=rng.normal(size=(3,5,7,3));offset=rng.normal(size=(3,5,1,3))
np.testing.assert_allclose(scores(p)['contrast'],scores(p+offset)['contrast'],atol=1e-12)
assert not np.allclose(scores(p)['absolute'],scores(p+offset)['absolute'])
y=rng.normal(size=(7,3));da=rng.normal(size=(3,3));db=rng.normal(size=(3,3))
tau=(y[1::2]-y[2::2]).T;observations=[]
for arm in range(7):
 z=np.zeros((3,3))
 if arm:z[:,(arm-1)//2]=y[arm]*7*(1 if arm%2 else -1)
 observations.append(((da*da-db*db)-2*(da-db)*z).sum())
np.testing.assert_allclose(np.mean(observations),((da-tau)**2-(db-tau)**2).sum(),atol=1e-12)
print('PASS common-offset cancellation and exact population randomized relative-risk identity')
