"""Tiny exact-moment algebra smoke; deliberately detects a missing correction."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.task.CmResidual.return_corrected_gradient import raw_gradients

n=12;epsilon=np.concatenate((np.eye(n),-np.eye(n)))*np.sqrt(n)
assert np.allclose(epsilon.T@epsilon/len(epsilon),np.eye(n),atol=1e-14,rtol=0)
derivative=np.broadcast_to(np.linspace(-.7,.8,n),epsilon.shape)
reward=epsilon@np.linspace(.5,-.4,n);baseline=np.zeros(len(epsilon))
base=raw_gradients(reward,baseline,epsilon,np.zeros_like(derivative))
corrected=raw_gradients(reward,baseline,epsilon,derivative)
assert np.allclose(base.mean(0),corrected.mean(0),atol=1e-14,rtol=0)
assert not np.allclose(base.mean(0),(corrected-derivative).mean(0))
print('PASS exact first/second-moment control correction; dropping +b changes the expected gradient. CPU algebra smoke, no scientific data.')
