"""Conservation smoke: free fall, weight support, mass scaling, damping."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.task.CmResidual.force_aware_impulse import impulse_target,force_prior

def main():
    g=np.array([0.,0.,-9.81]);dt=1/30;v=np.zeros((2,3));m=np.array([.002593613,.010374452])
    assert np.allclose(impulse_target(v,v+g*dt,g,dt),0)
    assert np.allclose(impulse_target(v,v,g,dt),[[0,0,1]]*2)
    force=-m[:,None]*g;assert np.allclose(force_prior(force,v,m,g),[[0,0,1]]*2)
    moving=np.array([[.3,-.2,.1],[.6,-.4,.2]]);accel=force/m[:,None]+g-.01*moving
    assert np.allclose(impulse_target(moving,moving+accel*dt,g,dt),force_prior(force,moving,m,g))
    print('PASS: free fall zero nongravity impulse, weight support one, correct mass normalization, declared first-order damping prior')

if __name__=='__main__':main()
