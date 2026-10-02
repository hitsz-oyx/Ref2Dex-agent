"""Small semantic smoke: independently known SDK inverse poses and matched controls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.task.CmResidual.measured_geometry_barriers import relative_geometry
from scripts.fit_measured_geometry_barriers import features
q=np.array([[0.,0.,np.sqrt(.5),np.sqrt(.5)]]);root=np.zeros((1,13));root[:,:3]=[1.,2.,3.];root[:,3:7]=q
body=np.repeat(np.array([[[1.,3.,3.]]]),5,axis=1);quat=np.repeat(q[:,None,:],5,axis=1)
p,r=relative_geometry(root,body,quat);assert np.max(np.abs(p-[1,0,0]))<1e-12
assert np.max(np.abs(r.reshape(1,5,3,2)-np.eye(3)[:,:2]))<1e-12
shift=np.array([4.,-8.,2.]);other=root.copy();other[:,:3]+=shift;p2,r2=relative_geometry(other,body+shift,quat);assert np.max(np.abs(p2-p))<1e-12 and np.array_equal(r2,r)
d=dict(state=np.ones((2,70),np.float32),extra=np.arange(160,dtype=np.float32).reshape(2,80),action=np.ones((2,12),np.float32),cells=np.array([0,1]));m=np.zeros(80,np.float32);s=np.ones(80,np.float32)*100;st=np.zeros((12,70),np.float32);et=np.zeros((12,80),np.float32)
cm=features(d,m,s,st,et,'cm');no=features(d,m,s,st,et,'state_only');ge=features(d,m,s,st,et,'geometry_off_action');glob=features(d,m,s,st,et,'global_action')
assert cm.shape==(2,162) and np.array_equal(cm[:,:150],no[:,:150]) and not no[:,150:].any()
assert not ge[:,70:115].any() and not ge[:,135:150].any() and np.array_equal(ge[:,115:135],cm[:,115:135])
assert not glob[:,:70].any() and np.array_equal(glob[:,70:],cm[:,70:]);print('PASS known inverse rotation, common world translation, and all control feature boundaries (CPU tiny smoke)')
