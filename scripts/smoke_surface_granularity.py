"""Two-window/three-update engineering check; not scientific performance evidence."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.surface_granularity_prior import encode_granularity,GranularityPrior
from src.task.CmResidual.surface_motion_prior import encode_raw,numpy_predict
from scripts.audit_surface_granularity_prior import independent_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);a=p.parse_args()
    with np.load(a.packet) as f:raw={k:f[k][:2] for k in f.files}
    old,_=encode_raw(raw);x,_=encode_granularity(raw,'mean')
    collapsed=np.concatenate((x[:,:,:6],x[:,:,6:9],x[:,:,18:21],x[:,:,30:33],x[:,:,42:]),axis=-1)
    assert np.allclose(collapsed,old,atol=2e-6)
    # Apply the synthetic global rigid transform in float64, so rounding world
    # positions before displacement subtraction cannot pollute a frame test.
    w=Rotation.from_rotvec([.31,-.12,.23]).as_matrix();shift=np.array([.2,-.3,.15])
    transformed={k:v.astype(np.float64) for k,v in raw.items()}
    for k in ('obj','previous_obj','next_obj','hand','next_hand'):transformed[k]=raw[k].astype(np.float64)@w.T+shift
    for k in ('normal','hand_normal','global_hand_flow'):transformed[k]=raw[k].astype(np.float64)@w.T
    transformed['pose'][:,:3,:3]=w@raw['pose'][:,:3,:3].astype(np.float64)
    transformed['pose'][:,:3,3]=raw['pose'][:,:3,3].astype(np.float64)@w.T+shift
    frame_max=0.
    for agg in ('mean','detail'):
        x,y=encode_granularity(raw,agg);i,j=independent_features(raw,agg)
        assert np.allclose(x,i,atol=2e-5,rtol=2e-6) and np.allclose(y,j,atol=2e-5,rtol=2e-6)
        xx,yy=encode_granularity(transformed,agg);frame_max=max(frame_max,float(np.abs(xx-x).max()),float(np.abs(yy-y).max()))
        assert np.allclose(xx,x,atol=2e-5,rtol=2e-6) and np.allclose(yy,y,atol=2e-5,rtol=2e-6)
        altered=dict(raw,next_obj=raw['next_obj']+.01)
        ax,_=encode_granularity(altered,agg);assert np.array_equal(ax,x)
    permuted=dict(raw)
    for k in ('hand','hand_normal','next_hand'):permuted[k]=raw[k][:,:,::-1].copy()
    x,_=encode_granularity(raw,'mean');xx,_=encode_granularity(permuted,'mean');assert np.allclose(x,xx,atol=1e-6)
    x,_=encode_granularity(raw,'detail');xx,_=encode_granularity(permuted,'detail');assert np.max(np.abs(x-xx))>1e-3
    torch.set_num_threads(2);torch.manual_seed(3903);model=GranularityPrior();opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    x,y=encode_granularity(raw,'detail')
    for _ in range(3):
        opt.zero_grad();loss=(model(torch.from_numpy(x))-torch.from_numpy(y)).square().mean();loss.backward();opt.step()
    with torch.no_grad():predicted=model(torch.from_numpy(x)).numpy()
    assert np.isfinite(predicted).all() and np.max(np.abs(predicted))>0
    network_max=float(np.max(np.abs(predicted-numpy_predict(model.state_dict(),x))));assert network_max<2e-4
    print(json.dumps(dict(engineering_smoke='PASS',cpu_reason='2windows/3updates avoid GPU startup',
                         parameters=sum(p.numel() for p in model.parameters()),se3_max=frame_max,numpy_max=network_max,
                         mean_equivalent_to_old22=True,detail_retains_distinctions=True)),flush=True)


if __name__=='__main__':main()
