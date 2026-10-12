"""Actual native-runtime regression for qualified PD-R singleton execution."""
import argparse
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser();p.add_argument('--safe',action='store_true');a=p.parse_args()
ROOT=Path(__file__).resolve().parents[5]
os.environ['CUDA_VISIBLE_DEVICES']='2'
sys.path.insert(0,str(ROOT/'src/task/trajectory-policy/src'))
import numpy as np
import torch
from trajectory_policy.pd_retargeter import load_model,decode_target,command
from trajectory_policy.native_R_inference import calibrated_PD_forward
owner=ROOT/'outputs/trajectory-policy'
d=np.load(owner/'wam-ref2-execution-startup-anchor-20261012-r1/native/trajectory.npz')
import json
m=json.loads((owner/'wam-ref2-execution-startup-anchor-20261012-r1/native/manifest.json').read_text())
torch.set_float32_matmul_precision('highest')
net=load_model(torch.load(owner/'competent-pd-fit-20261011-r1/tau.pt',map_location='cuda:0'),'cuda:0')
s=torch.tensor(d['state_features'][0,:1],device='cuda');t=torch.tensor(d['tau_features'][0,:1],device='cuda')
q=torch.tensor(d['q'][0,1:2],device='cuda');off,scale=[torch.tensor(m['native_controller'][k],device='cuda') for k in ('offset','scale')]
with torch.no_grad():
 encoded=(calibrated_PD_forward(net,s,t) if a.safe else net(s,t))
 applied=command(decode_target(encoded,q),q,off,scale)[1][0].cpu().numpy()
error=float(np.abs(applied-d['applied'][0,1]).max())
print({'torch':torch.__version__,'safe':a.safe,'calibrated_first_action_max_error':error})
assert error<=1e-5, 'native singleton differs from the actual calibrated first control'
