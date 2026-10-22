"""Verify trajectory-only inference equals full forward on observed robot H4."""
import argparse,json,os,sys,time
from pathlib import Path
T=Path(__file__).resolve().parents[2];R=T.parents[2];sys.path[:0]=[str(T/'src'),str(R/'src/task/wam-rl-policy-v3/src')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,default=2);p.add_argument('--output',type=Path,required=True);a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
 import numpy as np,torch
 from scipy.spatial.transform import Rotation
 from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime
 from wam_rl_policy_v3.robot_adapter import InspireAdapter
 torch.set_num_threads(2)
 base=R/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1';row=json.loads((base/'panel.json').read_text())['records'][0]
 with np.load(Path(row['source'])/'trajectory.npz') as z:poses=z['obj'][:4];hands=InspireAdapter.canonical_hand(z['hand'][:4]);goals=poses[-1:].copy()
 obj=np.concatenate([poses[:,:3,3],Rotation.from_matrix(poses[:,:3,:3]).as_quat()],-1)
 event=dict(event_type=1,event_delta_steps=0.,event_valid=True,event_hz=30);records=[]
 for name,path in [('healthy','local-expanded-ddp-20261021-r1/training/step004371.pt'),('flow','identifiability-audit-20261022-r1/flow-step008500.pt')]:
  rt=OnlineV6Runtime(R/'outputs/ptv3-shared-wam-v6'/path,R/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',base/'geometry')
  args=(obj[None],hands[None],goals,[event],['dexplore/'+row['object']])
  original_encode=rt.model.encode;captured=[]
  def capture(batch):
   encoded=original_encode(batch);captured.append(encoded);return encoded
  rt.model.encode=capture
  full=rt.predict(*args);fulltime=rt.seconds;rt.forecast_only=True
  rt.model.encode=lambda batch:captured[0]
  only=rt.predict(*args);err=float(np.max(np.abs(full-only)))
  if err>1e-6:raise ValueError('shared-encoding decoder mismatch '+str(err))
  rt.model.encode=original_encode
  before=rt.seconds;fresh=rt.predict(*args)
  records.append(dict(model=name,shared_encoding_decoder_max_difference_m=err,
   full_seconds=fulltime,fresh_forecast_only_seconds=rt.seconds-before,
   independent_encode_difference_m=float(np.max(np.abs(full-fresh))),
   note="Decoder parity freezes the same encoder activation. Fresh encoder numerics are separately reported; no bitwise end-to-end claim.",
   finite=bool(np.isfinite(only).all())))
  del rt;torch.cuda.empty_cache()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='PASS',records=records,torch=torch.__version__,python=sys.executable,gpu=a.gpu),indent=2)+'\n');print(a.output.read_text())
if __name__=='__main__':main()
