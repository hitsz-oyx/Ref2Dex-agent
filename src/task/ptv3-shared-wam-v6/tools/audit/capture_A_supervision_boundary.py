#!/usr/bin/env python3
"""Post-failure CPU ingress-only evidence; never executes a model forward or update."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6.data import V6Windows,collate
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.goal_policy import UNKNOWN_GOAL_POLICY,GOAL_KEYS,assert_unknown_inputs
from ptv3_shared_wam_v6.events import UNKNOWN_EVENT_POLICY

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(out,batch_size=8,workers=0):
 torch.set_num_threads(2);out.mkdir(exist_ok=False,parents=True)
 entry=load('ingress_training',TASK/'tools/run/train_pretrain.py');prior=load('prior_probe',TASK/'tools/audit/audit_goal_unlock_readiness.py')
 c=json.loads((TASK/'configs/tau_no_goal_no_event_train_v1.json').read_text());r=ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'
 options=dict(geometry_gt_object=True,event_input_policy=UNKNOWN_EVENT_POLICY)
 ds=V6Windows(r,'train',goal_input_policy=UNKNOWN_GOAL_POLICY,curated_manifest=c['curated_manifest'],**options);raw=V6Windows(r,'train',**options)
 if workers:
  from ptv3_shared_wam_v6.loader_ipc import configure_loader_ipc
  configure_loader_ipc()
 options=dict(multiprocessing_context='spawn',timeout=30) if workers else {}
 batch=next(iter(DataLoader(ds,batch_sampler=entry.Steps(ds,batch_size,170,0,1,1),collate_fn=entry.collate,num_workers=workers,**options)))
 by_tuple={tuple(map(int,x)):i for i,x in enumerate(raw.index)};reference=collate([raw[by_tuple[tuple(map(int,x))]] for x in batch['sample_id'].tolist()])
 model=SharedWAMV6(json.loads(Path(json.loads(r.read_text())['stats']).read_text()),c,geometry=prior.UnusedGeometry()).cpu()
 captured=[]
 def hook(module,args):
  captured.append({k:v.detach().cpu().clone() for k,v in args[0].items()});raise prior.BoundaryCaptured()
 handle=model.register_forward_pre_hook(hook)
 try:model(batch)
 except prior.BoundaryCaptured:pass
 finally:handle.remove()
 assert len(captured)==1;actual=captured[0];assert_unknown_inputs(actual)
 excluded=set(GOAL_KEYS);equal={k:actual[k].numpy().tobytes()==reference[k].numpy().tobytes() for k in actual if k not in excluded};assert all(equal.values())
 mask_keys=('trajectory_valid','start_hand_valid','scene_valid','object_valid','interaction_valid','interaction_query_valid','dense_hand_valid')
 tensor_keys=GOAL_KEYS+mask_keys+('event_type','event_delta_steps','event_valid','event_hz','sample_id','source_id')
 np.savez(out/'supervision-boundary.npz',**{k:actual[k].numpy() for k in tensor_keys})
 detail={k:dict(dtype=str(actual[k].dtype),shape=list(actual[k].shape),nonzero=int(torch.count_nonzero(actual[k])),first_values=actual[k].flatten()[:8].tolist()) for k in tensor_keys}
 targets=('trajectory_target','scene_target','contact_target','part_target','geometry_object_future_gt')
 j=dict(optimizer_created=False,optimizer_updates=0,network_forward_executed=False,actual_model_class=type(model).__name__,workers=workers,local_batch=batch_size,capture='SharedWAMV6.forward_pre_hook',tensors=detail,unchanged_fields_bits=equal,
 target_sha256={k:hashlib.sha256(actual[k].numpy().tobytes()).hexdigest() for k in targets},model_mask_denominators=dict(hand=int((actual['trajectory_valid']&actual['start_hand_valid'][:,None]).sum()),scene=int((actual['scene_valid']&actual['object_valid'][:,:,None,None]).sum()),contact=int(actual['interaction_valid'].sum())))
 (out/'supervision-boundary.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j['model_mask_denominators']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=0);a=p.parse_args();run(a.output,a.batch_size,a.workers)
