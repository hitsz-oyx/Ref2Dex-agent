"""Bounded GPU engineering timing on fixed full-cache records; no fitting."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy.scheduled_current_cache import RecordInputs
from wam_rl_policy.wam_v4_runtime import V4Runtime,v4_online_samples,collate_v4
from wam_rl_policy.tau_contract import v4_trajectory_to_student

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',default='6');a=p.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;torch.set_num_threads(2)
 root=TASK.parents[2];source=root/'outputs/shadow-finetune/policy-data-full-20261015-r1';snap=root/'outputs/ptv3-shared-wam-v4/local-control-snapshot-20261018-r1'
 meta=json.loads((source/'manifest.json').read_text());ids=[0,100,1000,4000,7000,10000]
 records=[dict(source=meta['records'][i],lo=0,hi=len(meta['records'][i]['ticks']),geometry=j) for j,i in enumerate(ids)]
 runtime=V4Runtime(snap/'step12000-local.pt',[r['source'] for r in records],source,dense_manifest=snap/'surface-manifest.json')
 dataset=RecordInputs(source,records,runtime.geometry);report=[]
 for index in range(len(records)):
  t=time.perf_counter();item=dataset[index];scheduler_s=time.perf_counter()-t;geometry=runtime.geometry[index]
  references=None
  for size in (32,128,256):
   times=dict(scheduler_s=scheduler_s,inputs_s=0.,gpu_s=0.,tau_s=0.);predicted=[];torch.cuda.reset_peak_memory_stats()
   for lo in range(0,len(item['selected']),size):
    hi=min(lo+size,len(item['selected']));t=time.perf_counter()
    samples=v4_online_samples(item['objects'][lo:hi],item['hands'][lo:hi],[geometry]*(hi-lo),goal_delta=item['goals'][lo:hi])
    batch={k:v.cuda() for k,v in collate_v4(samples).items()};torch.cuda.synchronize();times['inputs_s']+=time.perf_counter()-t;t=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=runtime.predict(batch)
    trajectory=pred['trajectory'].float().cpu().numpy();torch.cuda.synchronize();times['gpu_s']+=time.perf_counter()-t;t=time.perf_counter()
    tau=np.stack([v4_trajectory_to_student(v,object_rotation=r) for v,r in zip(trajectory,item['rotations'][lo:hi])]);times['tau_s']+=time.perf_counter()-t;predicted.append(tau)
   tau=np.concatenate(predicted)
   if references is None:references=tau
   row=dict(record=ids[index],batch=size,rows=len(tau),**times,peak_mib=torch.cuda.max_memory_allocated()/2**20,max_difference_vs_batch32_m=float(np.abs(tau-references).max()))
   report.append(row);print(json.dumps(row),flush=True)
   a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
