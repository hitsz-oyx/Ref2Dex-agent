"""Check tau equivalence and time full versus hand-only frozen WAM inference."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='0');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch
    from wam_rl_policy_v2.scheduled_current_cache import RecordInputs
    from wam_rl_policy_v2.wam_v4_runtime import V4Runtime,v4_online_samples,collate_v4
    torch.set_num_threads(2)
    source=ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1'
    snap=ROOT/'outputs/ptv3-shared-wam-v4/local-control-snapshot-20261018-r1'
    meta=json.loads((source/'manifest.json').read_text());ids=[0,100,1000]
    records=[dict(source=meta['records'][i],lo=0,hi=len(meta['records'][i]['ticks']),geometry=j) for j,i in enumerate(ids)]
    runtime=V4Runtime(snap/'step12000-local.pt',[r['source'] for r in records],source,dense_manifest=snap/'surface-manifest.json')
    data=RecordInputs(source,records,runtime.geometry);report=[]
    for i in range(len(records)):
        item=data[i];size=min(64,len(item['selected']))
        samples=v4_online_samples(item['objects'][:size],item['hands'][:size],[runtime.geometry[i]]*size,goal_delta=item['goals'][:size])
        batch={k:v.cuda() for k,v in collate_v4(samples).items()}
        timings={};predictions={}
        for repeat in range(3):
            for name,fn in [('full',runtime.predict),('hand',runtime.predict_hand)]:
                torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.cuda.synchronize();start=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=fn(batch)
                predictions[name]=result['trajectory'].clone();del result;torch.cuda.synchronize()
                if repeat:timings.setdefault(name,[]).append(time.perf_counter()-start)
        error=float((predictions['full']-predictions['hand']).abs().max())
        if error>1e-6:raise ValueError('Hand tau changed: '+str(error))
        report.append(dict(record=ids[i],batch=size,max_tau_difference_m=error,mean_seconds={k:float(np.mean(v)) for k,v in timings.items()}))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(dict(status='PASS',rows=report),indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
