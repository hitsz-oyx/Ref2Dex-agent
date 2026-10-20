"""Archived real-geometry GPU equivalence check for vectorized WAM inputs."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',default='6');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch
    from wam_rl_policy_v3.scheduled_current_cache import RecordInputs
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime,v4_online_samples,collate_v4
    from wam_rl_policy_v3.batched_geometry import online_batch
    torch.set_num_threads(2)
    source=ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1';snap=ROOT/'outputs/ptv3-shared-wam-v4/local-control-snapshot-20261018-r1'
    meta=json.loads((source/'manifest.json').read_text());ids=[0,100,1000]
    records=[dict(source=meta['records'][i],lo=0,hi=len(meta['records'][i]['ticks']),geometry=j) for j,i in enumerate(ids)]
    runtime=V4Runtime(snap/'step12000-local.pt',[r['source'] for r in records],source,dense_manifest=snap/'surface-manifest.json')
    data=RecordInputs(source,records,runtime.geometry);report=[]
    for i in range(len(records)):
        item=data[i];size=min(32,len(item['selected']));objects=item['objects'][:size];hands=item['hands'][:size];goals=item['goals'][:size];geometry=[runtime.geometry[i]]*size
        reference={k:v.cuda() for k,v in collate_v4(v4_online_samples(objects,hands,geometry,goal_delta=goals)).items()}
        candidate=online_batch(objects,hands,geometry,goals,'cuda')
        errors={}
        for key,value in reference.items():
            torch.testing.assert_close(candidate[key],value,rtol=1e-6,atol=2e-6)
            errors[key]=float((candidate[key].double()-value.double()).abs().max())
        predictions=[];configured_defer=runtime.defer_checks
        for index,batch in enumerate([reference,reference,candidate]):
            runtime.defer_checks=configured_defer if index==2 else False
            torch.manual_seed(0);torch.cuda.manual_seed_all(0)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                predictions.append(runtime.predict(batch)['trajectory'].clone())
        if configured_defer:
            # A valid nonfinite goal must still fail before a prediction escapes.
            bad={key:value.clone() for key,value in candidate.items()}
            bad['goal_object_delta'][0,0,0]=float('nan')
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):runtime.predict(bad)
            except ValueError as error:
                if 'nonfinite valid observation' not in str(error):raise
            else:raise AssertionError('deferred finite checks did not reject invalid goal')
        report.append(dict(record=ids[i],deferred_checks=configured_defer,batch=size,input_max_abs_difference=errors,
            reference_repeat_tau_max_m=float((predictions[0]-predictions[1]).abs().max()),
            candidate_tau_max_m=float((predictions[1]-predictions[2]).abs().max())))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(dict(input_status='PASS',rows=report),indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
