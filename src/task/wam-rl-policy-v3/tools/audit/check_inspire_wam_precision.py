"""Real fixed-input batch/single gate for the full Inspire WAM cache."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--precision',choices=['bf16','fp32'],required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh audit output required')
    os.environ.update(CUDA_VISIBLE_DEVICES='7',TMPDIR=str(ROOT/'tmp/inspire-precision'),REF2DEX_DEEP_OPT='all')
    (ROOT/'tmp/inspire-precision').mkdir(exist_ok=True)
    import numpy as np
    import torch
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
    spec=importlib.util.spec_from_file_location('inspire_full_bc_inputs',TASK/'tools/run/inspire_full_bc.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2)
    meta=json.loads((a.cache/'manifest.json').read_text());runtime=V4Runtime(meta['wam'],meta['records'],a.cache,
        dense_manifest=a.cache/'surface-manifest.json',allow_geometry_replacement=True)
    item=module.RecordSamples([meta['sequences'][1]],runtime.geometry,64)[0]
    batch={k:v.to('cuda') for k,v in item['batches'][0].items()};single={k:v[:1] for k,v in batch.items()}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
        packed=runtime.predict(batch)['trajectory'].float();one=runtime.predict(single)['trajectory'].float()
        packed_repeat=runtime.predict(batch)['trajectory'].float();one_repeat=runtime.predict(single)['trajectory'].float()
    diff=float((packed[0]-one[0]).abs().max());repeat=float((packed-packed_repeat).abs().max());repeat_one=float((one-one_repeat).abs().max())
    result=dict(status='PASS' if max(diff,repeat,repeat_one)<=.001 else 'FAIL',precision=a.precision,
        batch=64,sequence=meta['sequences'][1]['sequence'],max_batch_single_m=diff,
        repeat_batch_m=repeat,repeat_single_m=repeat_one,limit_m=.001,
        fixture='same tensors sliced at batch dimension; original measured H4 and scheduler goal',wam=runtime.conditioning_identity)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise AssertionError('real batch/single1mm gate failed')


if __name__=='__main__':main()
