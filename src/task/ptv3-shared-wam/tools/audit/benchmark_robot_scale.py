"""Short real-input GPU benchmarks for capacity/batch/implementation decisions."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--checkpoint-interaction',action='store_true')
    p.add_argument('--cases',help='comma-separated case names; defaults to all')
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--layers',type=int,default=16)
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam' not in out.parents or not 4<=a.steps<=16:
        raise ValueError('fresh owned output and4..16 measured steps required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    out.mkdir(parents=True)
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1',
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shared-wam-extensions'))
    import numpy as np
    import torch
    from ptv3_shared_wam.checkpoint import load_checkpoint,atomic_json
    from ptv3_shared_wam.model import SharedWAM
    from ptv3_shared_wam.robot import RobotPair,robot_loss
    from ptv3_shared_wam.expansion import expand_forecaster
    from ptv3_shared_wam.scene import SceneHistoryEncoder
    from ptv3_shared_wam.rl_robot import RLRobotWindows
    from ptv3_shared_wam.data import shared_collate
    from ptv3_shared_wam.loading import to_device
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    parent=load_checkpoint(a.checkpoint)
    if parent['stage']!='robot':raise ValueError('complete trained robot parent required')
    fstate={k[len('forecaster.'):]:v for k,v in parent['model'].items() if k.startswith('forecaster.')}
    rstate={k[len('retargeter.'):]:v for k,v in parent['model'].items() if k.startswith('retargeter.')}
    dataset=RLRobotWindows(a.source,'train')
    indices=dataset.sample_indices(np.random.default_rng(124),384)
    samples=[dataset[int(i)] for i in indices]
    results=[]
    cases=[('legacy128',128,4,128,False,False),('vector128',128,4,128,True,False),
        ('fast256',128,4,256,True,True),('large128',768,8,128,True,True),
        ('large256',a.width,a.layers,256,True,True),('large384',a.width,a.layers,384,True,True)]
    cases=[(n,a.width,a.layers,b,v,f) if n=='large128' else (n,w,l,b,v,f)
           for n,w,l,b,v,f in cases]
    if a.cases:
        requested=set(a.cases.split(','))
        if requested-set(case[0] for case in cases):raise ValueError('unknown benchmark case')
        cases=[case for case in cases if case[0] in requested]
    for name,width,layers,batch_size,vector,fused in cases:
        config=dict(parent['config'],interaction_width=width,interaction_layers=layers,batch_size=batch_size,
            vectorized_padding=vector,checkpoint_interaction=a.checkpoint_interaction and width>128)
        torch.manual_seed(124)
        model=RobotPair(SharedWAM(parent['stats'],config))
        if width==128:model.forecaster.load_state_dict(fstate,strict=True)
        else:expand_forecaster(model.forecaster,fstate,parent['config'])
        model.retargeter.load_state_dict(rstate,strict=True)
        if not vector:
            model.forecaster.scene._sample_tokens=SceneHistoryEncoder._sample_tokens.__get__(model.forecaster.scene)
        model=model.cuda().train()
        optimizer=torch.optim.AdamW(model.parameter_groups(config),weight_decay=config['weight_decay'],fused=fused)
        subset=samples[:batch_size//2]+samples[192:192+batch_size-batch_size//2]
        batch=to_device(shared_collate(subset),torch.device('cuda'))
        torch.backends.cuda.matmul.allow_tf32=fused
        torch.backends.cudnn.allow_tf32=fused
        torch.manual_seed(124)
        row=dict(case=name,width=width,layers=layers,batch_per_gpu=batch_size,
            vectorized_tokens=vector,fused_optimizer=fused,tf32=fused,
            checkpoint_interaction=config['checkpoint_interaction'],
            parameters=sum(p.numel() for p in model.parameters()))
        prediction=loss=terms=norm=None
        try:
            torch.cuda.reset_peak_memory_stats()
            for step in range(3+a.steps):
                if step==3:
                    torch.cuda.synchronize();started=time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=model(batch)
                    loss,terms=robot_loss(prediction,batch,config,model.retargeter)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite benchmark loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_grad'])
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite benchmark gradients')
                optimizer.step()
            torch.cuda.synchronize()
            seconds=(time.perf_counter()-started)/a.steps
            row.update(status='COMPLETED',seconds_per_update=seconds,samples_per_second=batch_size/seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                reserved_mib=torch.cuda.memory_reserved()/2**20,loss=float(loss.detach()))
            if name in ('legacy128','large256'):
                activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
                with torch.profiler.profile(activities=activities) as profile:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=model(batch);loss,terms=robot_loss(prediction,batch,config,model.retargeter)
                    loss.backward();optimizer.step();torch.cuda.synchronize()
                (out/(name+'-profile.txt')).write_text(profile.key_averages().table(sort_by='self_cpu_time_total',row_limit=25))
                profile.export_chrome_trace(str(out/(name+'-trace.json')))
        except torch.cuda.OutOfMemoryError:
            row.update(status='OOM',peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        results.append(row)
        atomic_json(out/'results.json',dict(status='RUNNING',checkpoint=str(a.checkpoint.resolve()),
            source=str(a.source.resolve()),gpu=a.gpu,measured_steps=a.steps,cases=results))
        print(json.dumps(row),flush=True)
        del prediction,loss,terms,norm,optimizer,model,batch
        gc.collect();torch.cuda.empty_cache()
    atomic_json(out/'results.json',dict(status='COMPLETED',checkpoint=str(a.checkpoint.resolve()),
        source=str(a.source.resolve()),gpu=a.gpu,measured_steps=a.steps,cases=results))


if __name__=='__main__':main()
