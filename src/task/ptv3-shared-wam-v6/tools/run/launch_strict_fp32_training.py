"""Start bounded strict-FP32 three-GPU geometry training from healthy step4371."""
import argparse
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import shutil
import torch

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6 import paths
from launch_local_training import read,write,gpu_state,released
from launch_local_scaled import audit
from ptv3_shared_wam_v6.geometry_consistency import GT_OBJECT_ENCODING


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','baseline-config','recipe','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--git-commit',required=True)
    p.add_argument('--gt-object-contact',action='store_true');a=p.parse_args()
    a.output.mkdir(exist_ok=False);out=a.output.resolve();source=a.checkpoint.resolve()
    state=dict(status='PREPARING',supervisor_pid=os.getpid(),git_commit=a.git_commit,
        work_version='v6.strict-fp32-supervisor.v1',experiment_id='P-20261021-wam-v6-strict-fp32',
        physical_gpus=[4,5,6],run_id=out.name,started_at=time.time(),stages=[])
    if a.gt_object_contact:
        state.update(work_version='v6.gt-object-contact-supervisor.v1',
                     experiment_id='P-20261021-wam-v6-gt-object-contact')
    def publish():write(out/'process.json',state)
    publish();torch.set_num_threads(2)
    released([4,5,6])
    if shutil.disk_usage(out).free<40*2**30:raise ValueError('insufficient transition/checkpoint peak headroom')
    packet=torch.load(source,map_location='cpu',weights_only=False)
    if packet['step']!=4371 or packet['config'].get('geometry_consistency',False):
        raise ValueError('healthy no-geometry step4371 initialization required')
    normalizer={k:v.clone() for k,v in packet['model'].items() if k.startswith('normalizer.')}
    state['source_audit']=audit(source,3,96,4371,normalizer)
    state['source_checkpoint']=str(source)
    state['resource_authorization']='User requests healthy-checkpoint FP32 training, retaining previously authorized GPUs4/5/6.'
    config=dict(read(a.baseline_config),amp=False,tf32=False,strict_fp32=True,
                fp32_attention_backend='efficient',monitor_activation_rms=True)
    if (config['geometry_weight']!=0.022920552754813268 or not config['geometry_consistency']
            or config['updates']!=10000 or config['max_seconds']!=14400):
        raise ValueError('matched geometry objective and original bounds required')
    if a.gt_object_contact:
        config.update(geometry_object_source='gt_dense2048',geometry_encoding=GT_OBJECT_ENCODING,
                      geometry_queries=2048,geometry_contact_radius_m=.003,
                      geometry_contact_falloff_m=.010,geometry_contact_smoothing_m=.001)
    state['matched_bf16_config']=str(a.baseline_config.resolve())
    state['changed_config_keys']=['amp','tf32','strict_fp32','fp32_attention_backend','monitor_activation_rms']
    del packet;gc.collect();publish()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_ASYNC_ERROR_HANDLING='1',TMPDIR=str(out),
        TRITON_CACHE_DIR=str(out/'triton'),TORCH_EXTENSIONS_DIR=str(out/'extensions'))
    requested=[];child=None
    signal.signal(signal.SIGTERM,lambda *_:requested.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:requested.append('SIGINT'))
    def run(name,command,environment,limit):
        nonlocal child
        current=out/name;record=dict(name=name,command=command,started_at=time.time())
        state['stages'].append(record);state.update(status='RUNNING',phase=name);state.pop('progress',None);publish()
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['pid']=child.pid;publish();start=time.monotonic();deadline=None;last=start;previous=-1
            while child.poll() is None:
                telemetry=gpu_state();progress=current/'progress.json'
                if progress.exists():
                    row=read(progress);state['progress']=row
                    if row['step']!=previous:previous=row['step'];last=time.monotonic()
                reason=requested or ((out/'STOP').exists() and 'owned STOP')
                if shutil.disk_usage(out).free<20*2**30:reason='disk floor'
                if time.monotonic()-start>limit:reason='stage time budget'
                if time.monotonic()-last>1200:reason='no progress for20min'
                if reason and deadline is None:
                    current.mkdir(exist_ok=True);(current/'STOP').touch();deadline=time.monotonic()+180
                    state.update(status='STOPPING',stop_reason=reason)
                if deadline and time.monotonic()>deadline:raise TimeoutError('owned child ignored STOP')
                state.update(gpu=telemetry,free_gib=shutil.disk_usage(out).free/2**30,elapsed_s=time.time()-state['started_at']);publish()
                with (out/'gpu.jsonl').open('a') as gpu:gpu.write(__import__('json').dumps(dict(time=time.time(),phase=name,gpus=telemetry))+'\n')
                time.sleep(10)
        record['exit_code']=child.returncode
        result=read(current/'result.json') if (current/'result.json').exists() else {}
        record['result']=result;publish();return result,child.returncode,record
    def command(name,config_path,world,checkpoint=None,extra=()):
        prefix=[sys.executable,'-u'] if world==1 else [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3']
        args=prefix+[str(Path(__file__).with_name('train_pretrain.py')),'--config',str(config_path),
            '--manifest',str(a.recipe.resolve()),'--stats',str(a.recipe.parent/'stats.json'),
            '--output',str(out/name),'--workers','8' if world==3 else '4','--git-commit',a.git_commit]
        args+=['--resume',str(checkpoint)] if checkpoint else ['--initialize-geometry-from',str(source)]
        return args+list(extra)
    try:
        if a.gt_object_contact:
            preliminary=out/'configuration-candidate.json';write(preliminary,config)
            preflight=[sys.executable,'-u',str(TASK/'tools/audit/check_geometry_gradients.py'),
                '--config',str(preliminary),'--manifest',str(a.recipe.resolve()),
                '--stats',str(a.recipe.parent/'stats.json'),'--checkpoint',str(source),
                '--output',str(out/'gpu-preflight')]
            _,code,_=run('gpu-preflight',preflight,dict(env,CUDA_VISIBLE_DEVICES='4'),600)
            if code:raise RuntimeError('GT-object contact gradient/coverage preflight failed')
            gradient=read(out/'gpu-preflight/report.json');state['gradient_preflight']=gradient
            config['geometry_weight']=gradient['recommended_geometry_weight'];publish()
        selected=None
        # Test matched batch96 first; only lower when real memory requires it.
        for batch in (96,80,64):
            released([4,5,6]);candidate=dict(config,batch_size=batch)
            path=out/('config-b%d.json'%batch);write(path,candidate);name='capacity-b%d'%batch
            result,code,record=run(name,command(name,path,1,extra=['--capacity-probe-updates','8']),dict(env,CUDA_VISIBLE_DEVICES='4'),600)
            if code:
                if 'out of memory' in (out/(name+'.log')).read_text().lower():continue
                raise RuntimeError('geometry capacity test failed')
            if result.get('status')!='COMPLETED':raise RuntimeError('capacity test did not complete')
            if result['peak_reserved_gib']<=21.5:
                selected=(batch,path,candidate);record['selected']=True;break
        if selected is None:raise RuntimeError('no bounded geometry batch passed capacity')
        batch,config_path,config=selected;state.update(local_batch=batch,global_batch=batch*3,
            geometry_weight=config['geometry_weight'],target_updates=10000,max_seconds=14400);publish()
        released([4,5,6]);checkpoint=None
        for name,extra in [('readiness',['--steps','8']),('strict-recovery',['--recovery-probe-updates','2']),('training',[])]:
            result,code,record=run(name,command(name,config_path,3,checkpoint,extra),env,1800 if name!='training' else 14700)
            if code:raise RuntimeError('strict FP32 '+name+' failed; inspect log')
            if result['status']!='COMPLETED':state.update(status=result['status']);publish();return
            step=result['last_step'];checkpoint=out/name/('step%06d.pt'%step)
            record['checkpoint_audit']=audit(checkpoint,3,batch,step,normalizer);publish()
        state.update(status='COMPLETED');publish()
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));publish();raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
