"""Detached owner-authorized FM trial followed by long three-GPU training."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import torch

TASK=Path(__file__).resolve().parents[2];sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v6.flow_matching import FLOW_ENCODING,OVERNIGHT_CAMPAIGN


def read(path):return json.loads(path.read_text())


def write(path,value):
    temporary=path.with_suffix('.json.part')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def gpu_state():
    result=subprocess.check_output(['nvidia-smi','-i','4,5,6',
        '--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    output={}
    for line in result.splitlines():
        i,m,u=[x.strip() for x in line.split(',')]
        output[int(i)]=dict(memory_mib=int(m) if m.isdigit() else None,utilization=int(u) if u.isdigit() else None)
    if set(output)!={4,5,6}:raise ValueError('only the exact authorized GPUs4/5/6 may be queried')
    return output


def released():
    deadline=time.monotonic()+60
    while True:
        state=gpu_state()
        if all(g['memory_mib'] is not None and g['memory_mib']<=256 for g in state.values()):return state
        if time.monotonic()>deadline:raise ValueError('authorized GPUs not free; no other process will be stopped')
        time.sleep(5)


def audit_checkpoint(path,world,batch,step,normalizers):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved['step']!=step or saved['world_size']!=world
            or saved['execution_contract']['effective_global_batch']!=world*batch
            or len(saved['random_state'])!=world):raise ValueError('checkpoint world/step/RNG/batch mismatch')
    states=saved['optimizer']['state'];expected=sum(len(g['params']) for g in saved['optimizer']['param_groups'])
    if len(states)!=expected or {int(v['step']) for v in states.values()}!={step}:
        raise ValueError('optimizer states or update counts incomplete')
    def finite(value):
        if isinstance(value,torch.Tensor):
            if value.is_floating_point() and not torch.isfinite(value).all():raise ValueError('nonfinite checkpoint')
        elif isinstance(value,dict):
            for child in value.values():finite(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite(child)
    finite(saved['model']);finite(saved['optimizer'])
    for key,value in normalizers.items():
        if not torch.equal(saved['model'][key],value):raise ValueError('normalization changed')
    result=dict(status='PASS',step=step,world_size=world,global_batch=world*batch,
                optimizer_states=len(states),all_optimizer_steps=step,rng_states=world,
                cumulative_elapsed_s=saved['elapsed_s'],bytes=path.stat().st_size)
    del saved;gc.collect();return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','baseline-config','recipe','output','run-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--git-commit',required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);out=a.output.resolve();root=a.run_root.resolve()
    torch.set_num_threads(2)
    state=dict(status='PREPARING',phase='source-audit',supervisor_pid=os.getpid(),
        work_version='v6.flow-overnight-supervisor.v1',experiment_id='P-20261022-wam-v6-flowmatching',
        run_id=out.name,git_commit=a.git_commit,started_at=time.time(),physical_gpus=[4,5,6],stages=[],
        resource_authorization='2026-10-22 owner via Feishu: trial then longer large-scale DDP on4/5/6 only; no credentials; detached',
        target_updates=30000,max_cumulative_training_seconds=43200)
    def publish():write(out/'process.json',state)
    publish();released()
    if shutil.disk_usage(out).free<40*2**30:raise ValueError('need40GiB free startup headroom')
    config=read(a.baseline_config)
    if (config.get('geometry_object_source')!='gt_dense2048' or config.get('geometry_weight')!=.05
            or config.get('amp') or config.get('tf32')):
        raise ValueError('exact latest strict-FP32 GT-object contact objective required')
    config.update(flow_matching=True,flow_encoding=FLOW_ENCODING,flow_steps=4,flow_noise_std=.1,
        flow_width=256,flow_layers=2,flow_matching_weight=1.,flow_lr=.0001,
        flow_eval_seed=172,flow_checkpoint=True,execution_campaign=OVERNIGHT_CAMPAIGN,
        updates=30000,max_seconds=43200,validation_probe_step=128,
        validation_interval=500,checkpoint_interval=500,log_interval=20)
    stats=read(a.recipe.parent/'stats.json')
    normalizers={f'normalizer.{name}_{key}':torch.tensor(stats[name][key],dtype=torch.float32)
                 for name in ('hand','scene') for key in ('mean','std')}
    state['source_audit']=audit_checkpoint(a.checkpoint,3,96,4371,normalizers)
    state['source_checkpoint']=str(a.checkpoint.resolve());publish()
    # Deliberate environment allowlist: do not inspect/copy authentication vars.
    env={k:os.environ[k] for k in ('PATH','HOME','LD_LIBRARY_PATH','CONDA_PREFIX') if k in os.environ}
    code_root=TASK.parents[2]
    env.update(CUDA_VISIBLE_DEVICES='4,5,6',REF2DEX_RUN_ROOT=str(root),REF2DEX_V6_V5_CODE_ROOT=str(code_root),
        REF2DEX_V5_BASE_CODE_ROOT=str(code_root),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1',NCCL_P2P_DISABLE='1',NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        NCCL_ASYNC_ERROR_HANDLING='1',TMPDIR=str(out),TRITON_CACHE_DIR=str(out/'triton'),
        TORCH_EXTENSIONS_DIR=str(out/'extensions'),CUDA_CACHE_PATH=str(out/'cuda-cache'),XDG_CACHE_HOME=str(out/'cache'))
    requested=[];child=None
    signal.signal(signal.SIGTERM,lambda *_:requested.append('SIGTERM'))
    signal.signal(signal.SIGINT,lambda *_:requested.append('SIGINT'))
    def run(name,command,environment,limit):
        nonlocal child
        stage_dir=out/name;record=dict(name=name,command=command,started_at=time.time())
        state['stages'].append(record);state.update(status='RUNNING',phase=name);state.pop('progress',None);publish()
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            record['pid']=child.pid;publish();start=time.monotonic();last=start;previous=-1;deadline=None
            while child.poll() is None:
                progress=stage_dir/'progress.json'
                if progress.exists():
                    row=read(progress);state['progress']=row
                    if row['step']!=previous:previous=row['step'];last=time.monotonic()
                    rate=row.get('updates_per_s')
                    if rate and rate>0:
                        remaining=min((config['updates']-row['step'])/rate,config['max_seconds']-row['elapsed_s'])
                        state.update(estimated_end_unix=time.time()+max(remaining,0),
                                     expected_final_step=min(config['updates'],int(row['step']+rate*max(config['max_seconds']-row['elapsed_s'],0))))
                telemetry=gpu_state()
                reason=requested or ((out/'STOP').exists() and 'owned STOP')
                if shutil.disk_usage(out).free<20*2**30:reason='disk floor'
                if any(g['memory_mib'] is None or g['utilization'] is None for g in telemetry.values()):reason='authorized GPU telemetry unavailable'
                if time.monotonic()-start>limit:reason='stage budget'
                if time.monotonic()-last>1200:reason='no progress20min'
                if time.time()-state['started_at']>47000:reason='absolute supervisor bound'
                if reason and deadline is None:
                    stage_dir.mkdir(exist_ok=True);(stage_dir/'STOP').touch();deadline=time.monotonic()+240
                    state.update(status='STOPPING',stop_reason=reason)
                if deadline and time.monotonic()>deadline:raise TimeoutError('owned child ignored saved STOP')
                state.update(gpu=telemetry,free_gib=shutil.disk_usage(out).free/2**30,elapsed_s=time.time()-state['started_at']);publish()
                with (out/'gpu.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),phase=name,gpus=telemetry))+'\n')
                time.sleep(10)
        record['exit_code']=child.returncode
        result=read(stage_dir/'result.json') if (stage_dir/'result.json').exists() else {}
        record['result']=result;publish()
        if child.returncode:raise RuntimeError(name+' failed; inspect stage log')
        return result,record
    def training_command(name,path,world,checkpoint=None,extra=()):
        prefix=[sys.executable,'-u'] if world==1 else [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3']
        command=prefix+[str(TASK/'tools/run/train_pretrain.py'),'--config',str(path),'--manifest',str(a.recipe.resolve()),
            '--stats',str(a.recipe.parent/'stats.json'),'--output',str(out/name),'--workers','8' if world==3 else '4','--git-commit',a.git_commit]
        command+=['--resume',str(checkpoint)] if checkpoint else ['--initialize-flow-from',str(a.checkpoint.resolve())]
        return command+list(extra)
    def flow_audit(name,path,checkpoint,initialize=False):
        command=[sys.executable,'-u',str(TASK/'tools/audit/check_flow_matching.py'),'--config',str(path),
            '--manifest',str(a.recipe.resolve()),'--stats',str(a.recipe.parent/'stats.json'),
            '--checkpoint',str(checkpoint),'--output',str(out/name)]
        if initialize:command.append('--initialize')
        run(name,command,dict(env,CUDA_VISIBLE_DEVICES='4'),600)
        return read(out/name/'report.json')
    try:
        candidate=out/'configuration-candidate.json';write(candidate,config)
        state['flow_preflight']=flow_audit('flow-preflight',candidate,a.checkpoint,True);publish()
        selected=None
        for batch in (96,80,64):
            released();cfg=dict(config,batch_size=batch);path=out/f'config-b{batch}.json';write(path,cfg)
            name=f'capacity-b{batch}'
            try:result,record=run(name,training_command(name,path,1,extra=['--capacity-probe-updates','8']),dict(env,CUDA_VISIBLE_DEVICES='4'),900)
            except RuntimeError:
                if 'out of memory' in (out/(name+'.log')).read_text().lower():continue
                raise
            if result.get('status')=='COMPLETED' and result['peak_reserved_gib']<=21.5:
                selected=batch,path,cfg;record['selected']=True;break
        if selected is None:raise RuntimeError('no safe FM batch passed capacity')
        batch,path,config=selected;state.update(local_batch=batch,global_batch=batch*3);publish();checkpoint=None
        for name,extra in [('readiness',['--steps','8']),('strict-recovery',['--recovery-probe-updates','2']),('flow-trial',['--steps','128'])]:
            released();result,record=run(name,training_command(name,path,3,checkpoint,extra),env,2400)
            if result['status']!='COMPLETED':state.update(status=result['status']);publish();return
            step=result['last_step'];checkpoint=out/name/f'step{step:06d}.pt'
            record['checkpoint_audit']=audit_checkpoint(checkpoint,3,batch,step,normalizers);publish()
        released();state['flow_trial_audit']=flow_audit('flow-trial-audit',path,checkpoint)
        ratio=state['flow_trial_audit']['teacher_to_zero_ratio']
        state['decision']=dict(status='PROCEED',reason='Finite actual training/recovery and deterministic finite ODE samples; exploratory large-scale run requested by owner.',
            teacher_to_zero_ratio=ratio,teacher_field_improved_vs_zero=ratio<1,
            limitation='Short fixed-TRAIN audit does not establish DEV superiority, multimodal coverage, or rollout success.')
        publish();released()
        result,record=run('training',training_command('training',path,3,checkpoint),env,43800)
        step=result['last_step'];checkpoint=out/'training'/f'step{step:06d}.pt'
        record['checkpoint_audit']=audit_checkpoint(checkpoint,3,batch,step,normalizers)
        state.update(status=result['status'],final_checkpoint=str(checkpoint));publish()
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));publish();raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
