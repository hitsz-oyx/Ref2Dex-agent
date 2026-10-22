#!/usr/bin/env python3
"""One-shot brief37 supervisor: admitted pool only, watchdog and wall-clock hard stop."""
import argparse, datetime, hashlib, importlib.util, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
PYTHON=Path('/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix(p.suffix+'.part');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def size(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def terminal_reason(code, result, watchdog_reason=None):
    if result and result.get('status', '').startswith('FAILED'):
        return result['status']+'_NO_RESTART'
    if code:
        return 'FAILED_NO_RESTART'
    return watchdog_reason or ('COMPLETED' if result and result.get('status')=='COMPLETED' else 'STOPPED')

def snapshot():
    g=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    uuids={r.split(',')[0].strip() for r in p.splitlines() if r.strip()};rows=[]
    for line in g.splitlines():
        i,u,used,total,free,util=[x.strip() for x in line.split(',')]
        rows.append(dict(index=int(i),uuid=u,used_bytes=int(used)*2**20,total_bytes=int(total)*2**20,free_bytes=int(free)*2**20,utilization_percent=int(util),has_process=u in uuids))
    return dict(at=stamp(),gpus=rows,compute_processes_csv=p,filesystem_free_bytes=shutil.disk_usage('/home2').free)

def preflight(checks,policy,config):
    r=snapshot();available=[g['index'] for g in r['gpus'] if g['index'] in (4,5,6) and not g['has_process'] and g['used_bytes']<256*2**20 and g['utilization_percent']==0]
    manifest=Path(policy['curated_manifest']);m=read(manifest);rows=[[int(x['source_id']),[int(x['sequence_id']),int(x['slot']),int(x['tick'])],int(x['tick'])] for x in m['selection']['train_window_rows']]
    identity=hashlib.sha256(json.dumps(rows,separators=(',',':'),sort_keys=True).encode()).hexdigest()
    expected=16*2**30
    gates=dict(three_checks=all(checks['checks'].get(str(i))=='PASS' for i in (1,2,3)),
        authorized_gpus_free=set(available)=={4,5,6},disk_headroom=r['filesystem_free_bytes']-expected>=20*2**30,
        data_manifest=sha(manifest)==policy['data_identity']['manifest_sha256'],ordered_6000=len(rows)==6000 and identity==policy['data_identity']['ordered_window_tuple_sha256'],
        recipe=sha(config['condition_recipe'])==config['condition_recipe_sha256'],deadline=time.time()<config['hard_stop_epoch']-90)
    r.update(gates=gates,pass_all=all(gates.values()),available_authorized=available,estimated_output_bytes=expected,
        expected_checkpoint_bytes=Path(policy['initialization']['checkpoint']).stat().st_size,
        checkpoint_growth_estimate_bytes_per_update=Path(policy['initialization']['checkpoint']).stat().st_size/1000.,
        gpu_authorization=policy['resource_policy'],spare_exemption='owner brief34/37 fixed pool4/5/6/7;7 executor;0-3 zyc; require surplus spares only within authorized pool; world3 uses4/5/6',
        disk_stop_bytes=20*2**30,disk_warn_bytes=30*2**30,artifact_alarm_bytes=32*2**30)
    return r

def main():
    a=argparse.ArgumentParser();a.add_argument('--checks',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--queue',action='store_true');args=a.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=os.open(str(out/'ONE_SHOT.lock'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o664);os.write(lock,str(os.getpid()).encode());os.close(lock)
    config_path=TASK/'configs/tau_no_goal_no_event_train_v1.json';config=read(config_path);policy=read(config['condition_recipe']);checks=read(args.checks)
    launch=None;process=None;reason=None
    while True:
        r=preflight(checks,policy,config);write(out/'preflight.json',r)
        with (out/'preflight-history.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        if r['pass_all']:break
        bad=[k for k,v in r['gates'].items() if not v]
        write(out/'queue-status.json',dict(status='WAITING' if args.queue else 'BLOCKED',blocked=bad,at=stamp(),supervisor_pid=os.getpid(),training_updates=0))
        if not args.queue or any(k!='authorized_gpus_free' for k in bad):return
        print(json.dumps(dict(status='WAITING_GPU_POOL',at=stamp(),available=r['available_authorized'])),flush=True);time.sleep(30)
    training=out/'training'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    cmd=[str(PYTHON),'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3',str(TASK/'tools/run/train_pretrain.py'),
         '--config',str(config_path),'--manifest',str(ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json'),
         '--stats',str(ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json'),'--output',str(training),
         '--initialize-condition-from',policy['initialization']['checkpoint'],'--workers','4','--git-commit',head,'--preserve-all-checkpoints','--checkpoint-budget-gib','32']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(out/'tmp'),REF2DEX_RUN_ROOT=str(out))
    (out/'tmp').mkdir(exist_ok=True)
    with (out/'train-launch.log').open('x') as log:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch=dict(status='STARTED',at=stamp(),pid=process.pid,pgid=process.pid,supervisor_pid=os.getpid(),physical_gpus=[4,5,6],head=head,
            policy_sha256=sha(config['condition_recipe']),config_sha256=sha(config_path),data_identity=policy['data_identity'],command=cmd,
            initialization=policy['initialization'],estimated_duration_s=min(4800,config['hard_stop_epoch']-time.time()),estimated_bytes=r['estimated_output_bytes'],hard_stop_epoch=config['hard_stop_epoch'])
        write(out/'launch.json',launch)
        while process.poll() is None:
            free=shutil.disk_usage('/home2').free;bytes_used=size(out);now=time.time()
            alarm=bytes_used>32*2**30
            record=dict(at=stamp(),pid=process.pid,free_bytes=free,artifact_bytes=bytes_used,disk_warning=free<30*2**30,artifact_alarm=alarm,
                stop_reason=reason,seconds_to_hard_stop=config['hard_stop_epoch']-now)
            with (out/'watchdog.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            write(out/'watchdog-latest.json',record)
            if free<20*2**30:reason='DISK_STOP_20_GIB'
            elif now>=config['hard_stop_epoch']-30:reason='HARD_STOP_SAVE_MARGIN_30_SECONDS'
            if reason:
                training.mkdir(exist_ok=True);(training/'STOP').touch(exist_ok=True)
            if now>=config['hard_stop_epoch']:
                reason='HARD_STOP_0715_TERMINATED_OWN_GROUP'
                # Only the newly spawned session; never GPUs0-3 or arbitrary PIDs.
                if process.pid in (2730517,2730518,2730519,2730520) or os.getpgid(process.pid)!=process.pid:raise RuntimeError('unsafe process group')
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(min(10,max(.1,config['hard_stop_epoch']-now)))
        code=process.wait()
    result=read(training/'result.json') if (training/'result.json').exists() else None
    progress=read(training/'progress.json') if (training/'progress.json').exists() else None
    write(out/'stop-receipt.json',dict(at=stamp(),pid=process.pid,returncode=code,reason=terminal_reason(code,result,reason),
        result=result,latest_progress=progress,artifact_bytes=size(out),automatic_restart=False))

if __name__=='__main__':main()
