"""Parallel same-checkpoint zero/learned residual evaluation, preserving shard/reset IDs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('panels','checkpoint','rl-checkpoint','cache','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--panel',choices=('dev','core_dev','train'),default='dev');p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--bc-kind',choices=('mlp','wam'),required=True);p.add_argument('--seconds',type=int,default=7200)
    p.add_argument('--seed',type=int,default=249);a=p.parse_args();out=a.output.resolve();gpus=[int(x) for x in a.gpus.split(',')]
    if out.exists() or ROOT/'outputs/wam-rl-policy' not in out.parents:raise ValueError('Fresh Task output')
    if not 1<=len(gpus)<=4 or len(gpus)!=len(set(gpus)):raise ValueError('1..4 unique GPUs')
    for gpu in gpus:
        if subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied: '+str(gpu))
    if not os.environ.get('REF2DEX_REMOTE'):
        available=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        idle=[int(line.split(',')[0]) for line in available.splitlines() if int(line.split(',')[1])<100]
        if len(set(idle)-set(gpus))<2:raise RuntimeError('Keep two local GPUs idle')
    out.mkdir(parents=True);panel=json.loads(a.panels.read_text())[a.panel];children=[];logs=[];stop=[False];started=time.monotonic()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    process=dict(status='RUNNING',pid=os.getpid(),gpus=gpus,paired_control='same RL checkpoint loaded, zero vs learned residual',max_seconds=a.seconds,parts=[])
    def save():
        process['elapsed_s']=time.monotonic()-started;part=out/'process.json.part';part.write_text(json.dumps(process,indent=2)+'\n');part.replace(out/'process.json')
    try:
        for rank,gpu in enumerate(gpus):
            indices=list(range(rank,len(panel),len(gpus)))
            if not indices:continue
            folder=out/('part%02d'%rank);folder.mkdir();jobs=[]
            for role in ('bc','candidate'):
                command=[sys.executable,'-u',str(TASK/'tools/run/evaluate_scale.py'),'--panels',str(a.panels.resolve()),
                    '--checkpoint',str(a.checkpoint.resolve()),'--rl-checkpoint',str(a.rl_checkpoint.resolve()),'--cache',str(a.cache.resolve()),
                    '--output',str(folder/role),'--panel',a.panel,'--gpu',str(gpu),'--bc-kind',a.bc_kind,'--seed',str(a.seed),
                    '--seconds',str(a.seconds),'--shard-indices',','.join(str(i) for i in indices)]
                if role=='bc':command+=['--force-zero-residual']
                jobs.append(dict(name=role,command=command))
            queue=folder/'queue.json';queue.write_text(json.dumps(dict(output=str(folder/'queue'),max_seconds=a.seconds,jobs=jobs),indent=2)+'\n')
            log=(folder/'worker.log').open('x');logs.append(log)
            child=subprocess.Popen([sys.executable,'-u',str(TASK/'tools/run/run_scale_queue.py'),'--queue',str(queue)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children.append(child)
            process['parts'].append(dict(rank=rank,gpu=gpu,shards=indices,pid=child.pid));save()
        while any(c.poll() is None for c in children):
            if stop[0] or (out/'STOP').exists():raise InterruptedError('Paired evaluation STOP')
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Evaluation part failed')
            if time.monotonic()-started>a.seconds+60:raise TimeoutError('Paired evaluation wall budget')
            time.sleep(10)
        if any(c.returncode!=0 for c in children):raise RuntimeError('Part failed before final poll')
        for role in ('bc','candidate'):
            folder=out/role;folder.mkdir();parts=[json.loads((out/('part%02d'%r)/role/'manifest.json').read_text()) for r in range(len(children))]
            merged=dict(parts[0]);merged.update(shards=panel,source_shard_indices=list(range(len(panel))),results=[],totals={k:0 for k in parts[0]['totals']},gpu=gpus)
            for rank,packet in enumerate(parts):
                if packet['status']!='COMPLETED':raise ValueError('Part incomplete')
                for key in merged['totals']:merged['totals'][key]+=packet['totals'][key]
                merged['results'].extend(packet['results'])
                for result in packet['results']:
                    name='shard%02d'%result['shard'];(folder/name).symlink_to(out/('part%02d'%rank)/role/name,target_is_directory=True)
            merged['results'].sort(key=lambda r:r['shard'])
            if [r['shard'] for r in merged['results']]!=list(range(len(panel))):raise ValueError('Incomplete/duplicated panel')
            (folder/'manifest.json').write_text(json.dumps(merged,indent=2)+'\n')
        audit=subprocess.run([sys.executable,str(TASK/'tools/audit/compare_scale.py'),'--bc',str(out/'bc'),'--candidate',str(out/'candidate'),'--output',str(out/'audit')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out/'audit.log').write_text(audit.stdout);process.update(status='COMPLETED' if audit.returncode==0 else 'FAILED_PAIRING_AUDIT',audit_exit_code=audit.returncode)
    except BaseException as error:process.update(status='FAILED',error=repr(error));raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for log in logs:log.close()
        save()


if __name__=='__main__':main()
