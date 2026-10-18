"""Resume an initialized cache on disjoint GPU shards; only coordinator publishes completion."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np


def merge_completion(base,shards):
    result=np.asarray(base,dtype=bool).copy()
    for rank,values in enumerate(shards):
        if values.shape!=result.shape:raise ValueError('Shard completion shape mismatch')
        result[rank::len(shards)] |= values[rank::len(shards)]
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seconds',type=int,default=21600);p.add_argument('--fit-output',type=Path)
    p.add_argument('--init-current',type=Path);p.add_argument('--allow-shared-gpu',action='store_true')
    a=p.parse_args();out=a.cache.resolve()
    if len(set(a.gpus))!=len(a.gpus) or len(a.gpus)<2:raise ValueError('Distinct GPUs required')
    lock=(out/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=json.loads((out/'process.json').read_text())
    cmdline=Path('/proc')/str(prior['pid'])/'cmdline'
    if cmdline.exists() and 'prepare-scheduled' in cmdline.read_text():raise RuntimeError('Single-GPU writer still active')
    identity=json.loads((out/'cache-identity.json').read_text());base=np.load(out/'done.npy').copy()
    oldmeta=out/'parallel.json'
    if oldmeta.exists():
        previous=json.loads(oldmeta.read_text());n=len(previous['gpus'])
        base=merge_completion(base,[np.load(out/('done-shard-%d.npy'%i)) for i in range(n)])
    meta=dict(status='RUNNING',pid=os.getpid(),gpus=a.gpus,workers_per_gpu=a.workers,
        records_total=len(base),records_initial=int(base.sum()),git_commit=os.environ.get('REF2DEX_CODE_COMMIT'),
        partition='global record index modulo GPU count; disjoint row intervals',seconds=a.seconds)
    def write():
        part=out/'parallel.json.part';part.write_text(json.dumps(meta,indent=2)+'\n');part.replace(oldmeta)
    for rank in range(len(a.gpus)):np.save(out/('done-shard-%d.npy'%rank),base)
    if (out/'STOP').exists():(out/'STOP').unlink()
    write();processes=[];streams=[];started=time.monotonic()
    def stop(*_):raise InterruptedError('Coordinator stop requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    entry=Path(__file__).with_name('train_current_bc.py')
    try:
        for rank,gpu in enumerate(a.gpus):
            args=[sys.executable,str(entry),'prepare-scheduled','--data',identity['base'],'--sequence-cache',identity['source'],
                '--wam-checkpoint',identity['wam'],'--wam-dense-manifest',identity['dense'],'--wam-batch',str(identity['batch']),
                '--output',str(out),'--resume-cache','--gpu',str(gpu),'--workers',str(a.workers),
                '--shard-index',str(rank),'--shard-count',str(len(a.gpus)),'--seconds',str(a.seconds)]
            if a.allow_shared_gpu:args.append('--allow-shared-gpu')
            stream=(out/('shard-%d.log'%rank)).open('a');streams.append(stream)
            processes.append(subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT))
        while True:
            codes=[p.poll() for p in processes]
            if any(c not in (None,0) for c in codes):raise RuntimeError('Cache shard failed: '+str(codes))
            merged=merge_completion(base,[np.load(out/('done-shard-%d.npy'%i),mmap_mode='r') for i in range(len(a.gpus))])
            elapsed=time.monotonic()-started;finished=int(merged.sum());rate=(finished-int(base.sum()))/max(elapsed,1)
            meta.update(records_done=finished,elapsed_s=elapsed,records_per_second=rate,
                eta_s=(len(base)-finished)/rate if rate else None,child_pids=[p.pid for p in processes]);write()
            if all(c==0 for c in codes):break
            if elapsed>a.seconds:raise TimeoutError('Parallel cache deadline')
            if (out/'STOP').exists():raise InterruptedError('Cache STOP requested')
            time.sleep(15)
        if not merged.all():raise RuntimeError('Successful shards did not cover all records')
        np.save(out/'done.npy',merged)
        manifest=json.loads((out/'manifest.json').read_text());manifest.update(status='COMPLETED',records_done=len(base),parallel=meta)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        meta['status']='CACHE_COMPLETED';write()
        if a.fit_output:
            if not a.init_current:raise ValueError('Parent BC required for queued fit')
            args=[sys.executable,str(entry),'train','--data',str(out),'--output',str(a.fit_output),
                '--init-current',str(a.init_current),'--gpu',str(a.gpus[-1]),'--seed','0','--steps','50000','--batch','2048','--seconds','7200']
            if a.allow_shared_gpu:args.append('--allow-shared-gpu')
            meta['status']='BC_RUNNING';write();subprocess.run(args,check=True,timeout=7320)
        meta['status']='COMPLETED';write()
    except BaseException as error:
        (out/'STOP').touch();meta.update(status='STOPPED' if isinstance(error,InterruptedError) else 'FAILED',error=repr(error));write()
        for proc in processes:
            if proc.poll() is None:
                try:proc.wait(timeout=45)
                except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=15)
        if processes:
            merged=merge_completion(base,[np.load(out/('done-shard-%d.npy'%i)) for i in range(len(a.gpus))])
            np.save(out/'done.npy',merged)
        raise
    finally:
        for stream in streams:stream.close()

if __name__=='__main__':main()
