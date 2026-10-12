#!/usr/bin/env python3
"""Bounded resumable full release acquisition, parallel to persistent GPU3 reconstruction.

Download every file member once via ZIP ranges; no duplicate full archives.
TRAIN, TEST and EXTRA keep separate release identities. Ready TRAIN packets
are individually CRC/SHA verified and consumed without changing any active fit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
import zlib

import requests
from urllib3.util.retry import Retry

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0,str(REPO/'src/task/cm-pointflow-effect-pretrain/tools/run/ref5_data_expansion'))
sys.path.insert(0,str(REPO/'src/task/cm-pointflow-effect-pretrain/tools/run'))
from download_egodex import RemoteZip
from launch_pointworld_action_ddp import belongs_to, gpu_processes, record


def direct_session():
    session=requests.Session();session.trust_env=False
    session.mount('https://',requests.adapters.HTTPAdapter(max_retries=Retry(
        total=4,backoff_factor=1,status_forcelist=[429,500,502,503,504],allowed_methods=['GET','HEAD'])))
    return session


class IndexedRemoteZip(RemoteZip):
    """Serve immutable central-directory reads locally for independent workers."""
    def __init__(self,url,session,cap,deadline,start,tail):
        super().__init__(url,session,cap,deadline)
        self.directory_start,self.directory_tail=start,tail

    def read(self,length=-1):
        length=self.size-self.pos if length<0 else min(length,self.size-self.pos)
        if self.pos>=self.directory_start:
            lo=self.pos-self.directory_start
            result=self.directory_tail[lo:lo+length];self.pos+=len(result);return result
        return super().read(length)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--template-plan',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=3)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seconds',type=int,default=604800)
    p.add_argument('--raw-cap-bytes',type=int,default=2_500_000_000_000)
    p.add_argument('--processed-cap-bytes',type=int,default=8_000_000_000_000)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=16 or a.seconds<=0 or gpu_processes([a.gpu]):
        raise ValueError('bounded acquisition and idle assigned reconstruction GPU required')
    if a.output.exists() and not a.resume: raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'inventories').mkdir(exist_ok=True)
    template=json.loads(a.template_plan.read_text())
    started=time.time();deadline=time.monotonic()+a.seconds
    state=dict(status='RUNNING',pid=os.getpid(),started_at=started,run_id=a.output.name,
               git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               gpu=a.gpu,workers=a.workers,raw_cap_bytes=a.raw_cap_bytes,
               processed_cap_bytes=a.processed_cap_bytes,raw_verified_bytes=0,reserved_bytes=0,
               completed_packets=0,failed_packets=0,network_bytes=0,archives=[],
               archive_order=['part2','part1','part3','part4','part5','test','extra'],
               download_policy='All non-directory members via verified ranges; no full ZIP duplicate',
               reconstruction_policy='Persistent TRAIN-only teachers; TEST/EXTRA isolated; long contexts remain pending')
    status_path=a.output/'acquisition_status.json'
    if a.resume:
        old=json.loads(status_path.read_text())
        if (old['raw_cap_bytes'],old['processed_cap_bytes'])!=(a.raw_cap_bytes,a.processed_cap_bytes):
            raise ValueError('resume storage envelope differs')
        state['raw_verified_bytes']=old['raw_verified_bytes']
        state['completed_packets']=old['completed_packets']
        state['network_bytes']=old['network_bytes']
    lock=threading.Lock();stop=threading.Event();child=None
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda *_:stop.set())
    cache=REPO/'tmp/egodex-full';cache.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(cache),PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TRITON_CACHE_DIR=str(cache/'triton'))
    cmd=[template['prepare_python'],str(Path(__file__).with_name('reconstruct_egodex_queue.py')),
         '--output',str(a.output),'--template-plan',str(a.template_plan),'--seconds',str(a.seconds),
         '--processed-cap-bytes',str(a.processed_cap_bytes)]
    log=(a.output/'reconstruction_worker.log').open('a')
    child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    state.update(reconstruction_pid=child.pid,reconstruction_command=cmd)

    def persist():
        with lock:
            state.update(checked_at=time.time(),elapsed_seconds=time.time()-started,
                         average_network_MBps=state['network_bytes']/max(time.time()-started,1)/1e6)
            record(status_path,state)

    def safety():
        if stop.is_set() or time.monotonic()>=deadline: raise RuntimeError('user stop/deadline')
        if shutil.disk_usage(a.output).free<10_000_000_000_000: raise RuntimeError('NAS free space below10TB')

    def monitor():
        while not stop.wait(30):
            try:
                safety()
                if any(not belongs_to(pid,child.pid) for pid in gpu_processes([a.gpu])):
                    raise RuntimeError('foreign GPU3 process')
                if child.poll() is not None and child.returncode!=0:
                    raise RuntimeError('reconstruction worker failed; see its log')
                persist()
            except Exception as exc:
                with lock:state['stop_reason']=repr(exc)
                stop.set()
    threading.Thread(target=monitor,daemon=True).start();persist()
    try:
        for archive in state['archive_order']:
            safety()
            url='https://ml-site.cdn-apple.com/datasets/egodex/'+archive+'.zip'
            split='official_train' if archive.startswith('part') else 'official_'+archive
            direct=direct_session()
            remote=RemoteZip(url,direct,128*2**20,deadline)
            with zipfile.ZipFile(remote) as master:
                infos=master.infolist();start=master.start_dir
                remote.seek(start);tail=remote.read(remote.size-start)
                inventory=dict(url=url,archive_size=remote.size,archive_etag=remote.etag,
                    files=[dict(name=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,crc32=i.CRC)
                           for i in infos])
            record(a.output/'inventories'/(archive+'.json'),inventory)
            files={i.filename:i for i in infos if not i.is_dir()}
            if any(Path(n).is_absolute() or '..' in Path(n).parts or Path(n).parts[0]!=archive for n in files):
                raise ValueError('unsafe or unexpected archive namespace')
            paired=sorted(n for n in files if n.endswith('.hdf5') and n[:-5]+'.mp4' in files)
            # Start unseen episodes before the26 old diagnostic source pairs.
            old={r['member'] for r in template['records']}
            paired.sort(key=lambda n:(n in old,n))
            used=set();packets=[]
            for n in paired:
                names=[n,n[:-5]+'.mp4'];packets.append((n,names,True));used.update(names)
            packets.extend((n,[n],False) for n in sorted(set(files)-used))
            jobs=queue.Queue()
            for packet in packets:jobs.put(packet)
            with lock:
                state['network_bytes']+=remote.transferred
                entry=dict(archive=archive,status='DOWNLOADING',bytes=remote.size,etag=remote.etag,
                           packets=len(packets),paired_episodes=len(paired),completed=0,failed=0)
                state['archives'].append(entry)
            persist()

            def worker(worker_id):
                session=direct_session()
                handle=IndexedRemoteZip(url,session,16*2**30,deadline,start,tail)
                if (handle.size,handle.etag)!=(remote.size,remote.etag): raise ValueError('archive identity drift')
                with zipfile.ZipFile(handle) as source_zip:
                    while not jobs.empty() and not stop.is_set():
                        try:member,names,is_pair=jobs.get_nowait()
                        except queue.Empty:break
                        key=archive+'/'+hashlib.sha256(member.encode()).hexdigest()[:20]
                        packet_root=a.output/'acquisition'/key
                        manifest_path=packet_root/'download_manifest.json'
                        if manifest_path.exists() and json.loads(manifest_path.read_text()).get('status')=='COMPLETED':
                            with lock:
                                entry['completed']+=1
                                # Re-enqueue on resume; completed reconstruction receipts deduplicate.
                                # This also recovers a crash between publication and the first enqueue.
                                if is_pair:
                                    ready=dict(key=key,member=member,raw_hdf5=str(packet_root/'raw'/member),
                                               source_manifest=str(manifest_path),release_split=split)
                                    with (a.output/'ready.jsonl').open('a') as f:f.write(json.dumps(ready)+'\n')
                            jobs.task_done();continue
                        size=sum(files[n].file_size for n in names)
                        with lock:
                            if state['raw_verified_bytes']+state['reserved_bytes']+size+16*2**30>a.raw_cap_bytes:
                                state['stop_reason']='raw storage envelope';stop.set();jobs.task_done();return
                            state['reserved_bytes']+=size
                        packet_root.mkdir(parents=True,exist_ok=True)
                        manifest=dict(status='RUNNING',url=url,split=split,archive_size=remote.size,
                                      archive_etag=remote.etag,files=[],training_allowed=False)
                        transferred_before=handle.transferred
                        try:
                            safety();handle.transferred=0;transferred_before=0
                            for name in names:
                                dest=packet_root/'raw'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                                part=dest.with_suffix(dest.suffix+'.part')
                                if dest.exists():
                                    # A failed packet's completed member is retained and checked once.
                                    digest=hashlib.sha256();crc=0
                                    with dest.open('rb') as f:
                                        for block in iter(lambda:f.read(8*2**20),b''):
                                            digest.update(block);crc=zlib.crc32(block,crc)
                                    if dest.stat().st_size!=files[name].file_size or crc!=files[name].CRC:
                                        raise ValueError('failed packet published member differs from ZIP CRC')
                                else:
                                    digest=hashlib.sha256()
                                    with source_zip.open(name) as f,part.open('wb') as target:
                                        for block in iter(lambda:f.read(8*2**20),b''):
                                            safety();target.write(block);digest.update(block)
                                    if part.stat().st_size!=files[name].file_size:raise ValueError('member size mismatch')
                                    part.replace(dest)
                                manifest['files'].append(dict(member=name,bytes=dest.stat().st_size,
                                                              sha256=digest.hexdigest(),crc32=files[name].CRC))
                            manifest['status']='COMPLETED';record(manifest_path,manifest)
                            with lock:
                                state['completed_packets']+=1;state['raw_verified_bytes']+=size;entry['completed']+=1
                                state['network_bytes']+=handle.transferred-transferred_before
                                if is_pair:
                                    ready=dict(key=key,member=member,raw_hdf5=str(packet_root/'raw'/member),
                                               source_manifest=str(manifest_path),release_split=split)
                                    with (a.output/'ready.jsonl').open('a') as f:f.write(json.dumps(ready)+'\n')
                        except Exception as exc:
                            manifest.update(status='FAILED',error=repr(exc));record(manifest_path,manifest)
                            with lock:state['failed_packets']+=1;entry['failed']+=1
                        finally:
                            with lock:state['reserved_bytes']-=size
                            jobs.task_done()
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(worker,i) for i in range(a.workers)]
                for future in futures:future.result()
            entry['status']='COMPLETED' if entry['completed']==entry['packets'] else 'INCOMPLETE'
            persist()
        state['status']='COMPLETED' if all(x['status']=='COMPLETED' for x in state['archives']) else 'INCOMPLETE'
    except Exception as exc:
        state.update(status='STOPPED' if stop.is_set() else 'FAILED',error=repr(exc))
    finally:
        state['acquisition_ended_at']=time.time();persist()
    # GPU consumer drains verified records independently of network completion.
    while child.poll() is None and not stop.is_set() and time.monotonic()<deadline:
        time.sleep(10)
    if child.poll() is None:
        os.kill(child.pid,signal.SIGUSR1)
        try:child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    stop.set();state['pipeline_ended_at']=time.time();state['reconstruction_exit_code']=child.returncode;persist()


if __name__=='__main__':
    main()
