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
from launch_pointworld_action_ddp import belongs_to, gpu_processes


def record(path,value):
    """Publish an explicitly readable owned receipt on the mapped NAS."""
    part=path.with_suffix('.json.part')
    part.write_text(json.dumps(value,indent=2)+'\n')
    part.chmod(0o644)
    part.replace(path)


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
        self.member_start,self.member_bytes=0,b''

    def prefetch_members(self,infos,max_bytes=16*2**20):
        """One bounded range for adjacent local headers and compressed payloads.

        Reserve the ZIP format's maximum local extra-field length. This works
        without guessing whether local and central extra fields match. Large
        or far-apart members retain the ordinary streaming/CRC path.
        """
        if not infos:return False
        lo=min(info.header_offset for info in infos)
        end=min(self.directory_start,max(info.header_offset+30+len(info.filename.encode('utf-8'))
                                        +65535+info.compress_size for info in infos))
        if end<=lo or end-lo>max_bytes:return False
        position=self.pos
        self.pos=lo
        try:data=super().read(end-lo)
        finally:self.pos=position
        self.member_start,self.member_bytes=lo,data
        return True

    def read(self,length=-1):
        length=self.size-self.pos if length<0 else min(length,self.size-self.pos)
        if self.pos>=self.directory_start:
            lo=self.pos-self.directory_start
            result=self.directory_tail[lo:lo+length];self.pos+=len(result);return result
        lo=self.pos-self.member_start
        if 0<=lo and lo+length<=len(self.member_bytes):
            result=self.member_bytes[lo:lo+length];self.pos+=len(result);return result
        return super().read(length)


def packet_groups(packets,files,max_bytes):
    """Keep packet integrity while batching physically adjacent ZIP members."""
    packets=sorted(packets,key=lambda p:min(files[n].header_offset for n in p[1]))
    group=[];lo=hi=0
    for packet in packets:
        left=min(files[n].header_offset for n in packet[1])
        right=max(files[n].header_offset+30+len(n.encode('utf-8'))+65535+files[n].compress_size
                  for n in packet[1])
        if group and (max(hi,right)-lo>max_bytes or len(group)>=128):
            yield group;group=[]
        if not group:lo=left;hi=right
        else:hi=max(hi,right)
        group.append(packet)
    if group:yield group


class ExistingReconstruction:
    """Observe the verified owned consumer without changing its lifetime."""
    def __init__(self,pid,output):
        self.pid=pid;self.output=output;self.returncode=None
        root=Path('/proc')/str(pid)
        cmd=root.joinpath('cmdline').read_bytes().split(b'\0')
        if (b'reconstruct_egodex_queue.py' not in b' '.join(cmd)
                or str(output).encode() not in cmd or root.stat().st_uid!=os.getuid()):
            raise ValueError('existing reconstruction identity mismatch')
        self.birth=root.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]

    def poll(self):
        try:
            stat=(Path('/proc')/str(self.pid)/'stat').read_text().rsplit(')',1)[1].split()
            if stat[19]==self.birth and stat[0]!='Z':return None
        except FileNotFoundError:pass
        # This process is not our child; only its recorded terminal state is known.
        status=json.loads((self.output/'reconstruction_status.json').read_text())['status']
        self.returncode=0 if status=='DRAINED' else 1
        return self.returncode


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
    p.add_argument('--prefetch-bytes',type=int,default=32*2**20)
    p.add_argument('--reuse-reconstruction-pid',type=int,
                   help='Resume acquisition while leaving this owned consumer unchanged')
    a=p.parse_args()
    existing=ExistingReconstruction(a.reuse_reconstruction_pid,a.output) if a.reuse_reconstruction_pid else None
    foreign=[pid for pid in gpu_processes([a.gpu]) if existing is None or not belongs_to(pid,existing.pid)]
    if (not 1<=a.workers<=16 or a.seconds<=0 or foreign
            or not 0<=a.prefetch_bytes<=128*2**20 or (existing and not a.resume)):
        raise ValueError('bounded acquisition and idle assigned reconstruction GPU required')
    if a.output.exists() and not a.resume: raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'inventories').mkdir(exist_ok=True)
    template=json.loads(a.template_plan.read_text())
    started=time.time();deadline=time.monotonic()+a.seconds
    state=dict(status='RUNNING',pid=os.getpid(),started_at=started,run_id=a.output.name,
               git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
               gpu=a.gpu,workers=a.workers,prefetch_bytes=a.prefetch_bytes,raw_cap_bytes=a.raw_cap_bytes,
               processed_cap_bytes=a.processed_cap_bytes,raw_verified_bytes=0,reserved_bytes=0,
               completed_packets=0,failed_packets=0,network_bytes=0,archives=[],
               archive_order=['part2','part1','part3','part4','part5','test','extra'],
               download_policy='Physical-order bounded coalesced ZIP ranges; CRC/SHA verified; no full ZIP duplicate',
               reconstruction_policy='Persistent TRAIN-only teachers; TEST/EXTRA isolated; long contexts remain pending')
    status_path=a.output/'acquisition_status.json'
    if a.resume:
        old=json.loads(status_path.read_text())
        previous_proc=Path('/proc')/str(old['pid'])/'cmdline'
        if previous_proc.exists():
            previous_command=previous_proc.read_bytes().split(b'\0')
            if (b'launch_egodex_full_pipeline.py' in b' '.join(previous_command)
                    and str(a.output).encode() in previous_command):
                raise RuntimeError('previous acquisition process is still active')
        if (old['raw_cap_bytes'],old['processed_cap_bytes'])!=(a.raw_cap_bytes,a.processed_cap_bytes):
            raise ValueError('resume storage envelope differs')
        state['raw_verified_bytes']=old['raw_verified_bytes']
        state['completed_packets']=old['completed_packets']
        state['network_bytes']=old['network_bytes']
        started=old['started_at'];state['started_at']=started
        deadline=time.monotonic()+max(0,started+a.seconds-time.time())
        state['previous_acquisition_pid']=old['pid']
    lock=threading.Lock();stop=threading.Event();child=None
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda *_:stop.set())
    cache=REPO/'tmp/egodex-full';cache.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(cache),PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TRITON_CACHE_DIR=str(cache/'triton'))
    cmd=[template['prepare_python'],str(Path(__file__).with_name('reconstruct_egodex_queue.py')),
         '--output',str(a.output),'--template-plan',str(a.template_plan),'--seconds',str(a.seconds),
         '--processed-cap-bytes',str(a.processed_cap_bytes)]
    if existing:child=existing
    else:
        log=(a.output/'reconstruction_worker.log').open('a')
        child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL,start_new_session=True)
    state.update(reconstruction_pid=child.pid,reconstruction_command=cmd)
    state['reconstruction_reused']=bool(existing)
    published=set()
    ready_path=a.output/'ready.jsonl'
    if a.resume and ready_path.exists():
        with ready_path.open() as f:
            for line in f:
                if line.endswith('\n'):published.add(json.loads(line)['key'])

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
            # Batch acquisition follows physical ZIP order for contiguous reads.
            used=set();packets=[]
            for n in paired:
                names=[n,n[:-5]+'.mp4'];packets.append((n,names,True));used.update(names)
            packets.extend((n,[n],False) for n in sorted(set(files)-used))
            jobs=queue.Queue()
            for group in packet_groups(packets,files,a.prefetch_bytes):jobs.put(group)
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
                        try:group=jobs.get_nowait()
                        except queue.Empty:break
                        try:
                            pending=[]
                            for member,names,is_pair in group:
                                key=archive+'/'+hashlib.sha256(member.encode()).hexdigest()[:20]
                                with lock:already_published=key in published
                                if already_published:
                                    with lock:entry['completed']+=1
                                else:pending.append((member,names,is_pair,key))
                            handle.transferred=0
                            if pending and a.prefetch_bytes:
                                try:
                                    handle.prefetch_members([files[n] for _,names,_,_ in pending for n in names],
                                                            max_bytes=a.prefetch_bytes)
                                except requests.RequestException:
                                    handle.member_bytes=b''
                                    with lock:state['prefetch_retries']=state.get('prefetch_retries',0)+1
                            with lock:state['network_bytes']+=handle.transferred
                            for member,names,is_pair,key in pending:
                                if stop.is_set():break
                                packet_root=a.output/'acquisition'/key
                                manifest_path=packet_root/'download_manifest.json'
                                previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                                if previous.get('status')=='COMPLETED':
                                    with lock:
                                        entry['completed']+=1
                                        if is_pair:
                                            ready=dict(key=key,member=member,raw_hdf5=str(packet_root/'raw'/member),
                                                       source_manifest=str(manifest_path),release_split=split)
                                            with ready_path.open('a') as f:f.write(json.dumps(ready)+'\n')
                                            published.add(key)
                                    continue
                                if previous.get('status')=='FAILED':
                                    with manifest_path.with_name('download_manifest.failed-'+str(time.time_ns())+'.json').open('x') as f:
                                        json.dump(previous,f,indent=2)
                                size=sum(files[n].file_size for n in names)
                                with lock:
                                    if state['raw_verified_bytes']+state['reserved_bytes']+size+16*2**30>a.raw_cap_bytes:
                                        state['stop_reason']='raw storage envelope';stop.set();break
                                    state['reserved_bytes']+=size
                                packet_root.mkdir(parents=True,exist_ok=True)
                                manifest=dict(status='RUNNING',url=url,split=split,archive_size=remote.size,
                                              archive_etag=remote.etag,files=[],training_allowed=False)
                                handle.transferred=0
                                try:
                                    safety()
                                    for name in names:
                                        dest=packet_root/'raw'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                                        part=dest.with_suffix(dest.suffix+'.part')
                                        if dest.exists():
                                            digest=hashlib.sha256();crc=0
                                            with dest.open('rb') as f:
                                                for block in iter(lambda:f.read(8*2**20),b''):
                                                    digest.update(block);crc=zlib.crc32(block,crc)
                                            if dest.stat().st_size!=files[name].file_size or crc!=files[name].CRC:
                                                raise ValueError('failed packet published member differs from ZIP CRC')
                                        else:
                                            info=files[name]
                                            right=min(handle.directory_start,info.header_offset+30+
                                                      len(name.encode('utf-8'))+65535+info.compress_size)
                                            cached=(handle.member_start<=info.header_offset and
                                                    right<=handle.member_start+len(handle.member_bytes))
                                            if a.prefetch_bytes and not cached:
                                                handle.prefetch_members([files[name]],max_bytes=a.prefetch_bytes)
                                            digest=hashlib.sha256()
                                            with source_zip.open(name) as f,part.open('wb') as target:
                                                for block in iter(lambda:f.read(8*2**20),b''):
                                                    safety();target.write(block);digest.update(block)
                                            if part.stat().st_size!=files[name].file_size:raise ValueError('member size mismatch')
                                            part.chmod(0o644)
                                            part.replace(dest)
                                        manifest['files'].append(dict(member=name,bytes=dest.stat().st_size,
                                                                      sha256=digest.hexdigest(),crc32=files[name].CRC))
                                    manifest['status']='COMPLETED';record(manifest_path,manifest)
                                    with lock:
                                        state['completed_packets']+=1;state['raw_verified_bytes']+=size;entry['completed']+=1
                                        if is_pair:
                                            ready=dict(key=key,member=member,raw_hdf5=str(packet_root/'raw'/member),
                                                       source_manifest=str(manifest_path),release_split=split)
                                            with ready_path.open('a') as f:f.write(json.dumps(ready)+'\n')
                                            published.add(key)
                                except Exception as exc:
                                    manifest.update(status='FAILED',error=repr(exc));record(manifest_path,manifest)
                                    with lock:state['failed_packets']+=1;entry['failed']+=1
                                finally:
                                    with lock:
                                        state['network_bytes']+=handle.transferred
                                        state['reserved_bytes']-=size
                        finally:jobs.task_done()
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
    if child.poll() is None and not existing:
        os.kill(child.pid,signal.SIGUSR1)
        try:child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    stop.set();state['pipeline_ended_at']=time.time()
    state['reconstruction_exit_code']=None if existing else child.returncode
    state['reconstruction_left_running']=bool(existing and child.poll() is None);persist()


if __name__=='__main__':
    main()
