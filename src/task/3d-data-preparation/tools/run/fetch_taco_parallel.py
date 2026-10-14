"""Accelerate the large TACO ZIP using verified nonoverlapping resumable ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import shutil
import threading
import time
import requests
from fetch_taco_pilot import BASE, SUFFIX, ARCHIVES


def write(path,value):
    temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seconds',type=int,default=86400)
    p.add_argument('--prefer-proxy',action='store_true',help='try verified proxy route before direct fallback')
    a=p.parse_args();root=a.output.resolve();status=root/'acquisition-status.json'
    if not 1<=a.workers<=8 or not 60<=a.seconds<=86400:p.error('bounded workers/deadline required')
    previous=json.loads(status.read_text());old_pid=previous['pid']
    if previous['status'] not in ('STOPPED','PARTIAL','COMPLETE') or (Path('/proc')/str(old_pid)).exists():
        raise RuntimeError('previous sequential writer must be stopped before range migration')
    rel,total=ARCHIVES['hand'];url=BASE+rel+SUFFIX;archive=root/'archives/Hand_Poses.zip'
    if archive.exists():return
    partial=archive.with_suffix('.zip.partial');state_path=root/'hand-range-status.json'
    chunk=64*2**20
    if state_path.exists():
        state=json.loads(state_path.read_text())
        if state['bytes']!=total or state['chunk_bytes']!=chunk or partial.stat().st_size!=total:
            raise ValueError('resumed native range identity changed')
    else:
        prefix=partial.stat().st_size if partial.exists() else 0
        state=dict(bytes=total,chunk_bytes=chunk,prefix_bytes=prefix,completed=[],status='RUNNING')
        write(state_path,state)
    prefix=state['prefix_bytes'];ranges=[(lo,min(total-1,lo+chunk-1)) for lo in range(prefix,total,chunk)]
    fd=os.open(str(partial),os.O_CREAT|os.O_RDWR,0o644);os.ftruncate(fd,total)
    started=time.monotonic();deadline=started+a.seconds;lock=threading.Lock();active={};network=[0];errors=[]
    def one(index):
        lo,hi=ranges[index]
        for attempt in range(6):
            if time.monotonic()>deadline or (root/'STOP').exists():raise TimeoutError('bounded acquisition stopped')
            if shutil.disk_usage(root).free<20*2**30:raise RuntimeError('physical disk reserve20GiB')
            session=requests.Session();session.trust_env=False
            if (attempt+int(a.prefer_proxy))%2:session.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
            try:
                with session.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(15,30)) as response:
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{total}':
                        raise ValueError('official server did not honor exact range')
                    received=0
                    for data in response.iter_content(1024*1024):
                        if time.monotonic()>deadline or (root/'STOP').exists():raise TimeoutError('stopped/deadline')
                        if received+len(data)>hi-lo+1:raise ValueError('range exceeds declared bytes')
                        offset=0
                        while offset<len(data):offset+=os.pwrite(fd,data[offset:],lo+received+offset)
                        received+=len(data)
                        with lock:active[index]=received;network[0]+=len(data)
                    if received!=hi-lo+1:raise IOError('incomplete range')
                return index
            except (requests.RequestException,IOError):
                with lock:active[index]=0
                if attempt==5:raise RuntimeError('range network failed; completed ranges retained; URL omitted') from None
            finally:session.close()
    other=sum(size for key,(_,size) in ARCHIVES.items() if key!='hand');full=total+other
    def publish(status_value):
        with lock:
            received=sum(active.values());completed_bytes=prefix+sum(ranges[i][1]-ranges[i][0]+1 for i in state['completed'])
        elapsed=time.monotonic()-started
        write(status,dict(status=status_value,pid=os.getpid(),dataset='taco',total_files=3,completed_files=2+(status_value=='COMPLETE'),
            completed_bytes=other+(total if status_value=='COMPLETE' else 0),active_received_bytes=0 if status_value=='COMPLETE' else completed_bytes+received,
            total_bytes=full,elapsed_s=elapsed,network_bytes=network[0],payload_bytes_per_s=network[0]/max(elapsed,1),
            completed_ranges=len(state['completed']),total_ranges=len(ranges),errors=errors,checked_at=time.time(),training_allowed=False))
    publish('RUNNING')
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(one,i):i for i in range(len(ranges)) if i not in state['completed']};pending=set(futures)
            while pending:
                done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    index=futures[f]
                    try:
                        f.result()
                        with lock:state['completed'].append(index);active.pop(index,None)
                        write(state_path,state);print('HAND_RANGES',len(state['completed']),'/',len(ranges),flush=True)
                    except Exception as error:
                        with lock:active.pop(index,None)
                        errors.append(dict(range=index,error_type=type(error).__name__))
                publish('RUNNING')
        os.fsync(fd)
    finally:os.close(fd)
    if not errors and len(state['completed'])==len(ranges):
        partial.replace(archive);state['status']='COMPLETE';publish('COMPLETE')
    else:
        state['status']='PARTIAL';publish('PARTIAL')
    write(state_path,state);write(root/'range-summary.json',dict(status=state['status'],prefix_reused_bytes=prefix,
        completed_ranges=len(state['completed']),network_bytes=network[0],elapsed_s=time.monotonic()-started,errors=errors))


if __name__=='__main__':main()
