"""Read-only resumable SFTP acquisition from the user's authorized Windows inbox."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json
import os
from pathlib import Path,PureWindowsPath
import shutil
import subprocess
import threading
import time
import zipfile

FILES={'camera_params.zip':2144,'HOI4D_Hand_pose.zip':425303390,
       'HOI4D_CAD_Model_for_release.zip':1512101743,'HOI4D_annotations.zip':22252507455}
LOCK=threading.Lock()


def write(path,value):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--known-hosts',type=Path,required=True);p.add_argument('--key',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=86400);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if not 60<=a.seconds<=86400 or not 1<=a.workers<=2:p.error('bounded deadline/workers required')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);archives=root/'archives';archives.mkdir(exist_ok=True)
    runtime=a.runtime.resolve();runtime.mkdir(parents=True,exist_ok=True)
    inventory=json.loads(a.inventory.read_text());selected={}
    for row in inventory['Files']:
        if row['Name'] not in FILES:continue
        source=PureWindowsPath(row['FullName']);prefix=PureWindowsPath('D:/BaiduNetdiskDownload')
        if source.parts[:len(prefix.parts)]!=prefix.parts or source.name!=row['Name']:raise ValueError('source outside authorized Windows inbox')
        if row['Name'] in selected or row['Length']!=FILES[row['Name']]:raise ValueError('duplicate or incomplete input')
        path='/'+source.as_posix()
        if any(c in path for c in ['"','\n','\r']):raise ValueError('unsafe SFTP batch path')
        selected[row['Name']]=dict(filename=row['Name'],bytes=row['Length'],source=path,source_last_write_utc=row['LastWriteTimeUtc'])
    if set(selected)!=set(FILES):raise ValueError('four complete official archives required')
    rows=list(selected.values());manifest=root/'windows-input-manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=rows:raise ValueError('source inventory drift; original data retained')
    write(manifest,rows);started=time.monotonic();deadline=started+a.seconds
    completed={};active={};errors=[];stop=threading.Event()
    def should_stop():
        return stop.is_set() or (root/'STOP').exists() or time.monotonic()>deadline or shutil.disk_usage(root).free<20*2**30
    def one(row):
        name=row['filename'];target=archives/name;partial=archives/(name+'.partial')
        if target.exists():
            if target.stat().st_size!=row['bytes']:raise ValueError('existing published size differs')
            with zipfile.ZipFile(target) as z:z.infolist()
            return name
        for attempt in range(8):
            if should_stop():raise InterruptedError('bounded transfer stopped')
            if partial.exists() and partial.stat().st_size>row['bytes']:raise ValueError('partial exceeds official size')
            if not partial.exists() or partial.stat().st_size<row['bytes']:
                batch=runtime/(name+'.sftp');batch.write_text('get -a "'+row['source']+'" "'+str(partial)+'"\nbye\n')
                cmd=['sftp','-b',str(batch),'-P','2222','-i',str(a.key.resolve()),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
                    '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(a.known_hosts.resolve()),'-o','ConnectTimeout=15',
                    '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-B','262144','-R','64','oyx@127.0.0.1']
                with (runtime/(name+'.log')).open('ab') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    with LOCK:active[name]=child.pid
                    while child.poll() is None:
                        if should_stop():
                            child.terminate()
                            try:child.wait(timeout=5)
                            except subprocess.TimeoutExpired:child.kill();child.wait()
                            raise InterruptedError('bounded transfer stopped; partial retained')
                        time.sleep(1)
                    with LOCK:active.pop(name,None)
                    code=child.returncode
                if code:
                    if attempt==7:raise RuntimeError('SFTP exhausted retries; partial retained')
                    time.sleep(min(2**attempt,30));continue
            if partial.stat().st_size!=row['bytes']:raise ValueError('received size differs')
            with zipfile.ZipFile(partial) as z:z.infolist()
            if target.exists():raise ValueError('published destination appeared; never overwrite')
            partial.replace(target);return name
        raise RuntimeError('transfer did not publish')
    def publish(status):
        with LOCK:running=dict(active)
        received=sum((archives/(name+'.partial')).stat().st_size for name in FILES if (archives/(name+'.partial')).exists())
        done=sum(FILES[n] for n in completed);elapsed=time.monotonic()-started
        write(root/'acquisition-status.json',dict(status=status,pid=os.getpid(),dataset='hoi4d',transport='authorized Windows reverse SSH/SFTP',
            total_files=4,completed_files=len(completed),completed_bytes=done,active_received_bytes=received,total_bytes=sum(FILES.values()),
            elapsed_s=elapsed,payload_bytes_per_s=(done+received)/max(elapsed,1),active_transfers=running,errors=errors,
            checked_at=time.time(),training_allowed=False))
    publish('RUNNING')
    # Publish the small camera/hand inputs early, then fetch CAD and annotations concurrently.
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(one,selected[n]):n for n in FILES}
        while pending:
            finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for f in finished:
                name=pending.pop(f)
                try:completed[f.result()]=True;print('COMPLETE',name,FILES[name],flush=True)
                except Exception as e:errors.append(dict(filename=name,error_type=type(e).__name__));print('FAILED',name,type(e).__name__,flush=True)
            publish('STOPPING' if should_stop() else 'RUNNING')
    status='COMPLETE' if len(completed)==4 else ('STOPPED' if should_stop() else 'PARTIAL');publish(status)
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
