"""Complete only missing official TACO ranges remotely, then merge them into the local archive."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from fetch_taco_pilot import BASE,SUFFIX,ARCHIVES


def write(path,value):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def remote_download(plan,url):
    # Self-contained so the authorized SSH launch sends code and URL only in memory.
    import json,os,time,shutil,requests
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    import threading
    root=Path(plan['remote_root']);root.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+plan['seconds'];started=time.monotonic();network=[0];done=[];errors=[];lock=threading.Lock()
    def save(value):
        tmp=root/'progress.partial.json';tmp.write_text(json.dumps(value));tmp.replace(root/'progress.json')
    def stopped():return (root/'STOP').exists() or time.monotonic()>deadline or shutil.disk_usage(root).free<20*2**30
    def one(row):
        index,lo,hi=row['index'],row['lo'],row['hi'];target=root/f'range-{index:04d}.bin';partial=root/f'range-{index:04d}.partial'
        if target.exists():
            if target.stat().st_size!=hi-lo+1:raise ValueError('existing range length mismatch')
            return index
        failures=0
        while not target.exists():
            if stopped():raise InterruptedError('bounded remote download stopped')
            n=partial.stat().st_size if partial.exists() else 0
            if n>hi-lo+1:raise ValueError('range partial exceeds size')
            if n==hi-lo+1:partial.replace(target);return index
            start=lo+n;end=min(hi,start+8*2**20-1);session=requests.Session();session.trust_env=False
            try:
                with session.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(10,25)) as response:
                    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{plan["archive_bytes"]}':
                        raise ValueError('exact official HTTP range required')
                    count=0
                    with partial.open('ab') as output:
                        for data in response.iter_content(1024*1024):
                            if stopped():raise InterruptedError('bounded remote download stopped')
                            if count+len(data)>end-start+1:raise ValueError('excess range bytes')
                            output.write(data);count+=len(data)
                            with lock:network[0]+=len(data)
                    if count!=end-start+1:raise IOError('short HTTP range')
            except (requests.RequestException,IOError):
                failures+=1
                if failures>=12:raise RuntimeError('remote transport retries exhausted; partial retained') from None
            finally:session.close()
    def publish(status):
        save(dict(status=status,pid=os.getpid(),completed=done,errors=errors,network_bytes=network[0],
            payload_bytes_per_s=network[0]/max(time.monotonic()-started,1),checked_at=time.time()))
    publish('RUNNING')
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(one,row):row['index'] for row in plan['ranges']}
        while pending:
            finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for f in finished:
                index=pending.pop(f)
                try:done.append(f.result());print('RANGE_COMPLETE',index,flush=True)
                except Exception as error:errors.append(dict(index=index,error_type=type(error).__name__))
            publish('RUNNING')
    publish('COMPLETE' if not errors else 'PARTIAL')


def main():
    import paramiko
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--remote-root',required=True);p.add_argument('--seconds',type=int,default=3600);a=p.parse_args()
    if not 60<=a.seconds<=7200:p.error('bounded remote deadline required')
    root=a.output.resolve();runtime=a.runtime.resolve();runtime.mkdir(parents=True,exist_ok=True)
    # Authenticate before asking a healthy local writer to stop.
    client=paramiko.SSHClient();client.load_host_keys(str(Path.cwd()/'tmp/ssh-4-4090/known_hosts'))
    client.connect('127.0.0.1',port=22023,username='wbcd',key_filename='/home2/wyy/.ssh/id_ed25519',look_for_keys=False,allow_agent=False,timeout=15)
    started=time.monotonic();deadline=started+a.seconds;old=json.loads((root/'acquisition-status.json').read_text())
    proc=Path('/proc')/str(old['pid']);stop=root/'STOP';marker='Owned migration to faster remote TACO range route.\n'
    if proc.exists():
        cmd=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
        if proc.stat().st_uid!=os.getuid() or 'fetch_taco_parallel.py' not in cmd:raise RuntimeError('local writer identity changed')
        if stop.exists():raise RuntimeError('existing STOP preserved; refuse migration')
        stop.write_text(marker)
        until=time.monotonic()+150
        while proc.exists() and time.monotonic()<until:time.sleep(2)
        if proc.exists():raise RuntimeError('local writer still active; preserve all outputs')
    previous=json.loads((root/'acquisition-status.json').read_text())
    if previous['status'] not in ['PARTIAL','STOPPED']:raise RuntimeError('published local source changed')
    if stop.exists():
        if stop.read_text()!=marker:raise RuntimeError('STOP intent changed')
        stop.unlink()
    state=json.loads((root/'hand-range-status.json').read_text());total=ARCHIVES['hand'][1];prefix=state['prefix_bytes'];chunk=state['chunk_bytes']
    ranges=[dict(index=i,lo=lo,hi=min(total-1,lo+chunk-1)) for i,lo in enumerate(range(prefix,total,chunk)) if i not in state['completed']]
    if sum(r['hi']-r['lo']+1 for r in ranges)>4*2**30:raise ValueError('bounded residual download exceeds4GiB')
    plan=dict(remote_root=a.remote_root,archive_bytes=total,ranges=ranges,seconds=a.seconds,
        git_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(runtime/'remote-plan.json',plan)
    code=inspect.getsource(remote_download)
    launch='import os,sys,json,pathlib\n'+code+'\nplan='+repr(plan)+'\nurl='+repr(BASE+ARCHIVES['hand'][0]+SUFFIX)+'''\nroot=pathlib.Path(plan['remote_root']);root.mkdir(parents=True,exist_ok=True)
pid=os.fork()
if pid:
 print(json.dumps(dict(pid=pid,remote_root=str(root))),flush=True);os._exit(0)
os.setsid()
log=os.open(str(root/'acquisition.log'),os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600)
os.dup2(log,1);os.dup2(log,2);os.dup2(os.open('/dev/null',os.O_RDONLY),0)
try:remote_download(plan,url)
except Exception as error:print('FAILED',type(error).__name__,flush=True)
os._exit(0)
'''
    stdin,stdout,stderr=client.exec_command('PYTHONDONTWRITEBYTECODE=1 python3 -',timeout=30)
    stdin.write(launch);stdin.flush();stdin.channel.shutdown_write();receipt=stdout.read().decode();code=stdout.channel.recv_exit_status()
    if code:raise RuntimeError('remote worker launch failed')
    remote_receipt=json.loads(receipt);write(runtime/'remote-launch.json',remote_receipt);print(json.dumps(remote_receipt),flush=True)
    sftp=client.open_sftp();incoming=runtime/'incoming';incoming.mkdir(exist_ok=True)
    fd=os.open(str(root/'archives/Hand_Poses.zip.partial'),os.O_RDWR);errors=[];remote={};other=sum(size for k,(_,size) in ARCHIVES.items() if k!='hand')
    def publish(status):
        assembled=prefix+sum(min(total-1,prefix+i*chunk+chunk-1)-(prefix+i*chunk)+1 for i in state['completed'])
        write(root/'acquisition-status.json',dict(status=status,pid=os.getpid(),dataset='taco',route='remote direct HTTP + pinned SSH/SFTP local merge',
            total_files=3,completed_files=3 if status=='COMPLETE' else 2,completed_bytes=other+(total if status=='COMPLETE' else 0),
            active_received_bytes=0 if status=='COMPLETE' else assembled,total_bytes=total+other,
            network_bytes=remote.get('network_bytes',0),payload_bytes_per_s=remote.get('payload_bytes_per_s',0),
            completed_ranges=len(state['completed']),total_ranges=(total-prefix+chunk-1)//chunk,
            remote_pid=remote_receipt['pid'],errors=errors,checked_at=time.time(),training_allowed=False))
    try:
        while True:
            if time.monotonic()>deadline or stop.exists() or shutil.disk_usage(root).free<20*2**30:
                with sftp.open(a.remote_root+'/STOP','w') as f:f.write('Bounded coordinator stopped.\n')
                publish('STOPPED');break
            try:
                with sftp.open(a.remote_root+'/progress.json','r') as f:remote=json.loads(f.read())
            except FileNotFoundError:time.sleep(3);continue
            for index in remote['completed']:
                if index in state['completed']:continue
                row=next(v for v in ranges if v['index']==index);file=incoming/f'range-{index:04d}.bin'
                sftp.get(a.remote_root+f'/range-{index:04d}.bin',str(file))
                if file.stat().st_size!=row['hi']-row['lo']+1:raise ValueError('remote published range size changed')
                with file.open('rb') as source:
                    offset=row['lo']
                    for data in iter(lambda:source.read(1024*1024),b''):
                        written=0
                        while written<len(data):written+=os.pwrite(fd,data[written:],offset+written)
                        offset+=len(data)
                os.fsync(fd);state['completed'].append(index);write(root/'hand-range-status.json',state);file.unlink()
                print('MERGED',len(state['completed']),flush=True);publish('RUNNING')
            if len(state['completed'])==(total-prefix+chunk-1)//chunk:
                os.fsync(fd);target=root/'archives/Hand_Poses.zip'
                if target.exists():raise ValueError('published archive appeared; never overwrite')
                (root/'archives/Hand_Poses.zip.partial').replace(target);state['status']='COMPLETE';write(root/'hand-range-status.json',state);publish('COMPLETE');break
            if remote['status']!='RUNNING':errors=remote.get('errors',[]);publish('PARTIAL');break
            publish('RUNNING');time.sleep(5)
    finally:os.close(fd);sftp.close();client.close()


if __name__=='__main__':main()
