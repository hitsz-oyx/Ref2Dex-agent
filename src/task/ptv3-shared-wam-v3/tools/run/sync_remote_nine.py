"""Resumable four-stream transfer followed by a detached remote import and CPU audit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--host',required=True)
    p.add_argument('--port',default='22')
    p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--known-hosts',type=Path,required=True)
    p.add_argument('--scope',required=True)
    p.add_argument('--main3',required=True)
    p.add_argument('--remote-python',required=True)
    p.add_argument('--vendor',required=True)
    a=p.parse_args(); a.run.mkdir(exist_ok=True,parents=True)
    ssh=['ssh','-p',a.port,'-i',str(a.identity.resolve()),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(a.known_hosts.resolve())]
    started=time.monotonic()
    def progress(status,**extra):
        v=dict(status=status,pid=os.getpid(),elapsed_s=time.monotonic()-started,scope=a.scope,**extra)
        tmp=a.run/'progress.json.part';tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,a.run/'progress.json')
        print(json.dumps(v),flush=True)
    def remote(code):
        result=subprocess.run(ssh+[a.host,'python3 -'],input=code,text=True,capture_output=True,timeout=60)
        if result.returncode: raise RuntimeError(result.stderr)
        return result.stdout
    try:
        progress('TRANSFERRING',streams=4)
        tasks=list(sorted(a.export.glob('native.part-*')))
        if not tasks: raise ValueError('split native archive into native.part-00 etc before launch')
        tasks += [a.export/'portable.json',a.code,Path(__file__).with_name('remote_portable_import.py')]
        def transfer(path):
            if shutil.disk_usage(a.run).free<20*2**30: raise ValueError('local 20GiB floor')
            if (a.run/'STOP').exists(): raise ValueError('requested STOP')
            command=['rsync','-a','--partial','--append-verify','--info=progress2','-e',shlex.join(ssh),str(path),a.host+':'+a.scope+'/staging/']
            with (a.run/(path.name+'.rsync.log')).open('a') as log:
                for attempt in range(3):
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
                    if result.returncode==0:return path.name
                    if (a.run/'STOP').exists():raise ValueError('requested STOP')
                    time.sleep(10)
            raise RuntimeError('rsync failed after three attempts: '+str(path))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(transfer,path) for path in tasks]
            while not all(f.done() for f in futures):
                done=[f.result() for f in futures if f.done()]
                progress('TRANSFERRING',completed_files=done,total_files=len(tasks))
                time.sleep(30)
            [f.result() for f in futures]
        command=[a.remote_python,a.scope+'/staging/remote_portable_import.py','--scope',a.scope,'--main3',a.main3,'--python',a.remote_python,'--vendor',a.vendor]
        launch="""import json,os,pathlib,subprocess
scope=pathlib.Path(SCOPE)
p=scope/'remote-progress.json'
if p.exists() and json.loads(p.read_text())['status']=='COMPLETED':
 print(p.read_text())
else:
 log=(scope/'remote-supervisor.log').open('a')
 child=subprocess.Popen(COMMAND,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (scope/'remote-launch.json').write_text(json.dumps(dict(pid=child.pid,command=COMMAND),indent=2))
 print(json.dumps(dict(pid=child.pid,status='STARTED')))
""".replace('SCOPE',repr(a.scope)).replace('COMMAND',repr(command))
        response=remote(launch)
        progress('REMOTE_IMPORT_STARTED',receipt=response)
        while True:
            time.sleep(30)
            raw=remote('import pathlib; print(pathlib.Path('+repr(a.scope+'/remote-progress.json')+').read_text())')
            value=json.loads(raw);progress('REMOTE_'+value['status'],remote=value)
            if value['status']=='COMPLETED':break
            if value['status']=='FAILED':raise RuntimeError(value.get('error','remote import failed'))
    except Exception as error:
        progress('FAILED',error=repr(error));raise


if __name__=='__main__':
    main()
