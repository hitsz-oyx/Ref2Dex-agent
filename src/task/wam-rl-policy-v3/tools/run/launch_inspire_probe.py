"""Bounded supervisor for one TRAIN Inspire BC fit and native 32/full rollouts."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[5]
MODEL_PY='/home2/wyy/miniconda3/envs/graspenv/bin/python'
NATIVE_PY='/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'
TOOLS=ROOT/'src/task/wam-rl-policy-v3/tools/run'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--controls',type=int,default=542)
    p.add_argument('--bc',type=Path,help='Reuse an existing bound BC; no repeated fit')
    a=p.parse_args();a.output=a.output.resolve();a.source=a.source.resolve();a.selection=a.selection.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in a.output.parents or not 32<=a.controls<=600:raise ValueError('bounded Task output required')
    a.output.mkdir(exist_ok=True,parents=True)
    state_path=a.output/'process.json'
    if state_path.exists():raise ValueError('fresh run identity required')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    started=time.monotonic();children=[];handles=[]
    state=dict(status='RUNNING',pid=os.getpid(),run_id=a.output.name,started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        gpus=[6,7],source=str(a.source),seconds_limit=3000,artifact_limit_bytes=2*1024**3,stages=[])
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    def launch(cmd,log):
        handle=open(a.output/log,'x');handles.append(handle)
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);state['stages'].append(dict(command=cmd,pid=child.pid,log=log));save();return child
    last_sample=-10
    def guard():
        nonlocal last_sample
        elapsed=time.monotonic()-started
        if elapsed>3000 or shutil.disk_usage(a.output).free<20*1024**3:raise RuntimeError('time/storage floor')
        if elapsed-last_sample>=10:
            last_sample=elapsed
            size=sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file() and not f.is_symlink())
            if size>2*1024**3:raise RuntimeError('output budget exceeded')
            usage=subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
            with (a.output/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(elapsed_s=elapsed,gpus=usage,bytes=size))+'\n')
    def wait(child,limit):
        stage_start=time.monotonic()
        while child.poll() is None:
            guard()
            if time.monotonic()-stage_start>limit:raise TimeoutError('stage deadline')
            time.sleep(1)
        if child.returncode:raise RuntimeError('child failed '+str(child.pid)+': '+str(child.returncode))
    try:
        if a.bc:
            bc=a.bc.resolve()
        else:
            fit=launch([MODEL_PY,str(TOOLS/'inspire_scheduled_probe.py'),'fit','--source',str(a.source),
                '--output',str(a.output/'fit'),'--gpu','6','--seconds','850'],'fit.log');wait(fit,900)
            bc=a.output/'fit/bc.pt'
        for name,controls in [('smoke32',32),('closed-loop',a.controls)]:
            folder=ROOT/'tmp'/('ins-'+name+'-'+str(os.getpid()));folder.mkdir(mode=0o700)
            sock=folder/'s'
            server=launch([MODEL_PY,str(TOOLS/'inspire_scheduled_probe.py'),'serve','--source',str(a.source),
                '--output',str(a.output/(name+'-model')),'--bc',str(bc),
                '--socket',str(sock),'--gpu','6','--seconds','850'],name+'-model.log')
            deadline=time.monotonic()+120
            while not sock.exists():
                guard()
                if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('WAM server failed to start')
                time.sleep(1)
            peer=launch([NATIVE_PY,str(TOOLS/'inspire_native_peer.py'),'--selection',str(a.selection),
                '--output',str(a.output/(name+'-native')),'--gpu','7','--seed','246','--seconds','800',
                '--capture-current-observation','--controls',str(controls),'--socket',str(sock),
                '--server-pid',str(server.pid)],name+'-native.log')
            wait(peer,850);wait(server,30)
        state['status']='COMPLETED'
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for handle in handles:handle.close()
        state.update(elapsed_s=time.monotonic()-started,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'));save()


if __name__=='__main__':main()
