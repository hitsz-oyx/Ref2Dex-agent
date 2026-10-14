"""Bounded recovery of the TACO acquisition and automatic post-conversion interface audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path(__file__).resolve().parents[5]
TOOLS=Path(__file__).resolve().parent


def write(path,value):
    tmp=path.with_suffix('.partial.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--seconds',type=int,default=7200);a=p.parse_args()
    if not 60<=a.seconds<=14400:p.error('bounded supervisor deadline required')
    runtime=a.runtime.resolve();runtime.mkdir(parents=True,exist_ok=True)
    base=REPO/'outputs/3d-data-preparation';source=base/'taco-native-full-20261014-r1';converted=base/'taco-posemesh-full-20261014-r1'
    started=time.monotonic();deadline=started+a.seconds;child=None;attempts=0;events=[]
    env=os.environ.copy();env['TMPDIR']=str(runtime)
    def launch(command,kind):
        with (runtime/(kind+f'-{len(events)}.log')).open('ab') as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        events.append(dict(kind=kind,pid=process.pid,command=command,started_at=time.time()));return process
    def status(value,**fields):
        write(runtime/'progress.json',dict(status=value,pid=os.getpid(),elapsed_s=time.monotonic()-started,
            events=events,checked_at=time.time(),training_allowed=False,**fields))
    while time.monotonic()<deadline:
        if (runtime/'STOP').exists() or (source/'STOP').exists():status('STOPPED');return
        raw=json.loads((source/'acquisition-status.json').read_text())
        if (source/'archives/Hand_Poses.zip').exists():break
        if child is not None and child.poll() is not None:child=None
        if child is None and raw['status'] in ['PARTIAL','STOPPED'] and not (Path('/proc')/str(raw['pid'])).exists():
            if attempts>=4:status('RETRY_BUDGET_EXHAUSTED');return
            child=launch(['python3','-u',str(TOOLS/'fetch_taco_parallel.py'),'--output',str(source),
                '--workers','4','--seconds','3600','--prefer-proxy'],'local');attempts+=1
        status('ACQUIRING',completed_ranges=raw.get('completed_ranges'),total_ranges=raw.get('total_ranges'),
            source_status=raw['status'],local_restarts=attempts)
        time.sleep(30)
    else:status('SUPERVISOR_DEADLINE');return
    while time.monotonic()<deadline:
        if (runtime/'STOP').exists():status('STOPPED');return
        file=converted/'progress.json';progress=json.loads(file.read_text()) if file.exists() else {}
        if progress.get('status')=='FINISHED':break
        if progress.get('status') in ['PARTIAL','WAIT_STOPPED']:
            status('CONVERTER_INCOMPLETE',converter=progress);return
        if progress.get('pid') and not (Path('/proc')/str(progress['pid'])).exists():status('CONVERTER_EXITED',converter=progress);return
        status('CONVERTING',converter=progress);time.sleep(30)
    else:status('SUPERVISOR_DEADLINE');return
    snapshot=base/'taco-interface-snapshot-20261014-r1';snapshot.mkdir(exist_ok=False)
    meta=json.loads((converted/'manifest.json').read_text())
    for kind in ['objects','sequences']:
        for row in meta[kind]:row['path']=os.path.relpath((converted/row['path']).resolve(),snapshot)
    write(snapshot/'manifest.json',meta)
    audit_env=env.copy();audit_env['PYTHONPATH']=str(REPO/'tmp/video-data-preparation-deps')
    status('AUDITING',snapshot=str(snapshot))
    python='/home2/wyy/miniconda3/envs/graspenv/bin/python'
    for name,args in [('audit_posemesh.py',[]),('plot_posemesh.py',['--source-id','5'])]:
        command=[python,str(TOOLS.parent/'audit'/name),'--root',str(snapshot)]+args
        with (runtime/(name+'.log')).open('ab') as log:
            result=subprocess.run(command,cwd=REPO,env=audit_env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if result.returncode:status('AUDIT_FAILED',tool=name,snapshot=str(snapshot));return
    failures=progress.get('failed_sequences',0)
    status('COMPLETE' if failures==0 else 'COMPLETE_WITH_CONVERSION_FAILURES',snapshot=str(snapshot),
        converted_sequences=len(meta['sequences']),converter_failed_sequences=failures)


if __name__=='__main__':main()
