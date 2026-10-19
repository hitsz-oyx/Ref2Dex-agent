"""Wait for bounded preparation stages, then freeze their available data automatically."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v5.storage import read_json


def write(path,value):
    pending=path.with_suffix('.json.part');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path);path.chmod(0o664)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','hands','contacts','geometry','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=21600);p.add_argument('--git-commit',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    write(a.output/'run-manifest.json',dict(pid=os.getpid(),git_commit=a.git_commit,run_id=a.output.name,
        work_version='v5.preparation-finalizer.v1',seconds=a.seconds,min_free_gib=20,
        inputs={name:str(getattr(a,name)) for name in ('plan','hands','contacts','geometry')},python=sys.version))
    while time.monotonic()-started<a.seconds:
        if (a.output/'STOP').exists() or shutil.disk_usage(a.output).free<20*2**30:break
        hand=read_json(a.hands/'progress.json') if (a.hands/'progress.json').exists() else {}
        contact=read_json(a.contacts/'progress.json') if (a.contacts/'progress.json').exists() else {}
        write(a.output/'progress.json',dict(status='WAITING_STAGES',pid=os.getpid(),elapsed_s=time.monotonic()-started,
            hands=hand,contacts=contact,checked_at=time.time()))
        if hand.get('status') in ('COMPLETED','PARTIAL','STOPPED') and contact.get('status')=='COMPLETED_FOR_AVAILABLE_HANDS':
            complete=hand.get('status')=='COMPLETED'
            recipe=a.output/'recipe'
            cmd=[sys.executable,str(Path(__file__).with_name('freeze_dataset.py')),
                '--plan',str(a.plan),'--hands',str(a.hands),'--contacts',str(a.contacts),
                '--geometry',str(a.geometry),'--output',str(recipe),'--git-commit',a.git_commit]
            if complete:cmd+=['--require-complete']
            write(a.output/'progress.json',dict(status='FREEZING_DATASET',pid=os.getpid(),hands_complete=complete))
            with (a.output/'freeze.log').open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
                    timeout=max(1,a.seconds-(time.monotonic()-started)))
            state='COMPLETED' if result.returncode==0 and complete else 'PARTIAL' if result.returncode==0 else 'FAILED'
            write(a.output/'progress.json',dict(status=state,pid=os.getpid(),hands=hand,contacts=contact,
                recipe=str(recipe),full_input_coverage=complete,training_launched=False,elapsed_s=time.monotonic()-started))
            return
        time.sleep(30)
    write(a.output/'progress.json',dict(status='STOPPED',pid=os.getpid(),elapsed_s=time.monotonic()-started,
        reason='deadline/STOP/free-space boundary; completed artifacts retained'))


if __name__=='__main__':main()
