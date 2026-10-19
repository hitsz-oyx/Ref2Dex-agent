"""Bounded local-to-NAS stream: atomic packets first, readiness records second."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--remote',required=True)
    p.add_argument('--ssh-key',required=True);p.add_argument('--known-hosts',required=True)
    p.add_argument('--seconds',type=int,default=21600);a=p.parse_args()
    ssh=f'ssh -i {a.ssh_key} -p 22023 -o UserKnownHostsFile={a.known_hosts}'
    start=time.monotonic();passes=0
    while time.monotonic()-start<a.seconds and not (a.input/'SYNC_STOP').exists():
        for directory in ('packets','records'):
            cmd=['rsync','-az','--timeout=120','--exclude=*.part*','-e',ssh,
                str(a.input/directory)+'/',a.remote+'/'+directory+'/']
            subprocess.run(cmd,check=True,timeout=300)
        passes+=1
        finished=a.input/'export-finished.json'
        if finished.exists():
            # The producer marker can appear during a pass; only publish it
            # after a second complete pass has flushed the final packet/record.
            for directory in ('packets','records'):
                subprocess.run(['rsync','-az','--timeout=120','--exclude=*.part*','-e',ssh,
                    str(a.input/directory)+'/',a.remote+'/'+directory+'/'],check=True,timeout=300)
            subprocess.run(['rsync','-az','-e',ssh,str(finished),a.remote+'/export-finished.json'],check=True,timeout=60)
            print(json.dumps(dict(status='COMPLETED',passes=passes,elapsed_s=time.monotonic()-start)),flush=True)
            return
        print(json.dumps(dict(status='SYNCING',passes=passes,elapsed_s=time.monotonic()-start)),flush=True)
        time.sleep(20)
    print(json.dumps(dict(status='STOPPED',passes=passes,elapsed_s=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
