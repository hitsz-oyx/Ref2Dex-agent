"""Bounded CPU reconstruction of the remaining ten expert reference motions."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within

SEQUENCES=('s7_airplane_lift_Retake','s9_airplane_lift','s7_apple_lift',
           's1_mug_lift','s1_toothpaste_lift','s1_alarmclock_lift',
           's1_cubesmall_lift','s1_duck_lift','s1_phone_lift','s1_waterbottle_lift')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    output=a.output.resolve()
    if not is_within(output,ROOT/'outputs/consequence-evaluator') or output.exists():
        p.error('fresh task-owned output required')
    recovery=TASK/'tools/run/recover_native_motion.py'
    frozen={str(f):sha(f) for f in [Path(__file__).resolve(),recovery]}
    output.mkdir(parents=True)
    record=dict(status='RUNNING',task='consequence-evaluator',run_id=output.name,
                pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                work_version='original-native-reference-recovery',device='cpu',
                cpu_reason='file/geometry reconstruction only; no model fit; does not compete for expert GPU memory',
                sources=frozen,seconds_budget=600,output_budget_bytes=4*2**30,
                sequences=SEQUENCES,recovered=[])
    started=time.monotonic();child=None
    def save():
        record['elapsed_s']=time.monotonic()-started
        path=output/'run_manifest.json';temporary=output/'manifest.partial'
        temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path)
    def stopped(signum,frame):
        raise KeyboardInterrupt('owned reference batch stopped')
    signal.signal(signal.SIGTERM,stopped);save()
    try:
        for name in SEQUENCES:
            destination=output/name
            command=[sys.executable,'-B',str(recovery),'--sequence',name,
                     '--output',str(destination),'--seconds','180']
            record.update(sequence=name,command=command);save()
            with (output/(name+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['child_pid']=child.pid;save()
                while child.poll() is None:
                    if time.monotonic()-started>600:raise TimeoutError('fixed reference batch deadline')
                    if any(sha(path)!=digest for path,digest in frozen.items()):
                        raise RuntimeError('reference batch code drift')
                    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>4*2**30:
                        raise RuntimeError('reference batch exceeds4GiB')
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('reference failed: '+name)
            result=json.loads((destination/'run_manifest.json').read_text())
            if result['status']!='COMPLETED':raise RuntimeError('reference has no successful terminal record')
            record['recovered'].append(dict(sequence=name,path=result['corrected_tensor'],
                                            sha256=result['corrected_sha256'],
                                            audit_sha256=sha(destination/'audit.json')))
            save()
        record['status']='COMPLETED'
    except BaseException as error:
        record.update(status='FAILED',error=repr(error));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save()


if __name__=='__main__':main()
