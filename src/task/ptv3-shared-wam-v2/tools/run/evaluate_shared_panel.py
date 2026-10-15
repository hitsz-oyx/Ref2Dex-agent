"""Run two frozen shared checkpoint panels concurrently on authorized idle GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--gpus',default='6,7');p.add_argument('--detach',action='store_true')
    a=p.parse_args();run=a.run.resolve();code=a.code_root.resolve()
    root=Path(os.environ.get('REF2DEX_RUN_ROOT',Path(__file__).resolve().parents[5])).resolve()
    if root/'outputs/ptv3-shared-wam-v2' not in run.parents:raise ValueError('Task output required')
    if a.detach:
        with (run/'panel-driver.log').open('x') as f:
            cmd=[sys.executable,str(Path(__file__).resolve())]+[x for x in sys.argv[1:] if x!='--detach']
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,run=str(run))));return
    gpus=a.gpus.split(',')
    if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('Two distinct authorized evaluation GPUs')
    gate=json.loads((run/'smoke-control/native/independent_audit.json').read_text())
    if not gate['pass']:raise ValueError('Shared native engineering audit failed')
    inputs=json.loads((run/'inputs/manifest.json').read_text())
    env=dict(os.environ,REF2DEX_RUN_ROOT=str(root),TORCH_EXTENSIONS_DIR=str(root/'tmp/shared-wam-v2-extensions'))
    entry=code/'src/task/ptv3-shared-wam/tools/run/launch_native.py'
    audit=code/'src/task/ptv3-shared-wam/tools/audit/audit_native.py'
    card=code/'src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261015-shared-native-grasp.md'
    runtime='/home2/wyy/miniconda3/envs/graspenv/bin/python'
    panels=[('airplane','s10_airplane_pass_1',149),('sphere','s4_spheresmall_pass_1',141),('mug','s10_mug_pass_1',172)]
    def worker(item):
        (arm,gpu)=item;checkpoint=inputs[arm]['snapshot'];rows=[]
        for short,sequence,controls in panels:
            out=run/(arm+'-'+short)
            cmd=[runtime,str(entry),'--model-version','v2','--shared-actor','--checkpoint',checkpoint,
                '--output',str(out),'--gpu',gpu,'--controls',str(controls),'--execution-steps','1',
                '--seed','148','--sequence',sequence,'--experiment-card',str(card)]
            with (run/(arm+'-'+short+'-driver.log')).open('x') as f:
                result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
                if result.returncode:raise RuntimeError('Native group failed: '+str(out))
                result=subprocess.run([runtime,str(audit),'--native',str(out/'native')],stdout=f,stderr=subprocess.STDOUT,env=env)
                if result.returncode:raise RuntimeError('Native independent audit failed: '+str(out))
            row=dict(arm=arm,sequence=sequence,group=str(out),
                result=json.loads((out/'native/result.json').read_text()),
                audit=json.loads((out/'native/independent_audit.json').read_text()))
            rows.append(row)
            print(json.dumps(dict(arm=arm,sequence=sequence,outcomes=row['result']['outcomes'])),flush=True)
        return rows
    status=dict(status='RUNNING',supervisor_pid=os.getpid(),inputs=inputs,gpus=gpus,
        code_root=str(code),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip())
    destination=run/'panel-status.json'
    def write():
        part=destination.with_suffix('.part');part.write_text(json.dumps(status,indent=2,allow_nan=False)+'\n');part.replace(destination)
    write()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(worker,zip(('control','forecast'),gpus)))
        status.update(status='COMPLETED',rows=[row for group in rows for row in group],
            scope='Three selected heldout motions, one reset seed, short-episode Probe; student repeat rows are not independent seeds')
    except BaseException as error:
        status.update(status='FAILED',error=repr(error));write();raise
    write()


if __name__=='__main__':main()
