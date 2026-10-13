"""Two bounded workers collect independently audited current-policy takeovers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import time
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
PYTHON='/home2/wyy/miniconda3/envs/graspenv/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--families',type=int,nargs='+',default=[0]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ticks',type=int,nargs='+',default=[32,38,44,50,56,62])
    a=p.parse_args();out=a.output.resolve()
    if any(t<32 or t>80 for t in a.ticks):raise ValueError('bounded contact ticks required')
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents or any(f not in range(4) for f in a.families):raise ValueError('fresh owned bounded collection required')
    out.mkdir();started=time.monotonic();jobs=[(f,t) for f in a.families for t in a.ticks]
    manifest=dict(status='RUNNING',experiment_id='P-20261013-goal-point-wam-visited-realign',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),jobs=jobs,gpus=[3,4],admissions=[],failures=[])
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    def worker(gpu,assigned):
        entries=[]
        for family,tick in assigned:
            if time.monotonic()-started>1800:raise TimeoutError('1800s collection wave budget')
            run=ROOT/f'outputs/trajectory-policy/wam-ref2-execution-v32-f{family}-t{tick}-r2'
            shift=((0,0,0),(.5,0,0),(-.5,0,0),(0,.5,0))[family]
            cmd=[PYTHON,str(TASK/'tools/run/launch_wam_native_bridge.py'),'--output',str(run),'--gpu',str(gpu),'--controls','128','--expert-recovery','--own-row-recovery','--takeover-tick',str(tick),'--prefix-xyz-mm',*map(str,shift),'--experiment-id',manifest['experiment_id'],'--head-dir',str(ROOT/'outputs/trajectory-policy/wam-ref2-joint-multitakeover-20261012-r1'),'--chunk-dir',str(ROOT/'outputs/trajectory-policy/wam-ref31-own-R-20261013-r1'),'--effect-checkpoint',str(ROOT/'outputs/goal-point-wam/wam-effect-ddp-20261013-r3/train/checkpoint_step50500.pt'),'--control-checkpoint',str(ROOT/'outputs/goal-point-wam/wam-control-context-20261013-r3/stage2_step4500.pt'),'--allow-repaired-chunk']
            with (out/f'f{family}-t{tick}.log').open('w') as log:
                code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if code==0:code=subprocess.call([PYTHON,str(TASK/'tools/audit/audit_wam_recovery.py'),'--native',str(run/'native')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if code:
                entry=dict(family=family,takeover=tick,native=str(run/'native'),status='FAILED',exit_code=code)
                entries.append(entry);print(json.dumps(entry),flush=True)
                # No further jobs on this worker after an engineering/pair failure.
                break
            result=json.loads((run/'native/result.json').read_text())
            entry=dict(family=family,takeover=tick,native=str(run/'native'),status='COMPLETED',held=result['outcomes'][3]['maximum_held_frames'],expert_clipping=result['outcomes'][3]['clipping_rate'],elapsed_s=time.monotonic()-started)
            entries.append(entry);print(json.dumps(entry),flush=True)
        return entries
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,jobs[i::2]) for i,gpu in enumerate((3,4))]
        for future in as_completed(futures):
            for entry in future.result():manifest['admissions' if entry['status']=='COMPLETED' else 'failures'].append(entry)
            save()
    manifest.update(status='COMPLETED' if len(manifest['admissions'])==len(jobs) else 'INCOMPLETE',elapsed_s=time.monotonic()-started)
    save();print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
