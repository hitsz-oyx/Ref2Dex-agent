"""Bounded post-training source handoff and one paired GT-R execution screen."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[5]
PYTHON='/home2/wyy/miniconda3/envs/graspenv/bin/python'
TASK=ROOT/'src/task/ptv3-shared-wam'


def choose_idle_gpu(memories, pids, owners, uid):
    """Count own and unidentifiable consumers; other users do not use our quota."""
    active=0
    for gpu,memory in memories.items():
        consumers=pids[gpu]
        if any(owners.get(pid) in (None,uid) for pid in consumers) or (memory>512 and not consumers):
            active+=1
    if active>1:return None  # four remote + one existing local + this probe <= six
    return next((g for g in [5,7] if memories[g]<512 and not pids[g]),None)


def assess(outcomes):
    calibration=all(rows[0]['maximum_held_frames']>=45 for rows in outcomes.values())
    teacher=outcomes['B_repair'][0]['maximum_held_frames'];gt=outcomes['B_repair'][1]
    gate=calibration and gt['maximum_held_frames']>=max(45,.8*teacher) and gt['clipping']<.01
    conclusion='PROMISING' if gate else ('UNPROMISING' if calibration else 'UNCLEAR')
    return calibration,gate,conclusion


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--card',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or out.parent!=ROOT/'outputs/ptv3-shared-wam':raise ValueError('fresh Task output required')
    import yaml
    card=yaml.safe_load(a.card.read_text().split('---',2)[1])
    if card['task']!='ptv3-shared-wam' or 143 not in card['seeds']:raise ValueError('assigned paired execution seed required')
    out.mkdir();started=time.monotonic();deadline=started+14400
    status=dict(status='WAITING_FOR_REPAIR',pid=os.getpid(),candidate=str(a.candidate.resolve()),sequence='s10_airplane_pass_1',
                seed=143,experiment_id=card['experiment_id'],resource='one idle local GPU, remote upper bound four',
                max_gpu_seconds=480,artifact_budget_bytes=1000000000,
                orchestrator_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def write(state,**extra):
        status.update(status=state,elapsed_s=time.monotonic()-started,**extra)
        temp=out/'status.json.part';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(out/'status.json')
    def git(*args):
        return subprocess.run(['git',*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    def storage():
        report=ROOT/'outputs/ptv3-shared-wam-v2/pretrain-main3-remote-80k-20261014-r1/aggregate_budget_status.json'
        d=json.loads(report.read_text())
        if time.time()-d['checked_at']>300:raise RuntimeError('shared budget report stale; do not start new physics')
        if d['combined_bytes']+1_000_000_000>=393_000_000_000:raise RuntimeError('insufficient shared artifact margin')
        if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>=1_000_000_000:raise RuntimeError('qualification artifact budget')
    def idle_gpu():
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memories={int(row.split(',')[0]):int(row.split(',')[1]) for row in lines.splitlines()}
        pids={gpu:list(map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
               '--format=csv,noheader'],text=True).split())) for gpu in memories}
        owners={}
        for consumers in pids.values():
            for pid in consumers:
                try:owners[pid]=Path('/proc',str(pid)).stat().st_uid
                except OSError:owners[pid]=None
        return choose_idle_gpu(memories,pids,owners,os.getuid())
    try:
        while time.monotonic()<deadline:
            deployment=json.loads((ROOT/'tmp/rl660-ref2-deployment/status.json').read_text())
            if deployment['status']=='APPLIED_UNCOMMITTED':break
            if deployment['status']=='STOPPED_WITHOUT_OVERWRITING':raise RuntimeError('repair handoff stopped: '+deployment.get('error',''))
            write('WAITING_FOR_REPAIR');time.sleep(30)
        else:raise TimeoutError('four-hour qualification wait limit')
        metadata=json.loads((ROOT/'tmp/rl660-ref2-deployment/metadata.json').read_text());paths=metadata['paths']
        # Reject all target edits including partial user staging; commit only the
        # reviewed paths, preserving unrelated staged work via git commit --only.
        if git('diff','--cached','--name-only','--',*paths).stdout.strip():raise RuntimeError('repair targets have staged changes')
        for name in paths:
            expected=git('show',metadata['tip']+':'+name).stdout
            if (ROOT/name).read_bytes()!=expected:raise RuntimeError('repair target changed before qualification: '+name)
        git('add','--',*paths)
        git('commit','--only','-m','fix: apply reviewed RL660 control repair after baseline exit','--',*paths)
        commit=git('rev-parse','HEAD').stdout.decode().strip();write('WAITING_FOR_GPU',source_commit=commit)
        gpu=None
        while time.monotonic()<deadline-600:
            storage();gpu=idle_gpu()
            if gpu is not None:break
            write('WAITING_FOR_GPU');time.sleep(30)
        if gpu is None:raise TimeoutError('no GPU within remaining wait budget')
        baseline=ROOT/'outputs/ptv3-shared-wam/rl660-R-control-repair-20261014-r1/A_all8_best.pt'
        checkpoint=(ROOT/'outputs/ptv3-shared-wam/rl660-joint-large-20261014-r1/latest.pt').resolve()
        for name,candidate,controls,seconds in [('engineering',a.candidate,4,120),('A_all8',baseline,128,180),('B_repair',a.candidate,128,180)]:
            storage()
            if idle_gpu()!=gpu:raise RuntimeError('GPU/resource assignment changed between stages')
            destination=out/name
            cmd=[PYTHON,str(TASK/'tools/run/launch_native.py'),'--checkpoint',str(checkpoint),'--retargeter',str(candidate.resolve()),
                 '--sequence',status['sequence'],'--execution-steps','1','--gpu',str(gpu),'--controls',str(controls),
                 '--seed','143','--deadline-seconds',str(seconds),'--experiment-card',str(a.card.resolve()),'--output',str(destination)]
            if name=='engineering':cmd.append('--smoke')
            write('RUNNING',stage=name,gpu=gpu,checkpoint=str(checkpoint))
            with (out/(name+'.log')).open('x') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,TMPDIR=str(out),PYTHONDONTWRITEBYTECODE='1'))
            if result.returncode:raise RuntimeError(name+' native execution failed; preserve logs')
            group=json.loads((destination/'group_status.json').read_text())
            if group['status']!='COMPLETED':raise RuntimeError(name+' group incomplete')
            audit=subprocess.run([PYTHON,str(TASK/'tools/audit/audit_native.py'),'--native',str(destination/'native')],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (out/(name+'-audit.log')).write_bytes(audit.stdout)
            if audit.returncode:raise RuntimeError(name+' independent audit failed')
        outcomes={name:json.loads((out/name/'native/result.json').read_text())['outcomes'] for name in ['A_all8','B_repair']}
        calibration,gate,conclusion=assess(outcomes)
        report=dict(status='COMPLETED',conclusion=conclusion,calibration=calibration,GT_gate=gate,outcomes=outcomes,
                    source_commit=commit,scope='one fixed sequence GT control Probe; no independent generalization or Cm claim')
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');write('COMPLETED',conclusion=conclusion,GT_gate=gate)
    except BaseException as error:
        write('STOPPED',error=repr(error));raise


if __name__=='__main__':main()
