"""Run the pinned n40 panel after GPU456 and the existing handoff release resources."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TOOLS=ROOT/'src/task/wam-rl-policy-v3/tools/run'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--detached',action='store_true');a=p.parse_args();out=a.run.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('task output required')
    reg=json.loads((out/'registration.json').read_text())
    if reg['n_per_arm_per_subset']!=40 or reg['maximum_controls']!=72000 or reg['learning_updates']!=0 or reg['ppo_authorized']:raise ValueError('frozen zero-update n40 budget required')
    if a.detached:
        log=out/'supervisor.log';command=[sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(out)]
        with log.open('x') as handle:child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print(json.dumps(dict(pid=child.pid,command=command,log=str(log))));return
    path=out/'supervisor.json'
    if path.exists():raise ValueError('supervisor already exists; inspect, do not duplicate')
    receipt=dict(status='QUEUED',pid=os.getpid(),start=time.time(),queue_limit_s=10800,maximum_controls=72000,learning_updates=0,commands=[],pids=[])
    previous=out/'queue-before-resource-amendment.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        if old['status']!='QUEUED' or old['pids'] or old['commands']:raise ValueError('only a never-started queue may be replaced')
        receipt.update(start=old['start'],queue_amendment_time=time.time(),previous_queue=str(previous),resource_amendment='release GPU5/6 after prior COVERAGE; GPU4 remains reserved until prior handoff exits; science unchanged')
    def save():
        temporary=path.with_suffix('.new');temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(path)
    def execute(command,log):
        receipt['commands'].append(command);save()
        with log.open('x') as handle:subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    save();children=[];handles=[]
    try:
        pending={g:i for i,g in enumerate((4,5,6))};started_at={}
        # The existing, inspected handoff uses only GPU4 after its COVERAGE
        # stage. Release 5/6 individually without changing scientific jobs.
        later_stages={'BC','BC_EVAL','PPO','PPO_EVAL','DONE'}
        while pending:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            handoff=ROOT/'outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1/manifest.json'
            prior=json.loads(handoff.read_text()) if handoff.exists() else {}
            receipt.update(gpu_latest=gpu,prior_handoff_status=prior.get('status'),prior_handoff_stage=prior.get('stage'),last_check=time.time());save()
            memory={int(row.split(',')[0]):int(row.split(',')[1]) for row in gpu.splitlines()}
            for g,i in list(pending.items()):
                released=prior.get('status') not in ('STARTING','RUNNING','QUEUED') or (g in (5,6) and prior.get('stage') in later_stages)
                if memory.get(g,999999)>64 or not released:continue
                command=[sys.executable,'-u',reg['worker_source'],'--registration',str(out/'registration.json'),'--worker',str(i),'--gpu',str(g),'--output',str(out/('g%d'%g))]
                log=(out/('g%d.log'%g)).open('x');handles.append(log);child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);children.append(child);started_at[child.pid]=time.time()
                receipt.update(status='RUNNING',simulation_start=receipt.get('simulation_start',time.time()),gpu_before=gpu)
                receipt['commands'].append(command);receipt['pids'].append(child.pid);pending.pop(g);receipt['pending_gpus']=list(pending);save()
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('started worker failed; preserve traces, no resampling')
            if pending and time.time()-receipt['start']>receipt['queue_limit_s']:raise TimeoutError('queue budget exhausted')
            if pending:time.sleep(45)
        while any(c.poll() is None for c in children):
            if any(c.poll() is None and time.time()-started_at[c.pid]>reg['worker_wall_s']+300 for c in children):
                # Only directly owned children; never touch another session.
                for child in children:
                    if child.poll() is None:child.terminate()
                raise TimeoutError('registered worker wall exceeded; preserve partial traces')
            time.sleep(30)
        receipt['returncodes']=[c.returncode for c in children];save()
        if any(c.returncode for c in children):raise RuntimeError('worker failure; no resampling or data deletion')
        receipt['status']='VERIFYING';save()
        manifests=[]
        for g in (4,5,6):manifests+=['--manifest',str(out/('g%d'%g)/'manifest.json')]
        execute([sys.executable,str(TOOLS/'summarize_large_grasp.py'),'--registration',str(out/'registration.json'),*manifests,'--output',str(out/'summary.json')],out/'summary.log')
        execute([sys.executable,str(TOOLS/'verify_large_initials.py'),'--arms','A1,A2,A0',*manifests,'--output',str(out/'initial-parity.json')],out/'initial-parity.log')
        execute([sys.executable,str(TOOLS/'verify_grasp_ablation.py'),*manifests,'--output',str(out/'ablation.json')],out/'ablation.log')
        for g in (4,5,6):execute([sys.executable,str(TOOLS/'rescore_large_grasp.py'),'--worker',str(out/('g%d'%g)),'--output',str(out/('g%d-rescore.json'%g)),'--verify-mutant'],out/('g%d-rescore.log'%g))
        receipt.update(status='COMPLETED',elapsed_s=time.time()-receipt['start']);save()
    except Exception as error:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        receipt.update(status='FAILED',error=repr(error),elapsed_s=time.time()-receipt['start']);save();raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':main()
