"""Freeze source-only n40, queue GPU456 individually, retain every assigned failure."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
BASELINE=ROOT/'outputs/wam-rl-policy-v3/grasp-v11-coverage40-20261022-r1'


def write(path,obj):
    temporary=path.with_suffix('.new');temporary.write_text(json.dumps(obj,indent=2)+'\n');temporary.replace(path)


def prepare(out,predecessor=None):
    if (out/'registration.json').exists():raise ValueError('already registered; no overwrite')
    out.mkdir(parents=True,exist_ok=True)
    launch=out/'launch-code';shutil.copytree(BASELINE/'frozen-source',launch)
    old=str(BASELINE/'frozen-source')
    for file in launch.rglob('*.py'):
        text=file.read_text()
        if old in text:file.write_text(text.replace(old,str(launch)))
    shutil.copyfile(TASK/'src/wam_rl_policy_v3/source_ceiling.py',launch/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/source_ceiling.py')
    worker=(TASK/'tools/run/run_source_ceiling_worker.py').read_text().replace("ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'", "ROOT=Path(%r);TASK=Path(%r)"%(str(ROOT),str(launch/'src/task/wam-rl-policy-v3')))
    (out/'worker.py').write_text(worker)
    reg=json.loads((BASELINE/'registration.json').read_text())
    reg.update(protocol='source.keygrasp.contact-latched-close.n40.v7',experiment_id='P-20261023-source-controller-ceiling',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        arms=['A0'],subsets=['development_same_v11'],maximum_controls=24000,worker_wall_s=14400,queue_wall_s=14400,learning_updates=0,ppo_authorized=False,
        worker_source=str(out/'worker.py'),runtime_source=str(launch),baseline=str(BASELINE),
        target_successes=32,target_pregrasp_joint_abort_max=2,
        source_clock='none: measured wrist arrival starts closure; measured valid contact/coupling drives acquisition',
        source_geometry='source actual object-local keygrasp and measured preshape; closure uses preceding native PD command, not contact-deflected measured q; R unused; not GT-tau/R',
        closing_geometry='capture world grasp goal at first measured matched contact; no chasing; release capture after .12s measured contact loss/unload',
        approach_rate=dict(far_linear_mps=.75,far_angular_radps=2.4,near_linear_mps=.3,near_angular_radps=1.2,near_surface_m=.06),
        acquisition_probe=dict(maximum_m=.01,rate_mps=.06,trigger='existing measured force-valid grasp_candidate; does not certify acquisition or success'),
        predecessor=str(predecessor.resolve()) if predecessor else None,
        postgrasp='exact baseline RelativePoseServo and hold-force controller; no changes',
        main_rule='source final successes >=32/40; no WAM comparison before this gate',
        scope='same 20 objects x2 sequences; source-only development repair, not tau comparison or new holdout')
    for j in reg['jobs']:j.update(order=['A0'],original_subset=j['subset'],subset='development_same_v11')
    write(out/'registration.json',reg)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--detached',action='store_true');p.add_argument('--predecessor',type=Path);a=p.parse_args();out=a.run.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('task output required')
    if a.prepare:prepare(out,a.predecessor);return
    if a.detached:
        with (out/'supervisor.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--run',str(out)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,log=str(out/'supervisor.log'))));return
    reg=json.loads((out/'registration.json').read_text());path=out/'supervisor.json'
    if path.exists():raise ValueError('inspect existing supervisor, do not duplicate')
    if len(reg['jobs'])!=40 or reg['maximum_controls']!=24000 or any(j['order']!=['A0'] for j in reg['jobs']):raise ValueError('frozen source n40 required')
    receipt=dict(status='QUEUED',pid=os.getpid(),start=time.time(),commands=[],pids=[],pending_gpus=[4,5,6]);children=[];handles=[];started={}
    try:
        pending={g:i for i,g in enumerate((4,5,6))}
        while pending or any(c.poll() is None for c in children):
            if reg.get('predecessor') and not children:
                prior=[]
                for manifest in Path(reg['predecessor']).glob('g[456]/manifest.json'):
                    prior.extend(json.loads(manifest.read_text())['runs'])
                failures=sum(not r['final_success'] for r in prior)
                receipt['predecessor_progress']=dict(completed=len(prior),failures=failures,
                    maximum_possible_successes=40-failures)
                if len(prior)==40 and failures<=8:
                    receipt.update(status='NOT_NEEDED_PREDECESSOR_PASSED');write(path,receipt);return
                if failures<9:
                    write(path,receipt)
                    if time.time()-receipt['start']>reg['queue_wall_s']:raise TimeoutError('predecessor decision/resource queue budget')
                    time.sleep(45);continue
            reading=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in reading.splitlines()}
            receipt.update(gpu_latest=reading,last_check=time.time());write(path,receipt)
            for gpu,worker in list(pending.items()):
                ancestor=reg.get('predecessor');released=True
                while ancestor:
                    ancestor=Path(ancestor);m=ancestor/('g%d'%gpu)/'manifest.json'
                    if not m.exists() or json.loads(m.read_text())['status']!='COMPLETED':released=False;break
                    ancestor=json.loads((ancestor/'registration.json').read_text()).get('predecessor')
                if not released:continue
                if memory.get(gpu,999999)>64:continue
                command=[sys.executable,'-u',reg['worker_source'],'--registration',str(out/'registration.json'),'--worker',str(worker),'--gpu',str(gpu),'--output',str(out/('g%d'%gpu))]
                handle=(out/('g%d.log'%gpu)).open('x');handles.append(handle)
                child=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                children.append(child);started[child.pid]=time.time();pending.pop(gpu)
                receipt.update(status='RUNNING',pending_gpus=list(pending));receipt['pids'].append(child.pid);receipt['commands'].append(command);write(path,receipt)
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError('worker failed; retain exact partial trace, no resampling')
            if pending and time.time()-receipt['start']>reg['queue_wall_s']:raise TimeoutError('resource queue budget; no GPU preemption')
            if any(c.poll() is None and time.time()-started[c.pid]>reg['worker_wall_s']+300 for c in children):raise TimeoutError('worker wall budget')
            if pending or any(c.poll() is None for c in children):time.sleep(45)
        receipt['status']='VERIFYING';write(path,receipt)
        for gpu in (4,5,6):
            cmd=[sys.executable,str(TASK/'tools/run/rescore_large_grasp.py'),'--worker',str(out/('g%d'%gpu)),'--output',str(out/('g%d-rescore.json'%gpu)),'--verify-mutant']
            with (out/('g%d-rescore.log'%gpu)).open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        cmd=[sys.executable,str(TASK/'tools/run/summarize_source_ceiling.py'),'--baseline',str(BASELINE/'summary.json'),'--output',str(out/'summary.json')]
        for gpu in (4,5,6):cmd+=['--manifest',str(out/('g%d'%gpu)/'manifest.json')]
        with (out/'summary.log').open('x') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt.update(status='COMPLETED',elapsed_s=time.time()-receipt['start']);write(path,receipt)
    except BaseException as error:
        # Owned clients only. Their finally blocks close their own native child.
        # Do not signal another session's model or simulator.
        for child in children:
            if child.poll() is None:child.terminate()
        receipt.update(status='FAILED',error=repr(error),elapsed_s=time.time()-receipt['start']);write(path,receipt);raise
    finally:
        for handle in handles:handle.close()


if __name__=='__main__':main()
