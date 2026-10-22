"""Bounded two-stage v13 queue: component mechanism first, then same-interface A3/A4."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TOOLS=ROOT/'src/task/wam-rl-policy-v3/tools/run'

def write(path,data):
    tmp=path.with_suffix('.new');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def register(out):
    old=json.loads((ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/registration.json').read_text())
    for stage,arms in [('mechanism',['WRIST','FINGER','TIME','T0']),('comparators',['A3','A4'])]:
        jobs=[]
        for j in old['jobs']:
            if stage=='mechanism' and j['subset']!='seen':continue
            j=dict(j);shift=j['replicate']%len(arms);j['order']=arms[shift:]+arms[:shift];jobs.append(j)
        reg=dict(old,protocol='grasp.v13.components.frozen-v10-15mm.v1',experiment_id='P-20261022-grasp-intent-mechanism',stage=stage,work_version='source-gt.native11-through-measured-acquisition.v1' if stage=='comparators' else 'component-output-swaps.v1',jobs=jobs,arms=arms,maximum_controls=sum(len(j['order']) for j in jobs)*600,worker_wall_s=14400,learning_updates=0)
        write(out/(stage+'-registration.json'),reg)
    return old

def summarize(out,mechanism_run=None):
    import numpy as np
    from scipy.stats import fisher_exact,binomtest
    allrows=[]
    for stage in ('mechanism','comparators'):
        for gpu in (4,5,6):allrows+=json.loads(((mechanism_run if stage=='mechanism' and mechanism_run is not None else out)/stage/f'g{gpu}/manifest.json').read_text())['runs']
    old=ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1'
    for gpu in (4,5,6):allrows+=json.loads((old/f'g{gpu}-resume2/manifest.json').read_text())['runs']
    groups={};comparisons={}
    for subset in ('seen','unseen'):
        arms=sorted({r['arm'] for r in allrows if r['subset']==subset});groups[subset]={};comparisons[subset]={}
        for arm in arms:
            rows=[r for r in allrows if r['subset']==subset and r['arm']==arm];n=len(rows);k=sum(r['final_success'] for r in rows);z=1.959963984540054;den=1+z*z/n;center=(k/n+z*z/(2*n))/den;half=z*np.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
            groups[subset][arm]=dict(n=n,success=k,wilson95=[center-half,center+half],hold=sum(r['final_hold'] for r in rows),aborts={cause:sum(r['physical_abort']==cause for r in rows) for cause in sorted({r['physical_abort'] for r in rows if r['physical_abort']})})
            for control in ('A1','T0'):
                if control not in arms or control==arm:continue
                cr={r['trial']:r for r in allrows if r['subset']==subset and r['arm']==control};ar={r['trial']:r for r in rows};pairs=sorted(ar.keys()&cr.keys());b=sum(ar[i]['final_success'] and not cr[i]['final_success'] for i in pairs);c=sum(cr[i]['final_success'] and not ar[i]['final_success'] for i in pairs);ck=sum(r['final_success'] for r in cr.values())
                comparisons[subset][arm+'_vs_'+control]=dict(fisher_two_sided_p=float(fisher_exact([[k,n-k],[ck,len(cr)-ck]]).pvalue),paired_n=len(pairs),discordant=[b,c],mcnemar_exact_p=1. if b+c==0 else float(binomtest(b,b+c,.5).pvalue))
    # Holm across the three primary single-component versus T0 comparisons.
    names=sorted([a+'_vs_T0' for a in ('WRIST','FINGER','TIME')],key=lambda n:comparisons['seen'][n]['fisher_two_sided_p']);running=0.
    for i,name in enumerate(names):running=max(running,min(1.,(3-i)*comparisons['seen'][name]['fisher_two_sided_p']));comparisons['seen'][name]['holm_primary_p']=running
    result=dict(mechanism_run=str(mechanism_run or out),comparator_run=str(out),groups=groups,comparisons=comparisons,new_controls=sum(r['controls'] for r in allrows if r['arm'] not in ('A1','A2')),learning_updates=0,reused_A1_A2=str(old/'summary-final.json'),rows=allrows)
    engineering=(mechanism_run or out)/'GT-preflight-interruption.json'
    if engineering.exists():
        pilot=json.loads(engineering.read_text());result.update(old_provider_engineering=pilot,all_physical_controls_lower=result['new_controls']+pilot['acknowledged_controls'],all_physical_controls_upper=result['new_controls']+pilot['physical_controls_upper'])
    write(out/'summary-final.json',result);print(json.dumps(groups,indent=2))

def idle_gpus(csv):
    return {int(line.split(',')[0]) for line in csv.splitlines() if int(line.split(',')[0]) in (4,5,6) and int(line.split(',')[1])<=64}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--detached',action='store_true');p.add_argument('--resume-empty-queue',action='store_true');p.add_argument('--comparators-only',action='store_true');p.add_argument('--mechanism-run',type=Path);p.add_argument('--queue-wall-s',type=int,default=14400);a=p.parse_args();out=a.output.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('task-local output required')
    if bool(a.comparators_only)!=bool(a.mechanism_run):raise ValueError('comparators-only needs its completed mechanism run')
    if a.comparators_only:
        a.mechanism_run=a.mechanism_run.resolve()
        if a.resume_empty_queue:raise ValueError('fresh comparator run, not an empty-queue resume')
        for gpu in (4,5,6):
            if json.loads((a.mechanism_run/'mechanism'/f'g{gpu}/manifest.json').read_text())['status']!='COMPLETED' or json.loads((a.mechanism_run/'mechanism'/f'g{gpu}-rescore.json').read_text())['status']!='PASS':raise ValueError('mechanism physical batch and raw rescoring must finish first')
    if a.detached:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--queue-wall-s',str(a.queue_wall_s)]
        if a.resume_empty_queue:command+=['--resume-empty-queue']
        if a.comparators_only:command+=['--comparators-only','--mechanism-run',str(a.mechanism_run)]
        with (out.parent/(out.name+('-supervisor-resume.log' if a.resume_empty_queue else '-supervisor.log'))).open('x') as f:child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print(json.dumps(dict(pid=child.pid,command=command)));return
    out.mkdir(parents=True,exist_ok=True)
    if a.resume_empty_queue:
        receipt=json.loads((out/'supervisor.json').read_text())
        if receipt['status']!='QUEUE_STOPPED_BEFORE_WORKERS' or receipt['commands'] or any((out/stage).exists() for stage in ('mechanism','comparators')):raise ValueError('only an explicitly stopped, never-launched queue may resume')
        frozen=out/'launch-code';receipt['queue_runner_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    else:
        if (out/'supervisor.json').exists():raise ValueError('refuse overwrite or duplicate launch')
        register(out);frozen=out/'launch-code';frozen.mkdir()
        for rel in ['src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/pose_consumer.py','src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_components.py','src/task/wam-rl-policy-v3/tools/run/run_grasp_component_worker.py']:
            target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True)
            source=(a.mechanism_run or ROOT/'outputs/wam-rl-policy-v3/grasp-v13-mechanism-20261022-r1')/'launch-code'/rel if rel.endswith('/pose_consumer.py') else ROOT/rel
            shutil.copyfile(source,target)
        receipt=dict(status='QUEUED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.executable,queued_at=time.time(),queue_wall_s=a.queue_wall_s,maximum_new_controls=96000 if a.comparators_only else 192000,mechanism_run=str(a.mechanism_run) if a.mechanism_run else None,learning_updates=0,commands=[])
    worker=frozen/'src/task/wam-rl-policy-v3/tools/run/run_grasp_component_worker.py'
    receipt.update(status='QUEUED',supervisor_pid=os.getpid());write(out/'supervisor.json',receipt)
    runtime=ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/g4-resume2/runtime-code'
    for stage in (('comparators',) if a.comparators_only else ('mechanism','comparators')):
        queue_start=receipt['queued_at'] if stage=='mechanism' else time.time();children={};logs=[];starts={};pending={4,5,6};(out/stage).mkdir()
        while pending or any(c.poll() is None for c in children.values()):
            gpu=subprocess.check_output(['nvidia-smi','-i','4,5,6','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            if pending and time.time()-queue_start>a.queue_wall_s:
                receipt.update(status='QUEUE_EXPIRED',pending_gpus=sorted(pending));write(out/'supervisor.json',receipt)
                for c in children.values():
                    if c.poll() is None:c.terminate()
                return
            for g in sorted(pending & idle_gpus(gpu)):
                i=(4,5,6).index(g);command=[sys.executable,'-u',str(worker),'--registration',str(out/(stage+'-registration.json')),'--worker',str(i),'--gpu',str(g),'--output',str(out/stage/f'g{g}'),'--runtime-source',str(runtime)]
                log=(out/stage/f'g{g}.log').open('x');logs.append(log);env=dict(os.environ,REF2DEX_COMPONENT_ROOT=str(ROOT),REF2DEX_COMPONENT_CODE=str(frozen));children[g]=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);starts[g]=time.time();pending.remove(g);receipt['commands'].append(command)
            for g,c in children.items():
                if c.poll() is None and time.time()-starts[g]>14700:c.terminate()
            receipt.update(status='RUNNING' if children else 'QUEUED',stage=stage,pids=[c.pid for c in children.values()],pending_gpus=sorted(pending),gpu_latest=gpu);write(out/'supervisor.json',receipt)
            time.sleep(30)
        for log in logs:log.close()
        if any(c.poll()!=0 for c in children.values()):receipt.update(status='FAILED',returncodes={g:c.poll() for g,c in children.items()});write(out/'supervisor.json',receipt);return
        for g in (4,5,6):
            command=[sys.executable,str(TOOLS/'rescore_large_grasp.py'),'--worker',str(out/stage/f'g{g}'),'--output',str(out/stage/f'g{g}-rescore.json')];subprocess.run(command,cwd=ROOT,check=True)
    summarize(out,mechanism_run=a.mechanism_run);receipt.update(status='COMPLETED',finished_at=time.time());write(out/'supervisor.json',receipt)
if __name__=='__main__':main()
