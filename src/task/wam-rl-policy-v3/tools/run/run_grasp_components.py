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
        reg=dict(old,protocol='grasp.v13.components.frozen-v10-15mm.v1',experiment_id='P-20261022-grasp-intent-mechanism',stage=stage,jobs=jobs,arms=arms,maximum_controls=sum(len(j['order']) for j in jobs)*600,worker_wall_s=14400,learning_updates=0)
        write(out/(stage+'-registration.json'),reg)
    return old

def summarize(out):
    import numpy as np
    from scipy.stats import fisher_exact,binomtest
    allrows=[]
    for stage in ('mechanism','comparators'):
        for gpu in (4,5,6):allrows+=json.loads((out/stage/f'g{gpu}/manifest.json').read_text())['runs']
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
    result=dict(groups=groups,comparisons=comparisons,new_controls=sum(r['controls'] for r in allrows if r['arm'] not in ('A1','A2')),learning_updates=0,reused_A1_A2=str(old/'summary-final.json'),rows=allrows)
    write(out/'summary-final.json',result);print(json.dumps(groups,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--detached',action='store_true');p.add_argument('--queue-wall-s',type=int,default=14400);a=p.parse_args();out=a.output.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('task-local output required')
    if a.detached:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(out),'--queue-wall-s',str(a.queue_wall_s)]
        with (out.parent/(out.name+'-supervisor.log')).open('x') as f:child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print(json.dumps(dict(pid=child.pid,command=command)));return
    out.mkdir(parents=True,exist_ok=True)
    if (out/'supervisor.json').exists():raise ValueError('refuse overwrite or duplicate launch')
    register(out);frozen=out/'launch-code';frozen.mkdir()
    for rel in ['src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/pose_consumer.py','src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_components.py','src/task/wam-rl-policy-v3/tools/run/run_grasp_component_worker.py']:
        target=frozen/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    # Worker ROOT resolution must remain repository-based; execute a frozen copy at same depth under launch-code, with explicit root env.
    worker=frozen/'src/task/wam-rl-policy-v3/tools/run/run_grasp_component_worker.py'
    receipt=dict(status='QUEUED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.executable,queued_at=time.time(),queue_wall_s=a.queue_wall_s,maximum_new_controls=192000,learning_updates=0,commands=[]);write(out/'supervisor.json',receipt)
    runtime=ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/g4-resume2/runtime-code'
    for stage in ('mechanism','comparators'):
        receipt.update(status='QUEUED',stage=stage);write(out/'supervisor.json',receipt);queue_start=time.time()
        while True:
            gpu=subprocess.check_output(['nvidia-smi','-i','4,5,6','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            if all(int(line.split(',')[1])<=64 for line in gpu.splitlines()):break
            if time.time()-queue_start>a.queue_wall_s:receipt.update(status='QUEUE_EXPIRED',gpu=gpu);write(out/'supervisor.json',receipt);return
            time.sleep(30)
        children=[];logs=[];(out/stage).mkdir();start=time.time()
        for i,g in enumerate((4,5,6)):
            command=[sys.executable,'-u',str(worker),'--registration',str(out/(stage+'-registration.json')),'--worker',str(i),'--gpu',str(g),'--output',str(out/stage/f'g{g}'),'--runtime-source',str(runtime)]
            log=(out/stage/f'g{g}.log').open('x');logs.append(log);env=dict(os.environ,REF2DEX_COMPONENT_ROOT=str(ROOT),REF2DEX_COMPONENT_CODE=str(frozen));child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True);children.append(child);receipt['commands'].append(command)
        receipt.update(status='RUNNING',stage=stage,pids=[c.pid for c in children],gpu_before=gpu);write(out/'supervisor.json',receipt)
        while any(c.poll() is None for c in children):
            if time.time()-start>14700:
                for c in children:
                    if c.poll() is None:c.terminate()
                break
            time.sleep(30)
        for log in logs:log.close()
        if any(c.poll()!=0 for c in children):receipt.update(status='FAILED',returncodes=[c.poll() for c in children]);write(out/'supervisor.json',receipt);return
        for g in (4,5,6):
            command=[sys.executable,str(TOOLS/'rescore_large_grasp.py'),'--worker',str(out/stage/f'g{g}'),'--output',str(out/stage/f'g{g}-rescore.json')];subprocess.run(command,cwd=ROOT,check=True)
    summarize(out);receipt.update(status='COMPLETED',finished_at=time.time());write(out/'supervisor.json',receipt)
if __name__=='__main__':main()
