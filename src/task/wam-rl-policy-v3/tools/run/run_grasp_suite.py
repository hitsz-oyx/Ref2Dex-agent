"""One reproducible command for fresh v10 or geometry-coverage batches on GPU456."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TOOLS=ROOT/'src/task/wam-rl-policy-v3/tools/run'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['v10','coverage'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--detached',action='store_true');a=p.parse_args();out=a.output.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in out.parents or out.exists():raise ValueError('fresh task-local output required')
    if a.detached:
        # Log beside, not inside, the required fresh run directory.
        log=out.parent/(out.name+'-supervisor.log');command=[sys.executable,str(Path(__file__).resolve()),'--mode',a.mode,'--output',str(out)]
        with log.open('x') as handle:child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        print(json.dumps(dict(pid=child.pid,command=command,log=str(log))));return
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    if any(int(l.split(',')[1])>64 for l in gpu.splitlines() if int(l.split(',')[0]) in (4,5,6)):raise RuntimeError('GPU456 occupied; queue batch')
    if a.mode=='v10':
        subprocess.run([sys.executable,str(TOOLS/'register_grasp_repair.py'),'--output',str(out/'registration.json')],cwd=ROOT,check=True)
    else:subprocess.run([sys.executable,str(TOOLS/'prepare_grasp_coverage.py'),'--output',str(out)],cwd=ROOT,check=True)
    reg=json.loads((out/'registration.json').read_text());children=[];logs=[];commands=[]
    runner='run_grasp_repair_worker.py' if a.mode=='v10' else 'run_grasp_geometry_worker.py'
    for i,g in enumerate((4,5,6)):
        command=[sys.executable,'-u',str(TOOLS/runner),'--registration',str(out/'registration.json'),'--worker',str(i),'--gpu',str(g),'--output',str(out/('g%d'%g))]
        if a.mode=='v10':command+=['--runtime-source',str(ROOT/'outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/g4-resume2/runtime-code')]
        log=(out/('g%d.log'%g)).open('x');logs.append(log);children.append(subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True));commands.append(command)
    receipt=dict(status='RUNNING',mode=a.mode,python=sys.executable,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),commands=commands,pids=[c.pid for c in children],gpu_before=gpu,start=time.time(),maximum_controls=reg['maximum_controls'])
    def save():
        path=out/'supervisor.json';new=path.with_suffix('.new');new.write_text(json.dumps(receipt,indent=2)+'\n');new.replace(path)
    save()
    while any(c.poll() is None for c in children):
        if time.time()-receipt['start']>reg['worker_wall_s']+300:
            for c in children:
                if c.poll() is None:c.terminate()
            break
        time.sleep(30)
    receipt.update(status='COMPLETED' if all(c.poll()==0 for c in children) else 'FAILED',returncodes=[c.poll() for c in children],elapsed_s=time.time()-receipt['start']);save()
    for log in logs:log.close()
    if receipt['status']!='COMPLETED':raise RuntimeError('worker failed; preserve outputs, do not auto-resample')
    if a.mode=='v10':
        command=[sys.executable,str(TOOLS/'summarize_large_grasp.py'),'--registration',str(out/'registration.json'),'--output',str(out/'summary.json')]
        for g in (4,5,6):command+=['--manifest',str(out/('g%d'%g)/'manifest.json')]
    else:command=[sys.executable,str(TOOLS/'summarize_grasp_coverage.py'),'--run',str(out),'--output',str(out/'summary.json')]
    subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
