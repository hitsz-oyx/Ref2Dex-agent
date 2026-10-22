"""Detached, bounded v10 continuation; reuse complete records, retain interruptions.

Run from repository root. The supervisor and worker children survive a lost UI.
A stale RUNNING manifest is accepted only after its native/client PIDs have exited.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
WORKER=ROOT/'src/task/wam-rl-policy-v3/tools/run/run_grasp_repair_worker.py'


def write(path,data):
    temporary=path.with_suffix('.new');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def receipt(source,destination):
    m=json.loads((source/'manifest.json').read_text());reg=json.loads((source/'registration.json').read_text())
    if destination.exists():raise ValueError('fresh carry receipt required')
    active=source/m.get('active_episode',{}).get('id','unused')
    process=active/'process.json'
    if process.exists():
        for name,pid in json.loads(process.read_text()).items():
            if name.endswith('_pid') and Path('/proc',str(pid)).exists():raise RuntimeError('source process still alive: '+str(pid))
    completed={r['id'] for r in m['runs']}
    for result in source.glob('p*/result.json'):
        row=json.loads(result.read_text())
        if row['id'] not in completed:raise ValueError('unaccounted complete result; audit before continuation')
    partial=max(0,m['environment_controls']-sum(r['controls'] for r in m['runs']))
    interrupted=m.get('status')!='COMPLETED' and active.is_dir() and active.name not in completed
    # Last manifest checkpoint is every 60 controls. A killed process could have
    # advanced farther; charge the complete episode horizon conservatively.
    reserve=reg['horizon_controls'] if interrupted else 0
    m['environment_controls']+=max(0,reserve-partial)
    m.update(status='INTERRUPTED_CARRY_RECEIPT',source_manifest=str(source/'manifest.json'),
             interrupted_controls_lower_bound=partial,interrupted_controls_upper_bound=reserve,
             interruption_accounting='full horizon charged; actual unsaved controls unknown')
    destination.mkdir()
    for name in ('registration.json','runtime-code'):(destination/name).symlink_to(source/name,target_is_directory=name=='runtime-code')
    for row in m['runs']:(destination/row['id']).symlink_to(source/row['id'],target_is_directory=True)
    write(destination/'manifest.json',m)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-suffix',default='resume');p.add_argument('--attempt',default='resume2');p.add_argument('--detached',action='store_true');a=p.parse_args()
    run=a.run.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in run.parents:raise ValueError('task output required')
    if a.detached:
        command=[sys.executable,str(Path(__file__).resolve()),'--run',str(run),'--source-suffix',a.source_suffix,'--attempt',a.attempt]
        with (run/(a.attempt+'-supervisor.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,command=command,log=str(log.name))));return
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    occupied={int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu.strip().splitlines()}
    if any(occupied[g]>64 for g in (4,5,6)):raise RuntimeError('GPU4/5/6 not all free')
    workers=[];handles=[];status=[]
    for worker,g in enumerate((4,5,6)):
        source=run/('g%d-%s'%(g,a.source_suffix));carry=run/('g%d-%s-carry'%(g,a.attempt));out=run/('g%d-%s'%(g,a.attempt))
        receipt(source,carry)
        command=[sys.executable,'-u',str(WORKER),'--registration',str(run/'registration.json'),'--worker',str(worker),'--gpu',str(g),'--output',str(out),'--carry-run',str(carry)]
        log=(run/('g%d-%s.log'%(g,a.attempt))).open('x');handles.append(log)
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        workers.append(child);status.append(dict(gpu=g,pid=child.pid,command=command,output=str(out)))
    report=dict(status='RUNNING',workers=status,start_time=time.time(),maximum_supervisor_seconds=21600)
    file=run/(a.attempt+'-supervisor.json');write(file,report)
    while any(c.poll() is None for c in workers):
        if time.time()-report['start_time']>21600:
            for c in workers:
                if c.poll() is None:c.terminate()
            report['status']='BUDGET_STOP';break
        time.sleep(30)
    report.update(status='COMPLETED' if all(c.poll()==0 for c in workers) else 'FAILED',returncodes=[c.poll() for c in workers],elapsed_s=time.time()-report['start_time']);write(file,report)
    for log in handles:log.close()
    if report['status']!='COMPLETED':raise RuntimeError('worker failed; preserve output and audit before retry')

if __name__=='__main__':main()
