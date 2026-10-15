"""One owned GPU: engineering train/replay, then bounded pilot fit/replay."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=6);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--detach',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/shadow-finetune' not in out.parents:raise ValueError('fresh output')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    if a.detach:
        cmd=[sys.executable,str(Path(__file__).resolve()),*[x for x in sys.argv[1:] if x!='--detach']]
        with (scratch/'supervisor.log').open('x') as f:
            proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch)))
        print(json.dumps(dict(supervisor_pid=proc.pid,output=str(out),gpu=a.gpu)));return
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
    out.mkdir(parents=True);started=time.monotonic();child=None;stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda number,frame:stop.append(number))
    code=list((TASK/'src').rglob('*.py'))+list((TASK/'tools').rglob('*.py'))
    identity={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in code}
    status=dict(status='RUNNING',supervisor_pid=os.getpid(),gpu=a.gpu,stages=[],max_seconds=3600,
                authorization='user temporary one additional idle GPU for small Shadow test; preserve Inspire runs')
    def write():
        tmp=out/'process.json.part';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(out/'process.json')
    def run(name,cmd):
        nonlocal child
        stage=dict(name=name,command=cmd,status='RUNNING');status['stages'].append(stage);write()
        with (out/(name+'.log')).open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(scratch),OMP_NUM_THREADS='2'))
        stage['pid']=child.pid;write()
        while child.poll() is None:
            if stop:raise RuntimeError('user signal')
            if time.monotonic()-started>3600:raise TimeoutError('bounded pilot pipeline')
            if shutil.disk_usage(out).free<20*2**30:raise RuntimeError('20GiB floor')
            size=int(subprocess.check_output(['du','-sb',str(out)],text=True).split()[0])
            if size>14*10**9:raise RuntimeError('14GB pipeline budget')
            if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=v for p,v in identity.items()):raise RuntimeError('implementation changed')
            time.sleep(5)
        stage.update(status='COMPLETED' if child.returncode==0 else 'FAILED',exit_code=child.returncode);write()
        if child.returncode:raise RuntimeError('stage failed: '+name)
    try:
        train=[sys.executable,str(TASK/'tools/run/train.py'),'--data',str(a.data.resolve()),'--gpu',str(a.gpu)]
        run('smoke-train',train+['--output',str(out/'smoke'),'--smoke','--batch-size','16'])
        evaluate=[sys.executable,str(TASK/'tools/run/evaluate.py'),'--cache',str(a.data.resolve()),'--object-manifest',str(a.panel.resolve()),'--gpu',str(a.gpu)]
        run('smoke-native',evaluate+['--checkpoint',str(out/'smoke/latest.pt'),'--output',str(out/'smoke-native')])
        run('pilot-train',train+['--output',str(out/'pilot'),'--steps',str(a.steps),'--batch-size',str(a.batch_size)])
        run('pilot-native',evaluate+['--checkpoint',str(out/'pilot/best.pt'),'--output',str(out/'pilot-native')])
        status['status']='COMPLETED'
    except BaseException as error:
        status.update(status='FAILED',error=repr(error))
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status['elapsed_s']=time.monotonic()-started;write()
    if status['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
