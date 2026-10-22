"""Launch bounded, committed workers with closed stdio and independent sessions."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--suffix',default='detached');p.add_argument('--carry-suffix',default='interrupted');a=p.parse_args();run=a.run.resolve();receipt=run/('launch-'+a.suffix+'.json')
    if receipt.exists() or not a.suffix.isalnum() or not a.carry_suffix.isalnum():raise ValueError('fresh simple suffix required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if any(int(l.split(',')[1])>64 for l in gpu.splitlines() if int(l.split(',')[0]) in (4,5,6)):raise ValueError('registered GPU busy; queue all workers')
    reg=json.loads((run/'registration.json').read_text());rows=[]
    for worker,card in enumerate((4,5,6)):
        out=run/('g%d-%s'%(card,a.suffix));carry=run/('g%d-%s'%(card,a.carry_suffix))
        if out.exists() or not (carry/'manifest.json').exists():raise ValueError('new output and existing carry required')
        command=[sys.executable,'-u',str(TASK/'tools/run/run_grasp_repair_worker.py'),'--registration',str(run/'registration.json'),'--worker',str(worker),'--gpu',str(card),'--output',str(out),'--carry-run',str(carry)]
        with (run/(out.name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        rows.append(dict(worker=worker,gpu=card,pid=child.pid,command=command,output=str(out),carry=str(carry)))
    result=dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),time_unix=time.time(),protocol=reg['protocol'],gpu_before=gpu,workers=rows)
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
