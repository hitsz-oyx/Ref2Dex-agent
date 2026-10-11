"""Run a bounded synchronous fork from the pinned recovery worker command."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from probe_ref13_progress import ROOT, PYTHON, sha, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--candidate',type=int,choices=range(7),default=0)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--window',type=int,choices=(32,90),default=32)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--reanchor',action='store_true')
    parser.add_argument('--wall-cap',type=int,default=240)
    args=parser.parse_args(); root=args.run_dir.resolve()
    root.relative_to(ROOT/'outputs/cm-interaction-oracle')
    if Path(args.name).name!=args.name or args.offset not in range(0,89,8):
        raise ValueError('unique local name and supported offset required')
    if not 40<=args.wall_cap<=240:
        raise ValueError('bounded worker wall cap required')
    if not json.loads((root/'engineering-audit.json').read_text())['repeat']['old_contract_pass']:
        raise ValueError('historical repeat screen must pass first')
    original=json.loads((root/'baseline-status.json').read_text())
    if original['status']!='COMPLETED':
        raise ValueError('completed baseline required')
    hashes=dict(original['input_sha256'])
    for path in (Path(__file__).resolve(),args.schedule.resolve(),args.reference.resolve()/'trace.pt',
                 args.reference.resolve()/'initial_state.pt'):
        hashes[str(path)]=sha(path)
    if args.plan is not None:
        hashes[str(args.plan.resolve())]=sha(args.plan)
    if any(sha(path)!=digest for path,digest in hashes.items()):
        raise ValueError('pinned recovery inputs changed')
    if subprocess.check_output(['nvidia-smi','-i',str(args.gpu),
            '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied')
    folder=root/args.name
    if folder.exists():
        raise FileExistsError(folder)
    cmd=list(original['command'])
    cmd[cmd.index('--wall-seconds')+1]=str(min(180,args.wall_cap-20))
    for key,value in (('--run-dir',folder),('--output',folder/'native_eval.json'),
                      ('--output_path',folder/'native'),('--candidate',args.candidate)):
        cmd[cmd.index(key)+1]=str(value)
    cmd+=['--reference',str(args.reference.resolve()),'--anchor-schedule',str(args.schedule.resolve()),
          '--group-id','0']
    if args.reanchor:
        if args.candidate!=0 or args.plan is not None or args.offset!=0:
            raise ValueError('baseline reanchor only')
        cmd+=['--reanchor-baseline']
    else:
        cmd+=['--rolling-offset',str(args.offset),'--post-window',str(args.window),'--record-rolling-trace']
        if args.plan is not None:
            cmd+=['--rolling-plan',str(args.plan.resolve())]
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu),TMPDIR=str(ROOT/'tmp'),
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/torch_extensions'),MAX_JOBS='2',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        LD_LIBRARY_PATH=str(PYTHON.parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH',''))
    status=dict(status='RUNNING',command=cmd,input_sha256=hashes,physical_gpu=args.gpu,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        wall_seconds_cap=args.wall_cap,no_training=True)
    status_file=root/(args.name+'-status.json'); started=time.monotonic()
    with (root/(args.name+'.log')).open('x') as log:
        process=subprocess.Popen(cmd,cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT)
        status['pid']=process.pid; write(status_file,status)
        print(json.dumps(dict(name=args.name,pid=process.pid)),flush=True)
        try:
            code=process.wait(timeout=args.wall_cap)
            status.update(status='COMPLETED' if code==0 else 'FAILED',returncode=code)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            status.update(status='FAILED',error='owned worker interrupted or timed out')
            raise
        finally:
            status['elapsed_s']=time.monotonic()-started
            write(status_file,status)
    if any(sha(path)!=digest for path,digest in hashes.items()):
        status.update(status='FAILED',error='input drift'); write(status_file,status)
    print(json.dumps(dict(name=args.name,status=status['status'],elapsed_s=status['elapsed_s'])),flush=True)
    if status['status']!='COMPLETED':
        raise RuntimeError('fork failed; preserve log '+str(log.name))


if __name__=='__main__':
    main()
