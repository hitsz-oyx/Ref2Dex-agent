"""Execute four real eight-step replans per Y, with a shared initial panel."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from probe_ref13_progress import ROOT, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args(); root=args.run_dir.resolve()
    root.relative_to(ROOT/'outputs/cm-interaction-oracle')
    task=Path(__file__).resolve().parents[2]
    deadline=(root/'s3-schedule.json').stat().st_mtime+2400
    def run(command):
        if time.time()+120>deadline:
            raise TimeoutError('conditional total wall budget exhausted; retain partial evidence')
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
        if size>4*1024**3:
            raise RuntimeError('conditional artifact budget exhausted')
        if 'ref13_progress_fork.py' in ' '.join(command):
            command+=['--wall-cap',str(min(240,int(deadline-time.time())-15))]
            subprocess.run(command,check=True)
        else:
            subprocess.run(command,check=True,timeout=min(250,max(1,deadline-time.time())))
    def fork(name,reference,offset,candidate=0,plan=None,window=32):
        cmd=[sys.executable,str(task/'tools/run/ref13_progress_fork.py'),
            '--run-dir',str(root),'--gpu',str(args.gpu),'--name',name,
            '--reference',str(reference),'--schedule',str(root/'s3-schedule.json'),
            '--offset',str(offset),'--candidate',str(candidate),'--window',str(window)]
        if plan is not None:
            cmd+=['--plan',str(plan)]
        run(cmd)
    def score(folders,name):
        run([sys.executable,str(task/'tools/audit/score_ref13_progress_forks.py'),
             '--run-dir',str(root),'--gpu',str(args.gpu),'--folders',*folders,
             '--output',str(root/(name+'.json'))])
        return json.loads((root/(name+'.json')).read_text())
    records=[]
    try:
        first=json.loads((root/'initial-scores.json').read_text()) if (root/'initial-scores.json').exists() else score(['initial-k'+str(k) for k in range(7)],'initial-scores')
        # Cache only identical actual reference paths. Each arm otherwise gets
        # its own physically executed prefix, never a mosaic of future forks.
        cache={}; outcomes={}
        for arm in ('old','new'):
            reference=root/'reanchor-r2'
            for offset in (0,8,16,24):
                key=(str(reference),offset)
                if offset==0:
                    scored=first
                elif key in cache:
                    scored=cache[key]
                else:
                    folders=[f'{arm}-o{offset}-k{k}' for k in range(7)]
                    for k,name in enumerate(folders):
                        fork(name,reference,offset,k)
                    scored=score(folders,f'{arm}-o{offset}-scores'); cache[key]=scored
                choices=[0]*96
                for row,choice in zip(scored['rows'],scored[arm+'_choices']):
                    choices[row]=choice
                plan=root/f'{arm}-o{offset}-plan.json'
                write(plan,dict(choices=choices,selector=arm,offset=offset,
                    tie_rule='exact argmax, baseline then candidate order'))
                mixed_key=(str(reference),offset,tuple(choices))
                name=f'{arm}-o{offset}-actual'
                if mixed_key in outcomes:
                    name=outcomes[mixed_key]
                else:
                    fork(name,reference,offset,plan=plan,window=90 if offset==24 else 32)
                    outcomes[mixed_key]=name
                records.append(dict(arm=arm,offset=offset,actual_path=name,choices=choices,
                    differing_choices=scored['differing_choices']))
                reference=root/name
                write(root/'rolling-status.json',dict(status='RUNNING',records=records))
        write(root/'rolling-status.json',dict(status='COMPLETED',records=records))
    except BaseException as error:
        write(root/'rolling-status.json',dict(status='STOPPED',error=str(error),records=records))
        raise


if __name__=='__main__':
    main()
