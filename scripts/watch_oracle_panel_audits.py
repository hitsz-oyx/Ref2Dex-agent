"""Bounded CPU-only saved-artifact audit queue; never launches scientific collection."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import PYTHON,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--envs',type=int,default=96);a=p.parse_args();source=a.source.resolve()
    if ROOT not in source.parents:raise ValueError('owned run only')
    destination=source/'native_audit_manifest.json'
    if destination.exists():raise ValueError('one audit queue')
    begin=time.monotonic();record=dict(run_status='RUNNING',pid=os.getpid(),phases=[],no_new_model_or_physics=True,wall_limit_seconds=3600)
    def save():record['wall_seconds']=time.monotonic()-begin;destination.write_text(json.dumps(record,indent=2)+'\n')
    save()
    try:
        done=set()
        while time.monotonic()-begin<3600:
            manifest=json.loads((source/'run_manifest.json').read_text())
            panels=sorted(p.parent for p in source.glob('*/*/results.json') if (p.parent/'trace.pt').exists())
            if (source/'baseline/trace.pt').exists():panels=[source/'baseline']+panels
            for panel in panels:
                if str(panel) in done:continue
                if (panel/'native_audit.json').exists():
                    result=json.loads((panel/'native_audit.json').read_text())
                    if result['run_status']!='COMPLETED':raise ValueError('existing audit')
                    done.add(str(panel));continue
                name='_'.join(panel.relative_to(source).parts);phase=dict(panel=str(panel),run_status='RUNNING');record['phases'].append(phase);save()
                metadata=json.loads((panel/'results.json').read_text())
                command=[PYTHON,str(ROOT/'scripts/audit_oracle_native_panel.py'),'--panel',str(panel),'--envs',str(a.envs),'--ticks',str(metadata['ticks'])]
                env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
                with (source/(name+'_native_audit.log')).open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    phase.update(command=command,pid=process.pid);save()
                    try:code=process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.terminate();process.wait(timeout=5);raise
                if code:raise RuntimeError('native audit failed: '+name)
                phase.update(run_status='COMPLETED',sha256=sha(panel/'native_audit.json'));save();done.add(str(panel))
                print(json.dumps(dict(audited=name,status='COMPLETED')),flush=True)
            if manifest['run_status'] in ('COMPLETED','FAILED','STOPPED'):
                record.update(run_status='COMPLETED',scientific_run_status=manifest['run_status'],audited_panels=len(done));break
            time.sleep(10)
        else:raise TimeoutError('bounded audit queue')
    except BaseException as error:record.update(run_status='FAILED',error=repr(error));raise
    finally:save()

if __name__=='__main__':main()
