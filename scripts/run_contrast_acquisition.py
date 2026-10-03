"""Owned physical contrast-acquisition probe with protected legacy randomized data."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    base=ROOT/'src/task/CmResidual/research/contact_response/output'
    fit_source=base/'P-20261001-direct-randomized-response-fit-r1'
    test_source=base/'P-20261001-direct-randomized-response-test-r1'
    pool_source=base/'P-20261001-randomized-effect-risk-r2'
    hashes={}
    for folder in (fit_source,test_source):
        assert json.loads((folder/'run_manifest.json').read_text())['run_status']=='COMPLETED'
        assert json.loads((folder/'closeout_audit.json').read_text())['status']=='PASS'
        for name in ('run_manifest.json','closeout_audit.json','fit_data.pt' if folder==fit_source else 'predictions.pt'):
            path=folder/name;hashes[str(path.resolve())]=sha(path)
    previous=json.loads((fit_source/'run_manifest.json').read_text())
    fp=fit_source/'fit_data.pt'
    assert sha(fp)==previous['output_sha256'][str(fp.resolve())]
    for folder,seeds in ((pool_source,(492,493)),(test_source,(494,495))):
        for actor in (286,287):
            for seed in seeds:
                d=folder/f't{actor}_s{seed}';r=json.loads((d/'results.json').read_text())
                assert r['all_windows_complete'] and sha(d/'windows.pt')==r['windows_sha256']
                for name in ('windows.pt','results.json'):
                    path=d/name;hashes[str(path.resolve())]=sha(path)
    for rel in subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines():
        path=ROOT/rel;hashes[str(path)]=sha(path)
    card=ROOT/'docs/experiments/probes/P-20261003-contrast-acquisition.md';hashes[str(card)]=sha(card)
    def verify():
        for path,digest in hashes.items():assert sha(Path(path))==digest,'protected input drift '+path
    verify();out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261003-contrast-acquisition',run_id=out.name,run_status='RUNNING',pid=os.getpid(),
           git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,
           fit_source=str(fit_source),test_source=str(test_source),pool_source=str(pool_source),phases=[],wall_limit_seconds=600,storage_limit_bytes=64<<20,new_native_ticks=0,new_neural_optimizer_updates=0)
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def execute(name,command,gpu_phase):
        verify();gpu=None
        if gpu_phase:
            for number in (6,4,5,7,2):
                try:gpu=admission(number);break
                except RuntimeError:pass
            if gpu is None:raise RuntimeError('no idle GPU; no CPU substitution')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                 XDG_CACHE_HOME=str(out/'cache'),TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,run_status='RUNNING',admission=gpu,execution_device='cuda' if gpu else 'cpu');m['phases'].append(phase);save()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>600 or bytes_in(out)>64<<20:raise RuntimeError('qualification budget')
            if gpu:
                for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention; stop owned child only')
        print(json.dumps(dict(phase=name,status='STARTED',admission=gpu)),flush=True);start=time.monotonic()
        try:
            run_owned_child(command,ROOT,env,out/(name+'.log'),guard,600-(time.monotonic()-begin),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        execute('smoke',[PYTHON,'-u',str(ROOT/'scripts/smoke_contrast_acquisition.py')],False)
        execute('fit',[PYTHON,'-u',str(ROOT/'scripts/fit_contrast_acquisition.py'),'--root',str(out)],True)
        for path in (out/'fit').iterdir():
            if path.is_file():hashes[str(path)]=sha(path)
        execute('audit',[PYTHON,'-u',str(ROOT/'scripts/audit_contrast_acquisition.py'),'--root',str(out)],False)
        result=json.loads((out/'fit/results.json').read_text());audit=json.loads((out/'audit.json').read_text())
        assert audit['run_status']=='COMPLETED' and audit['label']==result['label']
        verify();(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True,new_neural_optimizer_updates=4800)
    except BaseException as error:m.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
