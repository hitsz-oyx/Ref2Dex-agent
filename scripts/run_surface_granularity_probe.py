"""Owned single-GPU, fixed-input bounded granularity experiment."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha, admission, PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child, bytes_in
from scripts.prepare_surface_prior_data import file_stat


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    parent_path=a.source/'data/provenance.json'; parent=json.loads(parent_path.read_text())
    assert json.loads((a.source/'run_manifest.json').read_text())['run_status']=='COMPLETED'
    hashes={str(parent_path.resolve()):sha(parent_path)}
    for rel in subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines():hashes[str(ROOT/rel)]=sha(ROOT/rel)
    card=ROOT/'docs/experiments/probes/P-20261003-cm-granularity.md';hashes[str(card)]=sha(card)
    for group,digest in parent['selected_packet_sha256'].items():
        path=a.source/'data'/(group+'.npz');assert sha(path)==digest;hashes[str(path.resolve())]=digest
    out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261003-cm-granularity',run_id=out.name,pid=os.getpid(),run_status='RUNNING',
           git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           input_sha256=hashes,phases=[],wall_limit_seconds=1800,storage_limit_bytes=2<<30,policy_training=False)
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def verify():
        for path,digest in hashes.items():assert sha(Path(path))==digest,'protected input drift '+path
        for path,stat in parent['source_stat'].items():assert file_stat(Path(path))==stat,'source metadata drift '+path
    def execute(name,command,gpu_phase):
        verify();gpu=None
        if gpu_phase:
            for number in (6,4,5,7):
                try:gpu=admission(number);break
                except RuntimeError:pass
            if gpu is None:raise RuntimeError('no idle GPU; no CPU substitution')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                 PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(out/'cache'),TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,run_status='RUNNING',admission=gpu,execution_device='cuda' if gpu else 'cpu');m['phases'].append(phase);save()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>1800 or bytes_in(out)>2<<30:raise RuntimeError('wall/storage bound')
            if gpu:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[v.strip() for v in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention; stop owned child only')
        print(json.dumps(dict(phase=name,status='STARTED',admission=gpu)),flush=True)
        start=time.monotonic()
        try:
            run_owned_child(command,ROOT,env,out/(name+'.log'),guard,1800-(time.monotonic()-begin),spawned)
            verify();phase['run_status']='COMPLETED'
        except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        execute('smoke',[PYTHON,'-u',str(ROOT/'scripts/smoke_surface_granularity.py'),'--packet',str(a.source/'data/mano_train.npz')],False)
        execute('data',[PYTHON,'-u',str(ROOT/'scripts/prepare_surface_granularity_data.py'),'--source',str(a.source),'--output',str(out/'data')],False)
        execute('fit',[PYTHON,'-u',str(ROOT/'scripts/fit_surface_granularity.py'),'--data',str(out/'data'),'--output',str(out/'fit')],True)
        execute('audit',[PYTHON,'-u',str(ROOT/'scripts/audit_surface_granularity_prior.py'),'--root',str(out)],False)
        result=json.loads((out/'fit/results.json').read_text());audit=json.loads((out/'audit.json').read_text());assert audit['label']==result['label']
        verify();m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True,actual_optimizer_updates=result['actual_optimizer_updates'])
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=result['label'],gains=result['gains'])),flush=True)
    except BaseException as error:m.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
