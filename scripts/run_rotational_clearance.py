"""Owned GPU rotational-clearance adequacy screen with immutable full native trace."""
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
    p=argparse.ArgumentParser();p.add_argument('--execution-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    execution=a.execution_source.resolve()
    ancestor=json.loads((execution/'run_manifest.json').read_text());assert ancestor['run_status']=='COMPLETED'
    assert json.loads((execution/'audit.json').read_text())['run_status']=='COMPLETED'
    hashes=dict(ancestor['input_sha256'])
    for filename in ('run_manifest.json','audit.json','results.json','qualified/train_rows.npz','qualified/held_rows.npz',
                     'qualified/features.npz','qualified/geometry.npz','qualified/execution_fit.json','qualified/joint_predictions.npz'):
        path=execution/filename;hashes[str(path)]=sha(path)
    for rel in subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines():
        path=ROOT/rel
        if str(path) in hashes:assert sha(path)==hashes[str(path)],'inherited code input drift '+str(path)
        hashes[str(path)]=sha(path)
    assets=ROOT/'third_party/DExplore/dexplore/data/assets/mjcf/objects'
    for path in [assets/'airplane/airplane.obj',assets/'table/table.obj']:hashes[str(path.resolve())]=sha(path)
    card=ROOT/'docs/experiments/probes/P-20261003-rotational-clearance-adequacy.md';hashes[str(card)]=sha(card)
    def verify():
        for path,digest in hashes.items():assert sha(Path(path))==digest,'protected input drift '+path
    verify();out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261003-rotational-clearance-adequacy',run_id=out.name,run_status='RUNNING',pid=os.getpid(),
           git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,
           source_execution=str(execution),phases=[],wall_limit_seconds=300,storage_limit_bytes=64<<20,new_native_ticks=0,new_neural_optimizer_updates=0)
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
            if time.monotonic()-begin>300 or bytes_in(out)>64<<20:raise RuntimeError('qualification budget')
            if gpu:
                for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention; stop owned child only')
        print(json.dumps(dict(phase=name,status='STARTED',admission=gpu)),flush=True);start=time.monotonic()
        try:
            run_owned_child(command,ROOT,env,out/(name+'.log'),guard,300-(time.monotonic()-begin),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        execute('smoke',[PYTHON,'-u',str(ROOT/'scripts/smoke_rotational_clearance.py')],False)
        execute('geometry',[PYTHON,'-u',str(ROOT/'scripts/probe_rotational_clearance.py'),'--execution-source',str(execution),'--output',str(out/'geometry')],True)
        execute('audit',[PYTHON,'-u',str(ROOT/'scripts/audit_rotational_clearance.py'),'--root',str(out),'--execution-source',str(execution)],False)
        result=json.loads((out/'geometry/results.json').read_text());audit=json.loads((out/'audit.json').read_text());assert audit['label']==result['label']
        verify();m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=result['label'],gates=result['gates'])),flush=True)
    except BaseException as error:m.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
