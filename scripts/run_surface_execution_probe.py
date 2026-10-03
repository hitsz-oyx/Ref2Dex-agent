"""Immutable-input owned-GPU execution/prior qualification without recollection."""
import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in


def main():
    p=argparse.ArgumentParser();p.add_argument('--native-source',type=Path,required=True);p.add_argument('--prior-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();native=a.native_source.resolve();prior=a.prior_source.resolve();assert ROOT in out.parents and not out.exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    hashes={}
    for source in (native,prior):
        manifest=source/'run_manifest.json';assert json.loads(manifest.read_text())['run_status']=='COMPLETED';hashes[str(manifest)]=sha(manifest)
    for name in ('initial.pt','trace.pt','physical_metadata.json','panel_audit.json','results.json'):
        path=native/'s655'/name;hashes[str(path)]=sha(path)
    for name in ('data/provenance.json','data/inspire_train.npz','fit/mano_7168.pt','fit/inspire_7168.pt'):
        path=prior/name;hashes[str(path)]=sha(path)
    for rel in subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines():hashes[str(ROOT/rel)]=sha(ROOT/rel)
    card=ROOT/'docs/experiments/probes/P-20261003-surface-execution-input.md';hashes[str(card)]=sha(card)
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf';hashes[str(urdf.resolve())]=sha(urdf)
    for mesh in ET.parse(urdf).getroot().findall('.//visual/geometry/mesh'):
        filename=mesh.get('filename');path=Path(filename) if Path(filename).is_absolute() else urdf.parent/filename
        hashes[str(path.resolve())]=sha(path)
    def verify():
        for path,digest in hashes.items():assert sha(Path(path))==digest,'protected input drift '+path
    verify();out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261003-surface-execution-input',run_id=out.name,run_status='RUNNING',pid=os.getpid(),
           git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_sha256=hashes,
           source_native=str(native),source_prior=str(prior),phases=[],wall_limit_seconds=900,storage_limit_bytes=1<<30,new_native_ticks=0,new_optimizer_steps=0)
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
            if time.monotonic()-begin>900 or bytes_in(out)>1<<30:raise RuntimeError('qualification budget')
            if gpu:
                for row in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('GPU contention; stop owned child only')
        print(json.dumps(dict(phase=name,status='STARTED',admission=gpu)),flush=True);start=time.monotonic()
        try:
            run_owned_child(command,ROOT,env,out/(name+'.log'),guard,900-(time.monotonic()-begin),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as error:phase.update(run_status='FAILED',error=repr(error));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        execute('smoke',[PYTHON,'-u',str(ROOT/'scripts/smoke_surface_execution.py'),'--source',str(native)],False)
        execute('qualification',[PYTHON,'-u',str(ROOT/'scripts/qualify_surface_execution.py'),'--native-source',str(native),'--prior-source',str(prior),'--output',str(out/'qualified')],True)
        execute('audit',[PYTHON,'-u',str(ROOT/'scripts/audit_surface_execution.py'),'--root',str(out),'--native-source',str(native),'--prior-source',str(prior)],False)
        result=json.loads((out/'qualified/results.json').read_text());audit=json.loads((out/'audit.json').read_text());assert audit['label']==result['label']
        verify();m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True)
        (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(label=result['label'],gates=result['gates'])),flush=True)
    except BaseException as error:m.update(run_status='FAILED',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
