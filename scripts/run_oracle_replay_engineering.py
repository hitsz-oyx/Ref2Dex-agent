"""Bounded full-scene replay qualification with owned process isolation."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_contact_response_probe import sha, admission, PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child, bytes_in

CONFIG = Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/src/task/CmResidual/research/physical_value/output/P-20260930-cm-physical-value/r7/environment.yaml')
REFERENCES = Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-contact-response/src/task/CmResidual/research/contact_response/output/P-20261001-hold-plateau-reference-r1/run_manifest.json')
POLICY = Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-contact-response/src/task/CmResidual/research/contact_response/output/P-20261002-reference-target-policy-r1/fit/policy.pt')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output', type=Path, required=True)
    p.add_argument('--physics',choices=('gpu','cpu'),default='gpu');a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or ROOT not in out.parents: raise ValueError('unique owned output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True): raise ValueError('fixed clean code')
    files=[CONFIG,REFERENCES,POLICY,Path(__file__),ROOT/'scripts/run_oracle_native_replay.py',ROOT/'src/task/CmResidual/oracle_native.py',ROOT/'third_party/DExplore/dexplore/env/tasks/base_dexplore_task.py',ROOT/'third_party/DExplore/dexplore/env/tasks/dexplore_inspire.py']
    hashes={str(f):sha(f) for f in files};out.mkdir();begin=time.monotonic()
    manifest=dict(run_status='RUNNING',experiment_id='engineering-oracle-native-replay',pid=os.getpid(),phases=[],engineering_only=True,input_sha256=hashes,wall_limit_seconds=600,storage_limit_bytes=1<<30,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        manifest['wall_seconds']=time.monotonic()-begin
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def verify():
        if any(sha(Path(f))!=h for f,h in hashes.items()): raise RuntimeError('input drift')
    save()
    try:
        for name in ('baseline','replay'):
            verify();gpu=admission(6)
            command=[PYTHON,'-u',str(ROOT/'scripts/run_oracle_native_replay.py'),'--output',str(out/name),'--env-config',str(CONFIG),'--references-manifest',str(REFERENCES),'--policy-checkpoint',str(POLICY),'--physics',a.physics]
            if name=='replay': command+=['--source',str(out/'baseline')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'))
            env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
            phase=dict(name=name,command=command,admission=gpu,run_status='RUNNING');manifest['phases'].append(phase);save()
            def spawned(child): phase.update(pid=child.pid,pgid=child.pid);save()
            def guard():
                if time.monotonic()-begin>600 or bytes_in(out)>1<<30: raise RuntimeError('budget')
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')): raise RuntimeError('GPU contention')
            print(json.dumps(dict(phase=name,status='STARTED')),flush=True)
            try:
                run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,270,spawned)
                verify();phase.update(run_status='COMPLETED')
            except BaseException as error:
                phase.update(run_status='FAILED',error=repr(error));raise
            finally: save()
        manifest.update(run_status='COMPLETED',inputs_unchanged=True,bytes=bytes_in(out))
    except BaseException as error:
        manifest.update(run_status='FAILED',error=repr(error));raise
    finally: save()

if __name__=='__main__': main()
