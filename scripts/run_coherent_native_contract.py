"""One bounded native engineering contract with observable idle-device admission."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,SCIENTIFIC_FILES

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED';heads=source/'u20/policy_heads.pt';base=Path(old['base_checkpoint']);hashes=dict(old['input_sha256'])
    for f in SCIENTIFIC_FILES:hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in ['scripts/run_coherent_native_contract.py','scripts/run_coherent_geometry_environment.py','scripts/audit_coherent_geometry_native_panel.py','scripts/check_coherent_motor_request.py','src/task/CmResidual/coherent_motor_request.py','docs/experiments/probes/ENG-20261002-coherent-native-contract.md']:
        hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in [source/'run_manifest.json',heads,base]:hashes[str(f.resolve())]=sha(f)
    def verify():
        for f,h in hashes.items():
            if sha(Path(f))!=h:raise ValueError('protected source drift '+f)
    verify();out.mkdir();begin=time.monotonic();gpu=None
    m=dict(experiment_id='ENG-20261002-coherent-native-contract',run_id=out.name,run_status='RUNNING',engineering_only=True,pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),isolated_worktree=str(ROOT),source_run=str(source),input_sha256=hashes,gpu=None,phases=[],wall_limit_seconds=600,storage_limit_bytes=512<<20,base_checkpoint=str(base),policy_sha256=old['policy_sha256'],panel_checkpoints={'583':dict(path=str(heads.resolve()),sha256=sha(heads),update=20)})
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def execute(name,cmd,gpu_compute,timeout):
        verify();fresh=admission(gpu['index']) if gpu_compute else None
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu_compute else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','');phase=dict(name=name,command=cmd,admission=fresh,run_status='RUNNING');m['phases'].append(phase);save();start=time.monotonic()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>600 or bytes_in(out)>512<<20:raise RuntimeError('bounded engineering budget')
            if gpu_compute:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; stop only owned child')
        print(json.dumps(dict(phase=name,status='STARTED')),flush=True)
        try:run_owned_child(cmd,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,600-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        phase=dict(name='idle_gpu_admission',run_status='RUNNING',wait_limit_seconds=300);m['phases'].append(phase);save();last_notice=-30.
        while gpu is None:
            for index in (6,4,5,0,7):
                try:gpu=admission(index);break
                except RuntimeError:pass
            elapsed=time.monotonic()-begin
            if gpu is not None:break
            if elapsed>=300:raise TimeoutError('idle GPU admission timeout; no native or model work')
            if elapsed-last_notice>=30:
                phase['latest_gpu_snapshot']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader'],text=True);phase['latest_process_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);phase['wait_seconds']=elapsed;save();print(json.dumps(dict(phase='idle_gpu_admission',status='WAITING',seconds=elapsed)),flush=True);last_notice=elapsed
            time.sleep(2)
        phase.update(run_status='COMPLETED',admission=gpu);m['gpu']=gpu;save();d=out/'s583';cmd=list(old['phases'][0]['command']);cmd[2]=str(ROOT/'scripts/run_coherent_geometry_environment.py')
        for flag,value in [('--run-dir',d),('--eval-seed',583),('--seed',583),('--output',d/'unused.json'),('--output_path',d/'player'),('--continuous-checkpoint',heads.resolve()),('--continuous-sha256',sha(heads)),('--expected-update',20)]:cmd[cmd.index(flag)+1]=str(value)
        assert '--deterministic' not in cmd
        execute('s583',cmd,True,240);execute('s583_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_coherent_geometry_native_panel.py'),'--directory',str(out),'--panel','583'],False,180)
        audit=json.loads((d/'panel_audit.json').read_text());assert audit['run_status']=='COMPLETED';m.update(run_status='COMPLETED',native_contract_valid=True,bytes=bytes_in(out),inputs_unchanged=True,no_model_or_policy_optimizer=True)
    except BaseException as e:
        for phase in m['phases']:
            if phase['run_status']=='RUNNING':phase.update(run_status='FAILED',error=repr(e))
        m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
