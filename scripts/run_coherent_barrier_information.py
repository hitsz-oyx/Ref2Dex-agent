"""Three fresh coherent native panels and matched physical forecasts, bounded and owned."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,SCIENTIFIC_FILES

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--engineering',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();engineering=a.engineering.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    old=json.loads((source/'run_manifest.json').read_text());eng=json.loads((engineering/'run_manifest.json').read_text());assert old['run_status']==eng['run_status']=='COMPLETED' and eng['native_contract_valid'] and eng['no_model_or_policy_optimizer']
    heads=source/'u20/policy_heads.pt';base=Path(old['base_checkpoint']);hashes=dict(old['input_sha256'])
    for f in SCIENTIFIC_FILES:hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in ['scripts/run_coherent_barrier_information.py','scripts/run_coherent_geometry_environment.py','scripts/audit_coherent_geometry_native_panel.py','scripts/audit_coherent_barrier_native_panel.py','scripts/fit_coherent_barrier_information.py','scripts/fit_measured_geometry_barriers.py','scripts/audit_coherent_barrier_information.py','scripts/check_coherent_motor_request.py','src/task/CmResidual/coherent_motor_request.py','src/task/CmResidual/coherent_barrier_information.py','src/task/CmResidual/measured_geometry_barriers.py','docs/experiments/probes/P-20261002-coherent-barrier-information.md']:
        hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in [source/'run_manifest.json',heads,base,engineering/'run_manifest.json',engineering/'s583/panel_audit.json']:hashes[str(f.resolve())]=sha(f)
    def verify():
        for f,h in hashes.items():
            if sha(Path(f))!=h:raise ValueError('protected source drift '+f)
    verify();out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261002-coherent-barrier-information',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),isolated_worktree=str(ROOT),source_run=str(source),engineering_source=str(engineering),input_sha256=hashes,phases=[],wall_limit_seconds=1200,storage_limit_bytes=1<<30,base_checkpoint=str(base),policy_sha256=old['policy_sha256'],panel_checkpoints={str(seed):dict(path=str(heads.resolve()),sha256=sha(heads),update=20) for seed in (584,585,586)})
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def pick(wait):
        start=time.monotonic();last_notice=-30.
        while True:
            for index in (6,4,5,0,7):
                try:return admission(index)
                except RuntimeError:pass
            elapsed=time.monotonic()-start
            if elapsed>=wait:return None
            if time.monotonic()-begin>1200:raise TimeoutError('whole scientific budget includes idle admission')
            if elapsed-last_notice>=30:
                m['latest_admission_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);save();print(json.dumps(dict(status='WAITING_IDLE_GPU',seconds=elapsed)),flush=True);last_notice=elapsed
            time.sleep(2)
    def execute(name,cmd,kind,timeout):
        verify();gpu=pick(120 if kind=='native' else 0) if kind in ('native','fit') else None
        if kind=='native' and gpu is None:raise TimeoutError('bounded native admission timeout; preserve valid phases')
        if kind=='fit':cmd=[*cmd,'--device','cuda' if gpu else 'cpu']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'));env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=cmd,admission=gpu,run_status='RUNNING',execution_device='cuda' if gpu else 'cpu',device_reason='native GPU physics' if kind=='native' else ('GPU fit/inference' if gpu else ('all candidate GPUs occupied; bounded small NN/common CPU batches' if kind=='fit' else 'independent NumPy physics/input/statistical audit')));m['phases'].append(phase);save();start=time.monotonic()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>1200 or bytes_in(out)>1<<30:raise RuntimeError('whole scientific probe budget')
            if gpu:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; stop only owned child')
        print(json.dumps(dict(phase=name,status='STARTED',device=phase['execution_device'],admission=gpu)),flush=True)
        try:run_owned_child(cmd,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,1200-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        for seed in (584,585,586):
            d=out/f's{seed}';cmd=list(old['phases'][0]['command']);cmd[2]=str(ROOT/'scripts/run_coherent_geometry_environment.py')
            for flag,value in [('--run-dir',d),('--eval-seed',seed),('--seed',seed),('--output',d/'unused.json'),('--output_path',d/'player'),('--continuous-checkpoint',heads.resolve()),('--continuous-sha256',sha(heads)),('--expected-update',20)]:cmd[cmd.index(flag)+1]=str(value)
            assert '--deterministic' not in cmd
            execute(f's{seed}',cmd,'native',240);execute(f's{seed}_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_coherent_barrier_native_panel.py'),'--directory',str(out),'--panel',str(seed)],'audit',180)
            for f in d.iterdir():
                if f.is_file():hashes[str(f.resolve())]=sha(f)
        execute('fit',[PYTHON,'-u',str(ROOT/'scripts/fit_coherent_barrier_information.py'),'--source',str(out),'--output',str(out/'fit')],'fit',420)
        result=json.loads((out/'fit/results.json').read_text());assert result['run_status']=='COMPLETED';m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True,no_policy_updates=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
