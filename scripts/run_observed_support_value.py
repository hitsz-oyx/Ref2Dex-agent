"""Two fresh native panels and current measured support task-value fits."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,SCIENTIFIC_FILES

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED'
    heads=Path(old['panel_checkpoints']['547']['path']);base=Path(old['base_checkpoint']);hashes=dict(old['input_sha256'])
    for f in SCIENTIFIC_FILES:hashes[str(ROOT/f)]=sha(ROOT/f)
    teacher=Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth').resolve()
    teacher_sha='16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f'
    assert sha(teacher)==teacher_sha
    for f in ['scripts/run_observed_support_value.py','scripts/audit_observed_support_native_panel.py','scripts/fit_observed_support_value.py','scripts/audit_observed_support_value.py','src/task/CmResidual/observed_support_value.py','docs/experiments/probes/P-20261002-observed-support-task-value.md','docs/decisions/D-20261002-observed-support-task-value.md','scripts/run_object_frame_geometry_environment.py','scripts/audit_truth_successor_native_panel.py','scripts/audit_truth_successor_value.py','src/task/CmResidual/truth_successor_value.py','src/task/CmResidual/measured_geometry_barriers.py']:
        hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in (source/'s547').iterdir():
        if f.is_file():hashes[str(f.resolve())]=sha(f)
    for f in [source/'run_manifest.json',heads,base,teacher]:hashes[str(f.resolve())]=sha(f)
    def verify():
        for f,h in hashes.items():
            if sha(Path(f))!=h:raise ValueError('protected source drift '+f)
    verify();out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261002-observed-support-task-value',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),isolated_worktree=str(ROOT),source_run=str(source),teacher_checkpoint=str(teacher),teacher_sha256=teacher_sha,input_sha256=hashes,phases=[],wall_limit_seconds=600,storage_limit_bytes=768<<20,base_checkpoint=str(base),policy_sha256=old['policy_sha256'],panel_checkpoints={str(seed):dict(path=str(heads.resolve()),sha256=sha(heads),update=0) for seed in (597,598)})
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def pick(wait):
        start=time.monotonic();last_notice=-30.
        while True:
            for index in (6,4,5,0,7):
                try:return admission(index)
                except RuntimeError:pass
            elapsed=time.monotonic()-start
            if elapsed>=wait:return None
            if time.monotonic()-begin>600:raise TimeoutError('whole scientific budget includes idle admission')
            if elapsed-last_notice>=30:
                m['latest_admission_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);save();print(json.dumps(dict(status='WAITING_IDLE_GPU',seconds=elapsed)),flush=True);last_notice=elapsed
            time.sleep(2)
    def execute(name,cmd,kind,timeout):
        verify();gpu=pick(120 if kind=='native' else 0) if kind in ('native','fit') else None
        if kind in ('native','fit') and gpu is None:raise TimeoutError('bounded native admission timeout; preserve valid phases')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'));env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=cmd,admission=gpu,run_status='RUNNING',execution_device='cuda' if gpu else 'cpu',device_reason='native GPU physics' if kind=='native' else ('GPU task-value fit/inference' if gpu else 'independent NumPy physics/input/statistical audit'));m['phases'].append(phase);save();start=time.monotonic()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>600 or bytes_in(out)>768<<20:raise RuntimeError('whole scientific probe budget')
            if gpu:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; stop only owned child')
        print(json.dumps(dict(phase=name,status='STARTED',device=phase['execution_device'],admission=gpu)),flush=True)
        try:run_owned_child(cmd,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,600-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        for seed in (597,598):
            d=out/f's{seed}';cmd=list(old['phases'][0]['command']);cmd[2]=str(ROOT/'scripts/run_object_frame_geometry_environment.py')
            for flag,value in [('--run-dir',d),('--num_envs',768),('--checkpoint',teacher),('--checkpoint-sha256',teacher_sha),('--eval-seed',seed),('--seed',seed),('--output',d/'unused.json'),('--output_path',d/'player'),('--continuous-checkpoint',heads.resolve()),('--continuous-sha256',sha(heads)),('--expected-update',0)]:cmd[cmd.index(flag)+1]=str(value)
            assert '--deterministic' not in cmd
            execute(f's{seed}',cmd,'native',240);execute(f's{seed}_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_observed_support_native_panel.py'),'--directory',str(out),'--panel',str(seed)],'audit',180)
            for f in d.iterdir():
                if f.is_file():hashes[str(f.resolve())]=sha(f)
        execute('fit',[PYTHON,'-u',str(ROOT/'scripts/fit_observed_support_value.py'),'--directory',str(out),'--output',str(out/'fit')],'fit',180)
        execute('value_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_observed_support_value.py'),'--directory',str(out)],'audit',120)
        result=json.loads((out/'fit/results.json').read_text());audit=json.loads((out/'value_audit.json').read_text());assert result['run_status']==audit['run_status']=='COMPLETED' and result['label']==audit['label'];m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True,no_actor_or_cm_updates=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
