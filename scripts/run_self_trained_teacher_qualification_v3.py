"""Two bounded native panels qualifying a frozen self-trained teacher."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,SCIENTIFIC_FILES

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED'
    heads=source/'u20/policy_heads.pt';base=Path(old['base_checkpoint']);hashes=dict(old['input_sha256'])
    for f in SCIENTIFIC_FILES:hashes[str(ROOT/f)]=sha(ROOT/f)
    teacher=Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth').resolve()
    teacher_sha='16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f'
    assert sha(teacher)==teacher_sha
    for f in ['scripts/run_self_trained_teacher_qualification_v3.py','scripts/run_self_trained_teacher_environment_v3.py','scripts/audit_self_trained_teacher_panel_v3.py','docs/decisions/D-20261002-teacher-import-path-correction.md','scripts/run_self_trained_teacher_qualification_v2.py','scripts/run_self_trained_teacher_environment_v2.py','scripts/audit_self_trained_teacher_panel_v2.py','scripts/replay_self_trained_teacher_v2.py','scripts/analyze_self_trained_teacher_v2.py','docs/experiments/probes/P-20261002-self-trained-teacher-qualification-goal-alignment.md','scripts/run_self_trained_teacher_qualification.py','scripts/run_self_trained_teacher_environment.py','scripts/audit_self_trained_teacher_panel.py','scripts/replay_self_trained_teacher.py','scripts/analyze_self_trained_teacher.py','docs/experiments/probes/P-20261002-self-trained-teacher-qualification.md','docs/decisions/D-20261002-after-coherent-substrate-review.md']:
        hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in [source/'run_manifest.json',heads,base,teacher]:hashes[str(f.resolve())]=sha(f)
    def verify():
        for f,h in hashes.items():
            if sha(Path(f))!=h:raise ValueError('protected source drift '+f)
    failed=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261002-self-trained-teacher-qualification-r1'
    prior=json.loads((failed/'run_manifest.json').read_text());assert prior['run_status']=='FAILED'
    prior_seconds=prior['wall_seconds'];prior_bytes=bytes_in(failed)
    second=ROOT/'src/task/CmResidual/research/contact_response/output/P-20261002-self-trained-teacher-qualification-r2'
    second_manifest=json.loads((second/'run_manifest.json').read_text());assert second_manifest['run_status']=='FAILED'
    assert not any(second.glob('s*/initial.pt')) and not any(second.glob('s*/trace.pt'))
    prior_seconds+=second_manifest['wall_seconds'];prior_bytes+=bytes_in(second)
    hashes[str(second/'run_manifest.json')]=sha(second/'run_manifest.json')
    hashes[str(failed/'run_manifest.json')]=sha(failed/'run_manifest.json')
    verify();out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261002-self-trained-teacher-qualification',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),isolated_worktree=str(ROOT),source_run=str(source),import_path_correction=True,failed_import_run=str(second),reference_goal_alignment_correction=True,excluded_old_physics=str(failed),prior_wall_seconds=prior_seconds,prior_bytes=prior_bytes,teacher_checkpoint=str(teacher),teacher_sha256=teacher_sha,input_sha256=hashes,phases=[],wall_limit_seconds=900,storage_limit_bytes=1<<30,base_checkpoint=str(base),policy_sha256=old['policy_sha256'],panel_checkpoints={str(seed):dict(path=str(heads.resolve()),sha256=sha(heads),update=20) for seed in (589,590)})
    def save():m['wall_seconds']=time.monotonic()-begin;m['cumulative_wall_seconds']=prior_seconds+m['wall_seconds'];(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def pick(wait):
        start=time.monotonic();last_notice=-30.
        while True:
            for index in (6,4,5,0,7):
                try:return admission(index)
                except RuntimeError:pass
            elapsed=time.monotonic()-start
            if elapsed>=wait:return None
            if prior_seconds+time.monotonic()-begin>900:raise TimeoutError('whole scientific budget includes idle admission')
            if elapsed-last_notice>=30:
                m['latest_admission_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);save();print(json.dumps(dict(status='WAITING_IDLE_GPU',seconds=elapsed)),flush=True);last_notice=elapsed
            time.sleep(2)
    def execute(name,cmd,kind,timeout):
        verify();gpu=pick(120 if kind=='native' else 0) if kind in ('native','replay') else None
        if kind in ('native','replay') and gpu is None:raise TimeoutError('bounded native admission timeout; preserve valid phases')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'));env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=cmd,admission=gpu,run_status='RUNNING',execution_device='cuda' if gpu else 'cpu',device_reason='native GPU physics' if kind=='native' else ('GPU independent full expert replay' if gpu else 'independent NumPy physics/input/statistical audit'));m['phases'].append(phase);save();start=time.monotonic()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if prior_seconds+time.monotonic()-begin>900 or prior_bytes+bytes_in(out)>1<<30:raise RuntimeError('whole scientific probe budget')
            if gpu:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; stop only owned child')
        print(json.dumps(dict(phase=name,status='STARTED',device=phase['execution_device'],admission=gpu)),flush=True)
        try:run_owned_child(cmd,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,900-prior_seconds-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        for seed in (589,590):
            d=out/f's{seed}';cmd=list(old['phases'][0]['command']);cmd[2]=str(ROOT/'scripts/run_self_trained_teacher_environment_v3.py')
            for flag,value in [('--run-dir',d),('--num_envs',192),('--checkpoint',teacher),('--checkpoint-sha256',teacher_sha),('--eval-seed',seed),('--seed',seed),('--output',d/'unused.json'),('--output_path',d/'player'),('--continuous-checkpoint',heads.resolve()),('--continuous-sha256',sha(heads)),('--expected-update',20)]:cmd[cmd.index(flag)+1]=str(value)
            assert '--deterministic' not in cmd
            execute(f's{seed}',cmd,'native',240);execute(f's{seed}_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_self_trained_teacher_panel_v3.py'),'--directory',str(out),'--panel',str(seed)],'audit',180)
            for f in d.iterdir():
                if f.is_file():hashes[str(f.resolve())]=sha(f)
            execute(f's{seed}_expert_replay',[PYTHON,'-u',str(ROOT/'scripts/replay_self_trained_teacher_v2.py'),'--directory',str(out),'--panel',str(seed)],'replay',180)
        execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_self_trained_teacher_v2.py'),'--directory',str(out)],'audit',60)
        result=json.loads((out/'results.json').read_text());assert result['run_status']=='COMPLETED';m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),cumulative_bytes=prior_bytes+bytes_in(out),inputs_unchanged=True,no_model_or_policy_updates=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
