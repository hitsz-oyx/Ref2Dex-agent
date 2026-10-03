"""Bounded fresh equal-budget physical auxiliary Q training and actual policy evaluation."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,SCIENTIFIC_FILES

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--partial-data-source',type=Path);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists()
    old=json.loads((source/'run_manifest.json').read_text());assert old['run_status']=='COMPLETED'
    heads=Path(old['panel_checkpoints']['547']['path']);base=Path(old['base_checkpoint']);hashes=dict(old['input_sha256'])
    for f in SCIENTIFIC_FILES:hashes[str(ROOT/f)]=sha(ROOT/f)
    teacher=Path('/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth').resolve()
    teacher_sha='16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f'
    assert sha(teacher)==teacher_sha
    for f in ['scripts/run_option_model_policy.py','scripts/audit_statistical_option_panel.py','scripts/analyze_option_model_policy.py','scripts/analyze_option_model_policy.py','src/task/CmResidual/paired_option_opportunity.py','docs/experiments/probes/P-20261003-budgeted-physical-critic.md','docs/decisions/D-20261002-option-model-policy.md','scripts/run_statistical_option_environment.py','scripts/audit_truth_successor_native_panel.py','scripts/audit_truth_successor_value.py','src/task/CmResidual/truth_successor_value.py','src/task/CmResidual/measured_geometry_barriers.py','src/task/CmResidual/option_model_policy.py','scripts/run_paired_option_environment.py','scripts/audit_paired_option_panel.py','scripts/run_trained_option_environment.py','scripts/audit_trained_option_panel.py','scripts/train_option_model_policy.py','scripts/audit_option_model_training.py','scripts/audit_option_feature_contract.py','scripts/check_option_model_gradient_seam.py','scripts/audit_observed_support_value.py']:
        hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in ['scripts/run_budgeted_physical_critic.py', 'scripts/run_budgeted_physical_data_environment.py', 'scripts/audit_budgeted_physical_data_panel.py', 'scripts/train_budgeted_physical_critic.py', 'scripts/audit_budgeted_physical_critic_training.py', 'scripts/audit_budgeted_physical_critic_panel.py', 'scripts/analyze_budgeted_physical_critic.py', 'src/task/CmResidual/budgeted_physical_q.py', 'docs/experiments/probes/P-20261003-budgeted-physical-critic.md', 'docs/decisions/D-20261003-budgeted-physical-critic.md', 'docs/decisions/D-20261003-budget-layout-prelaunch.md','docs/decisions/D-20261003-budget-normalizer-correction.md']:hashes[str(ROOT/f)]=sha(ROOT/f)
    for f in (source/'s547').iterdir():
        if f.is_file():hashes[str(f.resolve())]=sha(f)
    for f in [source/'run_manifest.json',heads,base,teacher]:hashes[str(f.resolve())]=sha(f)
    def verify():
        for f,h in hashes.items():
            if sha(Path(f))!=h:raise ValueError('protected source drift '+f)
    partial=a.partial_data_source.resolve() if a.partial_data_source else None
    if partial:
        prior=json.loads((partial/'run_manifest.json').read_text());assert prior['experiment_id']=='P-20261003-budgeted-physical-critic' and prior['run_status']=='FAILED' and not (partial/'fit').exists()
        assert not Path('/proc/'+str(prior['pid'])).exists()
        assert sha(ROOT/'scripts/run_budgeted_physical_data_environment.py')==prior['input_sha256'][str(ROOT/'scripts/run_budgeted_physical_data_environment.py')]
        panel_result=json.loads((partial/'s651/results.json').read_text());assert panel_result['run_status']=='COMPLETED' and panel_result['budget_panel_kind']=='short' and panel_result['physical_steps_each']==101
        for f in (partial/'s651').iterdir():
            if f.is_file():hashes[str(f.resolve())]=sha(f)
        hashes[str(partial/'run_manifest.json')]=sha(partial/'run_manifest.json')
    verify();out.mkdir();begin=time.monotonic()
    m=dict(experiment_id='P-20261003-budgeted-physical-critic',run_id=out.name,run_status='RUNNING',pid=os.getpid(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),isolated_worktree=str(ROOT),source_run=str(source),teacher_checkpoint=str(teacher),teacher_sha256=teacher_sha,input_sha256=hashes,phases=[],wall_limit_seconds=2400,storage_limit_bytes=3<<30,base_checkpoint=str(base),policy_sha256=old['policy_sha256'],panel_checkpoints={str(seed):dict(path=str(heads.resolve()),sha256=sha(heads),update=0) for seed in (651,652,653,654,655,656,657,658)})
    def save():m['wall_seconds']=time.monotonic()-begin;(out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def pick(wait):
        start=time.monotonic();last_notice=-30.
        while True:
            for index in (6,4,5,0,7):
                try:return admission(index)
                except RuntimeError:pass
            elapsed=time.monotonic()-start
            if elapsed>=wait:return None
            if time.monotonic()-begin>2400:raise TimeoutError('whole scientific budget includes idle admission')
            if elapsed-last_notice>=30:
                m['latest_admission_snapshot']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);save();print(json.dumps(dict(status='WAITING_IDLE_GPU',seconds=elapsed)),flush=True);last_notice=elapsed
            time.sleep(2)
    def execute(name,cmd,kind,timeout):
        verify();gpu=pick(120 if kind=='native' else 0) if kind in ('native','fit') else None
        if kind in ('native','fit') and gpu is None:raise TimeoutError('bounded native admission timeout; preserve valid phases')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'));env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=cmd,admission=gpu,run_status='RUNNING',execution_device='cuda' if gpu else 'cpu',device_reason='native GPU physics' if kind=='native' else ('GPU physical/value model and actor training' if gpu else 'independent NumPy physics/input/statistical audit'));m['phases'].append(phase);save();start=time.monotonic()
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            if time.monotonic()-begin>2400 or bytes_in(out)>3<<30:raise RuntimeError('whole scientific probe budget')
            if gpu:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                for row in rows.splitlines():
                    fields=[x.strip() for x in row.split(',')]
                    if len(fields)==2 and fields[0]==gpu['uuid'] and fields[1]!=str(phase.get('pid')):raise RuntimeError('contention; stop only owned child')
        print(json.dumps(dict(phase=name,status='STARTED',device=phase['execution_device'],admission=gpu)),flush=True)
        try:run_owned_child(cmd,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,2400-(time.monotonic()-begin)),spawned);verify();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(phase=name,status='COMPLETED')),flush=True)
    save()
    try:
        def collect(seed,evaluation=False,kind=None):
            d=out/f's{seed}';cmd=list(old['phases'][0]['command']);cmd[2]=str(ROOT/('scripts/run_trained_option_environment.py' if evaluation else 'scripts/run_budgeted_physical_data_environment.py'))
            for flag,value in [('--run-dir',d),('--num_envs',768),('--checkpoint',teacher),('--checkpoint-sha256',teacher_sha),('--eval-seed',seed),('--seed',seed),('--output',d/'unused.json'),('--output_path',d/'player'),('--continuous-checkpoint',heads.resolve()),('--continuous-sha256',sha(heads)),('--expected-update',0)]:cmd[cmd.index(flag)+1]=str(value)
            assert '--deterministic' not in cmd
            if evaluation:
                actor_file=out/('fit/actors_budget.pt' if seed in (657,658) else 'fit/actors.pt')
                cmd+=['--option-actors',str(actor_file),'--option-actors-sha256',m['panel_actor_sha256'][str(seed)]]
            else:cmd+=['--budget-panel-kind',kind]
            if partial and seed==651:
                d.mkdir()
                for f in (partial/'s651').iterdir():
                    if f.is_file():
                        target=d/f.name;shutil.copy2(f,target);assert sha(target)==hashes[str(f.resolve())]
                m['inherited_native_data']=dict(source=str(partial),panel=651,episodes=768,env_control_ticks=77568,no_recollection=True);save()
            else:execute(f's{seed}',cmd,'native',240)
            audit='scripts/audit_budgeted_physical_critic_panel.py' if evaluation else 'scripts/audit_budgeted_physical_data_panel.py'
            execute(f's{seed}_audit',[PYTHON,'-u',str(ROOT/audit),'--directory',str(out),'--panel',str(seed)],'audit',180)
            for f in d.iterdir():
                if f.is_file():hashes[str(f.resolve())]=sha(f)
        for seed,kind in ((651,'short'),(652,'short'),(653,'common'),(654,'extra')):collect(seed,kind=kind)
        clocks=[json.loads((out/f's{seed}'/'physical_metadata.json').read_text()) for seed in (651,652,653,654)]
        fields=('control_dt','simulation_dt','control_frequency_inverse','simulation_substeps')
        assert all(all(c[k]==clocks[0][k] for k in fields) for c in clocks);m['native_clock']={k:clocks[0][k] for k in fields};save()
        execute('train',[PYTHON,'-u',str(ROOT/'scripts/train_budgeted_physical_critic.py'),'--directory',str(out),'--output',str(out/'fit')],'fit',600)
        trained=json.loads((out/'fit/results.json').read_text());assert trained['run_status']=='COMPLETED';m['panel_actor_sha256']={str(seed):trained['actor_sha256']['budget' if seed in (657,658) else 'common'] for seed in (655,656,657,658)};save()
        for f in (out/'fit').iterdir():
            if f.is_file():hashes[str(f.resolve())]=sha(f)
        execute('training_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_budgeted_physical_critic_training.py'),'--directory',str(out)],'audit',180)
        for seed in (655,656,657,658):collect(seed,True)
        execute('policy_result',[PYTHON,'-u',str(ROOT/'scripts/analyze_budgeted_physical_critic.py'),'--directory',str(out)],'audit',120)
        result=json.loads((out/'results.json').read_text());assert result['run_status']=='COMPLETED';m.update(run_status='COMPLETED',label=result['label'],bytes=bytes_in(out),inputs_unchanged=True,actual_joint_critic_optimizer_steps=6000,physical_auxiliary_updates_subset=3000,actual_actor_optimizer_steps=4000,short_physical_episodes=1536,common_full_episodes=768,extra_full_episodes=768,evaluation_trajectories=3072,env_control_ticks_per_method=310272,actual_total_env_control_ticks=1085952,actual_new_env_control_ticks=1085952-(77568 if partial else 0),inherited_env_control_ticks=77568 if partial else 0)

    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
