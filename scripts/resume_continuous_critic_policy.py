"""Bounded exact continuation; no old-worktree mutation or invented exit receipt."""
import argparse,copy,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission,PYTHON

EXPERIMENT='P-20261002-continuous-critic-policy'
EXPECTED_GPU='GPU-33a9c1c9-cccb-61c2-afeb-4ec98be09963'
OLD_RESERVE_SECONDS=400
SCIENTIFIC_FILES=('scripts/run_continuous_critic_environment_v2.py',
    'scripts/audit_continuous_critic_panel.py','scripts/update_continuous_critic_policy.py',
    'scripts/audit_continuous_critic_update.py','scripts/analyze_continuous_critic_policy.py',
    'src/task/CmResidual/continuous_critic_cm.py','src/task/CmResidual/selective_finger_response.py',
    'docs/experiments/probes/P-20261002-continuous-critic-policy.md')

def read(p):return json.loads(Path(p).read_text())
def bytes_in(path):
    total=0
    for p in Path(path).rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:pass  # Temporary compiler/cache files may disappear.
    return total
def write_new(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2)+'\n')

def resume_order():
    order=[('update',2,548)]
    for seed in range(549,567):order.extend([('collect',seed-547,seed),('update',seed-546,seed)])
    return order+[('evaluate',20,568),('evaluate',20,569),('analyze',20,None)]

def assert_source_stable(source,pinned):
    if sha(Path(source)/'run_manifest.json')!=pinned:raise RuntimeError('original source progressed; do not run duplicate work')

def run_owned_child(command,cwd,env,log_path,guard,timeout,on_spawn):
    """Abort only this new process group if old source progresses or budget ends."""
    child=None;begin=time.monotonic()
    try:
        with Path(log_path).open('x') as log:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            on_spawn(child)
            while child.poll() is None:
                guard()
                if time.monotonic()-begin>timeout:raise TimeoutError('owned phase budget')
                try:child.wait(timeout=min(2,max(.1,timeout-(time.monotonic()-begin))))
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('owned phase failed; log retained')
            guard()
        return child.returncode
    except BaseException:
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=5)
        raise

def verify_stop_receipt(receipt,source,manifest,gpu,boot_id,pid_namespace,pid_exists):
    r=read(receipt)
    expected=dict(source_manifest_sha256=sha(Path(source)/'run_manifest.json'),
        original_parent_pid=manifest['pid'],gpu_uuid=EXPECTED_GPU,
        host_boot_id=boot_id,pid_namespace=pid_namespace,
        observed_on_original_host=True,original_process_stopped=True)
    if gpu['uuid']!=EXPECTED_GPU or any(r.get(k)!=v for k,v in expected.items()):raise ValueError('source stop receipt / original-host identity mismatch')
    original_ids=[manifest['pid']]+[p['pid'] for p in manifest['phases'] if p.get('run_status')=='RUNNING' and p.get('pid')]
    if any(pid_exists(pid) for pid in original_ids):raise RuntimeError('original or reused PID still visible; no duplicate launch')
    return r

def rebound_native_command(template,output,seed,heads):
    if seed not in list(range(549,567))+[568,569]:raise ValueError('fixed remaining native cohort only')
    command=list(template);command[2]=str(ROOT/'scripts/run_continuous_critic_environment_v2.py')
    d=Path(output)/f's{seed}'
    for flag,value in (('--run-dir',d),('--eval-seed',seed),('--seed',seed),
        ('--output',d/'unused.json'),('--output_path',d/'player'),
        ('--continuous-checkpoint',heads),('--continuous-sha256',sha(Path(heads))),
        ('--expected-update',seed-547 if seed<567 else 20)):
        command[command.index(flag)+1]=str(value)
    if '--deterministic' in command:command.remove('--deterministic')
    if seed in (568,569):command.append('--deterministic')
    return command

def prepare_plan(source,audited548):
    source=Path(source).resolve();audited548=Path(audited548).resolve();m=read(source/'run_manifest.json');pinned=sha(source/'run_manifest.json')
    if m['experiment_id']!=EXPERIMENT or m['training_seeds']!=list(range(547,567)) or m['evaluation_seeds']!=[568,569]:raise ValueError('fixed full experiment identity')
    if (source/'results.json').exists() or (source/'u02').exists() or (source/'s549').exists():raise ValueError('source advanced beyond the known prefix; inspect before resuming')
    hashes=dict(m['input_sha256'])
    for f,h in hashes.items():
        if sha(Path(f))!=h:raise ValueError('original protected input drift: '+f)
    old_root=Path(m['isolated_worktree'])
    # Validate ALL relocated protected inputs, not only the wrapper's main files.
    for filename,expected in list(hashes.items()):
        original=Path(filename)
        if old_root in original.parents:
            relocated=ROOT/original.relative_to(old_root)
            if not relocated.is_file() or sha(relocated)!=expected:raise ValueError('relocated protected input drift: '+str(relocated))
            hashes[str(relocated)]=expected
    for rel in SCIENTIFIC_FILES:
        original=old_root/rel
        if sha(original)!=sha(ROOT/rel):raise ValueError('scientific source differs from committed original: '+rel)
        if str(original.resolve()) not in hashes or hashes[str(original.resolve())]!=sha(original):raise ValueError('missing original scientific SHA: '+rel)
        hashes[str(ROOT/rel)]=sha(ROOT/rel)
    retained={}
    for seed,panel in ((547,source/'s547'),(548,audited548)):
        r=read(panel/'results.json');audit=read(panel/'panel_audit.json');rows=read(panel/'rows.json')
        if r['run_status']!='COMPLETED' or r['continuous_updates']!=seed-547 or r['deterministic'] or audit['run_status']!='COMPLETED' or len(rows)!=768 or any(q['seed']!=seed for q in rows):raise ValueError('retained full panel/schema/behavior audit')
        if not audit.get('causal_inputs_request_likelihood_target_projection_and_PD_verified') or not audit.get('all_cohorts_included') or audit.get('rows')!=768:raise ValueError('complete independent native audit required')
        if [q['environment'] for q in rows]!=list(range(768)):raise ValueError('complete retained row identity')
        if sha(panel/'initial.pt')!=r['initial_sha256'] or sha(panel/'trace.pt')!=r['trace_sha256'] or sha(panel/'physical_metadata.json')!=r['physical_metadata_sha256']:raise ValueError('retained native panel drift')
        if seed==548:
            for f in ('initial.pt','trace.pt','physical_metadata.json','results.json'):
                if sha(panel/f)!=sha(source/'s548'/f):raise ValueError('548audit is not the same recorded physical panel')
        names=('initial.pt','trace.pt','physical_metadata.json','results.json','rows.json','panel_audit.json')
        retained[f's{seed}']={name:panel/name for name in names}
    retained['u00']={f:source/'u00'/f for f in ('policy_heads.pt','results.json')}
    retained['u01']={f:source/'u01'/f for f in ('policy_heads.pt','update_packet.pt','results.json','gradient_audit.json')}
    ur=read(source/'u01/results.json');ga=read(source/'u01/gradient_audit.json')
    if ur['update']!=1 or ga['update']!=1 or ga['run_status']!='COMPLETED':raise ValueError('retained u01audit')
    if sha(source/'u01/policy_heads.pt')!=ur['checkpoint_sha256']:raise ValueError('retained checkpoint drift')
    for seed,u in ((547,'u00'),(548,'u01')):
        if read(retained[f's{seed}']['results.json'])['continuous_checkpoint_sha256']!=sha(retained[u]['policy_heads.pt']):raise ValueError('retained behavior/checkpoint mismatch')
    for packet in retained.values():
        for path in packet.values():hashes[str(path)]=sha(path)
    hashes[str(source/'run_manifest.json')]=pinned
    assert_source_stable(source,pinned)
    original_native=[p for p in m['phases'] if p['name']=='s547']
    if len(original_native)!=1:raise ValueError('native command template')
    return dict(source=source,audited548=audited548,manifest=m,pinned=pinned,hashes=hashes,retained=retained,template=original_native[0]['command'],original_bytes=bytes_in(source),order=resume_order())

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--audited548',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--inspect-only',action='store_true');p.add_argument('--report',type=Path);p.add_argument('--stop-receipt',type=Path);p.add_argument('--gpu',type=int,default=1);a=p.parse_args()
    plan=prepare_plan(a.source,a.audited548)
    if a.inspect_only:
        report=dict(run_status='COMPLETED',engineering_only=True,no_new_training_or_physics=True,
            original_source=str(plan['source']),original_manifest_sha256=plan['pinned'],
            source_protected_inputs_verified=len(plan['hashes']),scientific_sources_identical=True,
            retained_native_seeds=[547,548],retained_updates=[0,1],next_update=2,
            full_training_seeds=list(range(547,567)),final_only_evaluation=[568,569],
            remaining_phase_order=plan['order'],original_reserved_seconds=OLD_RESERVE_SECONDS,
            new_wall_limit_seconds=3600-OLD_RESERVE_SECONDS,original_bytes=plan['original_bytes'],
            combined_storage_limit_bytes=6<<30,gpu_required=True,stop_receipt_required=True,
            no_policy_utility_result=True)
        if a.report:
            if ROOT not in a.report.resolve().parents:raise ValueError('own report location')
            write_new(a.report,report)
        print(json.dumps(report));return
    if not a.output or not a.stop_receipt:raise ValueError('unique output and actual ORIGINAL-host stop receipt required')
    out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('new real output directory required')
    gpu=admission(a.gpu)
    receipt=verify_stop_receipt(a.stop_receipt,plan['source'],plan['manifest'],gpu,
        Path('/proc/sys/kernel/random/boot_id').read_text().strip(),os.readlink('/proc/self/ns/pid'),lambda pid:Path(f'/proc/{pid}').exists())
    # Only now may the continuation create outputs/launch children.
    import shutil
    out.mkdir();begin=time.monotonic();limit=3600-OLD_RESERVE_SECONDS
    hashes=plan['hashes'];hashes[str(Path(__file__).resolve())]=sha(Path(__file__))
    hashes[str(a.stop_receipt.resolve())]=sha(a.stop_receipt)
    hashes[str(ROOT/'docs/decisions/D-20261002-continuous-critic-resumption.md')]=sha(ROOT/'docs/decisions/D-20261002-continuous-critic-resumption.md')
    m=copy.deepcopy(plan['manifest']);m.update(run_id=out.name,run_status='RUNNING',pid=os.getpid(),
        isolated_worktree=str(ROOT),phases=[],input_sha256=hashes,panel_checkpoints={},
        source_run=str(plan['source']),source_manifest_sha256=plan['pinned'],source_stop_receipt=receipt,
        original_reserved_seconds=OLD_RESERVE_SECONDS,original_bytes=plan['original_bytes'],
        wall_limit_seconds=limit,total_wall_limit_seconds=3600,storage_limit_bytes=6<<30,
        scientific_git_commit=plan['manifest']['git_commit'],resume_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():
        m['wall_seconds']=time.monotonic()-begin;m['accounted_total_wall_seconds']=OLD_RESERVE_SECONDS+m['wall_seconds']
        (out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def check(full=True):
        assert_source_stable(plan['source'],plan['pinned'])
        if time.monotonic()-begin>limit:raise TimeoutError('original whole-run budget')
        if plan['original_bytes']+bytes_in(out)>6<<30:raise RuntimeError('original + continuation storage budget')
        if full and any(sha(Path(f))!=h for f,h in hashes.items()):raise ValueError('protected inputs changed')
    def protect(directory,names):
        for name in names:
            path=directory/name;hashes[str(path)]=sha(path)
        save()
    def execute(name,command,timeout,gpu_compute=True):
        check();admitted=admission(a.gpu) if gpu_compute else None
        if admitted and admitted['uuid']!=EXPECTED_GPU:raise ValueError('frozen GPU UUID')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED_GPU if gpu_compute else '',
            LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,run_status='RUNNING',admission=admitted,
            device_reason='GPU original simulation/inference/optimization' if gpu_compute else 'independent NumPy/physical/statistical audit; no fitting')
        m['phases'].append(phase);save();start=time.monotonic();print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        try:
            run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),lambda:check(full=False),min(timeout,max(.1,limit-(time.monotonic()-begin))),spawned)
            check();phase['run_status']='COMPLETED'
        except BaseException as e:
            phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(name=name,status='COMPLETED',wall_seconds=phase['wall_seconds'])),flush=True)
    save()
    try:
        for name,files in plan['retained'].items():
            d=out/name;d.mkdir()
            for filename,path in files.items():shutil.copy2(path,d/filename)
            protect(d,files)
        for seed,u in ((547,'u00'),(548,'u01')):
            heads=out/u/'policy_heads.pt';m['panel_checkpoints'][str(seed)]=dict(path=str(heads),sha256=sha(heads),update=seed-547,retained=True)
        heads=out/'u01/policy_heads.pt';save()
        for kind,update,seed in plan['order']:
            if kind in ('collect','evaluate'):
                m['panel_checkpoints'][str(seed)]=dict(path=str(heads),sha256=sha(heads),update=update);save()
                d=out/f's{seed}';execute(d.name,rebound_native_command(plan['template'],out,seed,heads),300)
                protect(d,('initial.pt','trace.pt','physical_metadata.json','results.json'))
                execute(d.name+'_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_continuous_critic_panel.py'),'--directory',str(out),'--panel',str(seed)],300,False)
                protect(d,('rows.json','panel_audit.json'))
            elif kind=='update':
                d=out/f'u{update:02d}';execute(d.name,[PYTHON,'-u',str(ROOT/'scripts/update_continuous_critic_policy.py'),'--output',str(d),'--previous',str(heads),'--panel',str(out/f's{seed}')],180)
                protect(d,('policy_heads.pt','update_packet.pt','results.json'))
                execute(d.name+'_gradient_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_continuous_critic_update.py'),'--directory',str(d)],180,False)
                protect(d,('gradient_audit.json',));heads=d/'policy_heads.pt'
            else:
                m.update(run_status='COLLECTION_COMPLETED',final_checkpoint_sha256=sha(heads));save()
                execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_continuous_critic_policy.py'),'--directory',str(out)],300,False)
        check();m.update(run_status='COMPLETED',label=read(out/'results.json')['label'],inputs_unchanged=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
