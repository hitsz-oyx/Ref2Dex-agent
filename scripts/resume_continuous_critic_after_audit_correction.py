"""Resume an exact u17 prefix after a separately witnessed ReLU audit correction.

All scientific files and scalar audit limits remain unchanged; retain failures.
"""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import admission,sha,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,assert_source_stable,rebound_native_command,SCIENTIFIC_FILES
GPU_INDEX=4
GPU_UUID='GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307'
PRIOR_SECONDS=2550

def read(p):return json.loads(Path(p).read_text())
def remaining_order():
    out=[]
    for seed in range(564,567):out.extend([('collect',seed-547,seed),('update',seed-546,seed)])
    return out+[('evaluate',20,568),('evaluate',20,569),('analyze',20,None)]
def verify_gpu(gpu):
    if gpu['index']!=GPU_INDEX or gpu['uuid']!=GPU_UUID:raise ValueError('exact admitted migration GPU')
def source_plan(source,correction):
    source=Path(source).resolve();correction=Path(correction).resolve();m=read(source/'run_manifest.json')
    if m['run_status']!='FAILED' or m['phases'][-1]['name']!='u17_gradient_audit' or (source/'u18').exists() or (source/'s564').exists():raise ValueError('exact diagnosed prefix required')
    if m['training_seeds']!=list(range(547,567)) or m['evaluation_seeds']!=[568,569]:raise ValueError('fixed cohorts')
    original=Path(m['original_source_run']).resolve();pins={str(source):sha(source/'run_manifest.json'),str(original):sha(original/'run_manifest.json')}
    hashes=dict(m['input_sha256'])
    for filename,expected in hashes.items():
        if sha(Path(filename))!=expected:raise ValueError('protected source drift: '+filename)
    original_root=Path(read(original/'run_manifest.json')['isolated_worktree'])
    if any(sha(ROOT/f)!=sha(original_root/f) for f in SCIENTIFIC_FILES):raise ValueError('scientific implementation changed')
    audit=read(correction/'gradient_audit.json')
    if audit['run_status']!='COMPLETED' or audit['update']!=17 or not audit['original_audit_retained_failed'] or not audit['gpu_saved_gradients_exactly_reproduced'] or not audit['no_optimizer_or_physics_updates']:raise ValueError('witnessed correction required')
    expected_limits=dict(input=0.,advantage=2e-5,forward=2e-5,loss=2e-5,gradient=2e-5,adam=2e-5,norm_relative=2e-6)
    if audit['scalar_tolerances_unchanged']!=expected_limits or any(audit['maximum_error'][k]>v for k,v in expected_limits.items()):raise ValueError('unchanged original limits')
    for filename,expected in {**audit['source_sha256'],**audit['diagnosis_sha256']}.items():
        if sha(Path(filename))!=expected:raise ValueError('correction source drift')
        hashes[filename]=expected
    hashes[str(correction/'gradient_audit.json')]=sha(correction/'gradient_audit.json')
    if (source/'u17/gradient_audit.json').exists():raise ValueError('do not replace failed original audit')
    retained=[]
    for seed in range(547,564):
        d=source/f's{seed}';q=read(d/'results.json');rows=read(d/'rows.json');a=read(d/'panel_audit.json')
        if q['run_status']!='COMPLETED' or q['continuous_updates']!=seed-547 or q['deterministic'] or a['run_status']!='COMPLETED' or a['rows']!=768 or len(rows)!=768 or [v['environment'] for v in rows]!=list(range(768)) or any(v['seed']!=seed for v in rows):raise ValueError('retained full native panel')
        if any(sha(d/f)!=q[k] for f,k in [('initial.pt','initial_sha256'),('trace.pt','trace_sha256'),('physical_metadata.json','physical_metadata_sha256')]) or q['continuous_checkpoint_sha256']!=sha(source/f'u{seed-547:02d}/policy_heads.pt'):raise ValueError('retained raw behavior hashes')
        retained.append(d)
    for update in range(18):
        d=source/f'u{update:02d}';q=read(d/'results.json')
        if update and (q['update']!=update or q['checkpoint_sha256']!=sha(d/'policy_heads.pt')):raise ValueError('retained actual checkpoint')
        if 0<update<17 and read(d/'gradient_audit.json')['run_status']!='COMPLETED':raise ValueError('retained original audit')
        if update<17:retained.append(d)
        for f in d.iterdir():
            if f.is_file():hashes[str(f.resolve())]=sha(f)
    for d,h in pins.items():hashes[str(Path(d)/'run_manifest.json')]=h
    diagnosis_file=next(iter(audit['diagnosis_sha256']));diagnosis_dir=Path(diagnosis_file).parent
    return dict(source=source,original=original,diagnosis=correction,manifest=m,pins=pins,hashes=hashes,retained=retained,
        previous_bytes=m['previous_bytes']+bytes_in(source)+bytes_in(correction)+bytes_in(diagnosis_dir),
        template=next(q['command'] for q in m['phases'] if q['name']=='s563'),correction=correction)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--correction',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect-only',action='store_true');a=p.parse_args()
    plan=source_plan(a.source,a.correction);out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('unique real own output')
    gpu=admission(GPU_INDEX);verify_gpu(gpu)
    model=subprocess.check_output(['nvidia-smi','-i',str(GPU_INDEX),'--query-gpu=name','--format=csv,noheader'],text=True).strip()
    if model!='NVIDIA GeForce RTX 3090':raise ValueError('same hardware model required')
    if Path('/proc/1/comm').read_text().strip()!='systemd':raise ValueError('actual host observation required')
    for m in (plan['manifest'],read(plan['original']/'run_manifest.json')):
        ids=[m['pid']]+[q['pid'] for q in m['phases'] if q.get('pid') and q.get('run_status') in ('RUNNING','FAILED')]
        if any(Path('/proc/'+str(pid)).exists() for pid in ids):raise ValueError('source/reused PID visible; inspect before duplication')
    out.mkdir();begin=time.monotonic();hashes=plan['hashes'];limit=3600-PRIOR_SECONDS
    for f in [Path(__file__),ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261002-continuous-gradient-branch-correction.md']:hashes[str(f.resolve())]=sha(f)
    if a.inspect_only:
        report=dict(run_status='COMPLETED',engineering_only=True,protected_paths_verified=len(hashes),retained_updates=list(range(18)),retained_panels=list(range(547,564)),remaining_order=remaining_order(),gpu=gpu,same_hardware_model=model,previous_bytes=plan['previous_bytes'],prior_reserved_seconds=PRIOR_SECONDS,new_limit_seconds=limit,no_new_training_or_physics=True)
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    m=copy.deepcopy(plan['manifest']);m.update(run_id=out.name,run_status='RUNNING',pid=os.getpid(),isolated_worktree=str(ROOT),phases=[],input_sha256=hashes,
        resume_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scientific_git_commit=plan['manifest']['scientific_git_commit'],
        source_run=str(plan['source']),original_source_run=str(plan['original']),source_manifest_sha256=plan['pins'][str(plan['source'])],
        runtime_migration=plan['manifest']['runtime_migration'],gpu_index=GPU_INDEX,gpu_uuid=GPU_UUID,audit_correction=dict(update=17,source=str(plan['correction']),sha256=sha(plan['correction']/'gradient_audit.json'),original_failed_parent_retained=str(plan['source']),scalar_tolerances_unchanged=True,no_optimizer_retry=True),
        original_reserved_seconds=PRIOR_SECONDS,previous_bytes=plan['previous_bytes'],wall_limit_seconds=limit,total_wall_limit_seconds=3600,storage_limit_bytes=6<<30)
    def save():
        m['wall_seconds']=time.monotonic()-begin;m['accounted_total_wall_seconds']=PRIOR_SECONDS+m['wall_seconds']
        (out/'run_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    def check(full=True):
        for d,h in plan['pins'].items():assert_source_stable(d,h)
        if time.monotonic()-begin>limit:raise TimeoutError('accumulated execution budget')
        if plan['previous_bytes']+bytes_in(out)>6<<30:raise RuntimeError('all original + resumed + diagnosis bytes budget')
        if full and any(sha(Path(f))!=h for f,h in hashes.items()):raise ValueError('protected input drift')
    def protect(d,names):
        for name in names:hashes[str((d/name).resolve())]=sha(d/name)
        save()
    def execute(name,command,timeout,gpu_compute=True):
        check();admitted=admission(GPU_INDEX) if gpu_compute else None
        if admitted:verify_gpu(admitted)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID if gpu_compute else '',LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(out/'cache/torch_extensions'),XDG_CACHE_HOME=str(out/'cache'))
        env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
        phase=dict(name=name,command=command,run_status='RUNNING',admission=admitted,device_reason='same-model admitted GPU execution' if gpu_compute else 'independent NumPy physical/gradient/statistical audit, no fitting');m['phases'].append(phase);save();start=time.monotonic()
        print(json.dumps(dict(name=name,status='STARTED')),flush=True)
        def spawned(child):phase.update(pid=child.pid,pgid=child.pid);save()
        def guard():
            check(False)
            if gpu_compute:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
                for line in apps.splitlines():
                    fields=[q.strip() for q in line.split(',')]
                    if len(fields)==3 and fields[0]==GPU_UUID and fields[1]!=str(phase.get('pid')):raise RuntimeError('new GPU contention; stop only our owned child')
        try:
            run_owned_child(command,ROOT/'third_party/DExplore',env,out/(name+'.log'),guard,min(timeout,max(.1,limit-(time.monotonic()-begin))),spawned)
            check();phase['run_status']='COMPLETED'
        except BaseException as e:phase.update(run_status='FAILED',error=repr(e));raise
        finally:phase['wall_seconds']=time.monotonic()-start;save()
        print(json.dumps(dict(name=name,status='COMPLETED',wall_seconds=phase['wall_seconds'])),flush=True)
    save()
    try:
        for d in plan['retained']:(out/d.name).symlink_to(d,target_is_directory=True)
        composite=out/'u17';composite.mkdir()
        for f in (plan['source']/'u17').iterdir():
            if f.is_file():(composite/f.name).symlink_to(f.resolve())
        (composite/'gradient_audit.json').symlink_to((plan['correction']/'gradient_audit.json').resolve())
        heads=composite/'policy_heads.pt';m.pop('error',None);save()
        for kind,update,seed in remaining_order():
            if kind in ('collect','evaluate'):
                m['panel_checkpoints'][str(seed)]=dict(path=str(heads),sha256=sha(heads),update=update);save();d=out/f's{seed}'
                execute(d.name,rebound_native_command(plan['template'],out,seed,heads),300)
                protect(d,['initial.pt','trace.pt','physical_metadata.json','results.json'])
                execute(d.name+'_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_continuous_critic_panel.py'),'--directory',str(out),'--panel',str(seed)],300,False)
                protect(d,['rows.json','panel_audit.json'])
            elif kind=='update':
                d=out/f'u{update:02d}';execute(d.name,[PYTHON,'-u',str(ROOT/'scripts/update_continuous_critic_policy.py'),'--output',str(d),'--previous',str(heads),'--panel',str(out/f's{seed}')],180)
                protect(d,['policy_heads.pt','update_packet.pt','results.json'])
                execute(d.name+'_gradient_audit',[PYTHON,'-u',str(ROOT/'scripts/audit_continuous_critic_update.py'),'--directory',str(d)],180,False)
                protect(d,['gradient_audit.json']);heads=d/'policy_heads.pt'
            else:
                m.update(run_status='COLLECTION_COMPLETED',final_checkpoint_sha256=sha(heads));save()
                execute('analysis',[PYTHON,'-u',str(ROOT/'scripts/analyze_continuous_critic_policy.py'),'--directory',str(out)],300,False)
        check();m.update(run_status='COMPLETED',label=read(out/'results.json')['label'],inputs_unchanged=True)
    except BaseException as e:m.update(run_status='FAILED',error=repr(e));raise
    finally:save()

if __name__=='__main__':main()
