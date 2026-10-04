"""Continue an audited u03 prefix on an admitted identical-model GPU.

Scientific files and all valid panels remain unchanged. The incomplete OOM
panel is preserved in its old directory and never supplied to an optimizer.
"""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import admission,sha,PYTHON
from scripts.resume_continuous_critic_policy import run_owned_child,bytes_in,assert_source_stable,rebound_native_command,SCIENTIFIC_FILES
GPU_INDEX=4
GPU_UUID='GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307'
PRIOR_SECONDS=650

def read(p):return json.loads(Path(p).read_text())
def remaining_order():
    out=[]
    for seed in range(550,567):out.extend([('collect',seed-547,seed),('update',seed-546,seed)])
    return out+[('evaluate',20,568),('evaluate',20,569),('analyze',20,None)]
def verify_gpu(gpu):
    if gpu['index']!=GPU_INDEX or gpu['uuid']!=GPU_UUID:raise ValueError('exact admitted migration GPU')
def source_plan(source,diagnosis):
    source=Path(source).resolve();diagnosis=Path(diagnosis).resolve();m=read(source/'run_manifest.json')
    if m['run_status']!='FAILED' or m['phases'][-1]['name']!='s550' or (source/'s550/results.json').exists() or (source/'u04').exists():raise ValueError('exact incomplete resource-failure prefix only')
    if m['training_seeds']!=list(range(547,567)) or m['evaluation_seeds']!=[568,569]:raise ValueError('fixed cohorts')
    original=Path(m['source_run']).resolve();pins={str(source):sha(source/'run_manifest.json'),str(original):sha(original/'run_manifest.json')}
    if pins[str(original)]!=m['source_manifest_sha256']:raise ValueError('original source progressed')
    hashes=dict(m['input_sha256'])
    for filename,expected in hashes.items():
        if sha(Path(filename))!=expected:raise ValueError('protected input drift: '+filename)
    original_root=Path(read(original/'run_manifest.json')['isolated_worktree'])
    for name in SCIENTIFIC_FILES:
        if sha(ROOT/name)!=sha(original_root/name):raise ValueError('scientific implementation changed: '+name)
    diagnosis_result=read(diagnosis/'results.json')
    if diagnosis_result['source_manifest_sha256']!=pins[str(source)] or not diagnosis_result['exact_symptom_reproduced'] or not diagnosis_result['never_training_or_evaluation_data']:raise ValueError('actual pinned diagnosis required')
    retained=[]
    for seed in range(547,550):
        d=source/f's{seed}';r=read(d/'results.json');a=read(d/'panel_audit.json');rows=read(d/'rows.json')
        if r['run_status']!='COMPLETED' or r['continuous_updates']!=seed-547 or r['deterministic'] or a['run_status']!='COMPLETED' or a['rows']!=768 or len(rows)!=768 or [q['environment'] for q in rows]!=list(range(768)) or any(q['seed']!=seed for q in rows):raise ValueError('complete retained native prefix')
        if any(sha(d/f)!=r[k] for f,k in [('initial.pt','initial_sha256'),('trace.pt','trace_sha256'),('physical_metadata.json','physical_metadata_sha256')]):raise ValueError('native prefix bytes')
        if r['continuous_checkpoint_sha256']!=sha(source/f'u{seed-547:02d}/policy_heads.pt'):raise ValueError('prefix behavior identity')
        retained.append(d)
    for update in range(4):
        d=source/f'u{update:02d}';r=read(d/'results.json')
        if update and (r['update']!=update or read(d/'gradient_audit.json')['run_status']!='COMPLETED' or r['checkpoint_sha256']!=sha(d/'policy_heads.pt')):raise ValueError('audited update prefix')
        retained.append(d)
    if sha(source/'u03/policy_heads.pt')!=m['panel_checkpoints']['550']['sha256']:raise ValueError('last fixed u03 behavior')
    for d in retained:
        for f in d.iterdir():
            if f.is_file():hashes[str(f.resolve())]=sha(f)
    for d,h in pins.items():hashes[str(Path(d)/'run_manifest.json')]=h
    for name in ['results.json','driver_allocation_check.json']:
        path=diagnosis/name;hashes[str(path)]=sha(path)
    return dict(source=source,original=original,diagnosis=diagnosis,manifest=m,pins=pins,hashes=hashes,retained=retained,
        previous_bytes=sum(bytes_in(d) for d in [source,original,diagnosis]),template=m['phases'][-1]['command'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect-only',action='store_true');a=p.parse_args()
    plan=source_plan(a.source,a.diagnosis);out=a.output.resolve()
    if out.exists() or ROOT not in out.parents:raise ValueError('unique real own output')
    gpu=admission(GPU_INDEX);verify_gpu(gpu)
    model=subprocess.check_output(['nvidia-smi','-i',str(GPU_INDEX),'--query-gpu=name','--format=csv,noheader'],text=True).strip()
    if model!='NVIDIA GeForce RTX 3090':raise ValueError('same hardware model required')
    if Path('/proc/1/comm').read_text().strip()!='systemd':raise ValueError('actual host observation required')
    for m in (plan['manifest'],read(plan['original']/'run_manifest.json')):
        ids=[m['pid']]+[q['pid'] for q in m['phases'] if q.get('pid') and q.get('run_status') in ('RUNNING','FAILED')]
        if any(Path('/proc/'+str(pid)).exists() for pid in ids):raise ValueError('source/reused PID visible; inspect before duplication')
    out.mkdir();begin=time.monotonic();hashes=plan['hashes'];limit=3600-PRIOR_SECONDS
    for f in [Path(__file__),ROOT/'docs/archive/2026-10-04-research-governance/decisions/D-20261002-continuous-gpu-migration.md']:hashes[str(f.resolve())]=sha(f)
    if a.inspect_only:
        report=dict(run_status='COMPLETED',engineering_only=True,protected_paths_verified=len(hashes),retained_updates=[0,1,2,3],retained_panels=[547,548,549],remaining_order=remaining_order(),gpu=gpu,same_hardware_model=model,previous_bytes=plan['previous_bytes'],prior_reserved_seconds=PRIOR_SECONDS,new_limit_seconds=limit,no_new_training_or_physics=True)
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    m=copy.deepcopy(plan['manifest']);m.update(run_id=out.name,run_status='RUNNING',pid=os.getpid(),isolated_worktree=str(ROOT),phases=[],input_sha256=hashes,
        resume_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scientific_git_commit=plan['manifest']['scientific_git_commit'],
        source_run=str(plan['source']),original_source_run=str(plan['original']),source_manifest_sha256=plan['pins'][str(plan['source'])],
        runtime_migration=dict(original_gpu_uuid=plan['manifest']['gpu_uuid'] if 'gpu_uuid' in plan['manifest'] else 'GPU-33a9c1c9-cccb-61c2-afeb-4ec98be09963',new_gpu_uuid=GPU_UUID,same_hardware_model=model,incomplete_panel_recollection_only=550,optimizer_retries=False),
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
        heads=out/'u03/policy_heads.pt';save()
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
