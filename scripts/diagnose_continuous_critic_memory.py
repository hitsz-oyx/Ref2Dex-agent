"""One bounded engineering reproduction; never feed these rows to training."""
import argparse,json,os,runpy,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import admission,sha,PYTHON
from scripts.resume_continuous_critic_policy import EXPECTED_GPU

def hook(command,report):
    target=Path(command[2]).resolve();observations=[]
    def trace(frame,event,arg):
        if Path(frame.f_code.co_filename)!=target:return None
        if event=='line' and frame.f_lineno==169 and frame.f_code.co_name=='run':
            torch=sys.modules['torch'];tick=frame.f_locals['tick'];task=frame.f_locals['task']
            free,total=torch.cuda.mem_get_info()
            rec=dict(tick=tick,free_bytes=free,total_bytes=total,
                torch_allocated_bytes=torch.cuda.memory_allocated(),
                torch_reserved_bytes=torch.cuda.memory_reserved(),
                goal_finite=bool(torch.isfinite(frame.f_locals['goal']).all()),
                current_q_finite=bool(torch.isfinite(task._dof_pos).all()),
                object_root_finite=bool(torch.isfinite(task._target_states).all()))
            observations.append(rec)
            with report.open('a') as f:f.write(json.dumps(rec)+'\n')
            if tick==0 or free<(256<<20):print('[DEBUG-memory] '+json.dumps(rec),flush=True)
        return trace
    sys.argv=command[2:];sys.settrace(trace)
    try:runpy.run_path(str(target),run_name='__main__')
    finally:sys.settrace(None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--hook-command',type=Path);p.add_argument('--hook-report',type=Path);a=p.parse_args()
    if a.hook_command:
        hook(json.loads(a.hook_command.read_text()),a.hook_report);return
    source=a.source.resolve();out=a.output.resolve();m=json.loads((source/'run_manifest.json').read_text())
    if m['run_status']!='FAILED' or m['phases'][-1]['name']!='s550' or (source/'s550/results.json').exists():raise ValueError('exact failed incomplete panel only')
    if out.exists() or ROOT not in out.parents:raise ValueError('unique own diagnosis directory')
    for path,h in m['input_sha256'].items():
        if sha(Path(path))!=h:raise ValueError('protected source changed: '+path)
    gpu=admission(1)
    if gpu['uuid']!=EXPECTED_GPU:raise ValueError('fixed GPU')
    command=list(m['phases'][-1]['command']);d=out/'native'
    for flag,value in [('--run-dir',d),('--output',d/'unused.json'),('--output_path',d/'player')]:command[command.index(flag)+1]=str(value)
    out.mkdir();(out/'native-command.json').write_text(json.dumps(command,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=EXPECTED_GPU,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TORCH_EXTENSIONS_DIR=str(source/'cache/torch_extensions'),XDG_CACHE_HOME=str(source/'cache'))
    env['LD_LIBRARY_PATH']='/home2/wyy/miniconda3/envs/graspenv/lib:'+env.get('LD_LIBRARY_PATH','')
    begin=time.monotonic();observed=[];child=None;reason=None
    try:
        with (out/'native.log').open('x') as log:
            child=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'--hook-command',str(out/'native-command.json'),'--hook-report',str(out/'memory.jsonl')],cwd=ROOT/'third_party/DExplore',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if time.monotonic()-begin>120:raise TimeoutError('single diagnosis budget')
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
                mine=[]
                for line in apps.splitlines():
                    values=[q.strip() for q in line.split(',')]
                    if len(values)==3 and values[1]==str(child.pid):mine.append(dict(uuid=values[0],memory=values[2]))
                observed.append(dict(seconds=time.monotonic()-begin,own_gpu_records=mine))
                if any(x['uuid']!=EXPECTED_GPU for x in mine):raise RuntimeError('owned process on unadmitted GPU; abort owned process only')
                try:child.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
    except BaseException as e:
        reason=repr(e)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
    text=(out/'native.log').read_text()
    result=dict(run_status='COMPLETED' if reason is None else 'FAILED',engineering_only=True,
        no_optimizer_updates=True,never_training_or_evaluation_data=True,source_manifest_sha256=sha(source/'run_manifest.json'),
        original_failed_seed=550,retained_update=3,admission=gpu,own_gpu_observations=observed,
        exact_symptom_reproduced='PxgCudaDeviceMemoryAllocator fail to allocate memory 67108864 bytes' in text,
        illegal_address_reproduced='illegal memory access' in text,child_exit_code=child.returncode if child else None,
        supervisor_error=reason,wall_seconds=time.monotonic()-begin)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='own_gpu_observations'}),flush=True)

if __name__=='__main__':main()
