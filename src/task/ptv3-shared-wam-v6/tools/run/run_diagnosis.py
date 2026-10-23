"""One-shot ordered diagnosis driver; terminal failures, no automatic retries."""
import argparse,json,os,signal,subprocess,sys,time,shutil
from pathlib import Path
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2];sys.path.insert(0,str(TASK/'tools/run'))
from remote_staged_data import read,write,sha
D=Path('/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending');OUT=D/'diagnosis'
PY='/home/wbcd/miniconda3/envs/graspenv/bin/python'

def environment(gpus):
 return dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,CONDA_PREFIX='/home/wbcd/miniconda3/envs/graspenv',TMPDIR=str(ROOT/'tmp/rh/scratch'),REF2DEX_RUN_ROOT=str(ROOT/'tmp/rh'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',XDG_CACHE_HOME=str(D/'cache'),TORCH_HOME=str(D/'cache/torch'),TORCH_EXTENSIONS_DIR=str(D/'cache/extensions'),TRITON_CACHE_DIR=str(D/'cache/triton'),CUMM_CACHE_PATH=str(D/'cache/cumm'),CUDA_CACHE_PATH=str(D/'cache/cuda'),MPLCONFIGDIR=str(D/'cache/mpl'),CUBLAS_WORKSPACE_CONFIG=':4096:8',NCCL_ASYNC_ERROR_HANDLING='1')

def launch(arm,gpus):
 folder=OUT/arm;r=read(folder/'recipe-v2.json');assert not (folder/'launch-receipt.json').exists()
 command=[PY,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node='+str(r['world_size']),str(TASK/'tools/run/train_diagnosis.py'),'--recipe',str(folder/'recipe-v2.json'),'--output',str(folder)]
 with (folder/'console.log').open('x') as log:p=subprocess.Popen(command,cwd=ROOT,env=environment(gpus),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 write(folder/'launch-receipt.json',dict(command=command,pid=p.pid,pgid=p.pid,gpus=gpus,config_sha256=r['config_sha256'],data_identity_sha256=r.get('data_identity_sha256',r['manifest_sha256']),recipe_sha256=sha(folder/'recipe-v2.json'),git_commit=r['git_commit'],initialization_sha256=r['initialization_sha256'],automatic_restart=False,at=time.time()),immutable=True)
 return p

def stop(p):
 if p.poll() is None:
  assert os.getpgid(p.pid)==p.pid;os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()

def watch(active):
 begun=time.monotonic();pending=dict(active)
 while pending:
  snapshot=dict(at=time.time(),free_bytes=shutil.disk_usage(OUT).free,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True),arms={})
  for arm,p in list(pending.items()):
   folder=OUT/arm;progress=folder/'progress.json'
   if progress.exists():snapshot['arms'][arm]=read(progress)
   code=p.poll()
   if code is not None:
    if code!=0:
     write(folder/'terminal-failure.json',dict(returncode=code,automatic_restart=False,at=time.time()),immutable=True)
     with (OUT/'README.md').open('a') as f:f.write('\n'+arm+' FAILED exit '+str(code)+'; no restart. See '+str(folder/'console.log')+'\n')
     raise RuntimeError(arm+' failed')
    subprocess.run([PY,str(TASK/'tools/audit/report_diagnosis_arm.py'),arm],cwd=ROOT,env=environment(''),check=True)
    del pending[arm]
  write(OUT/'watchdog-latest.json',snapshot)
  with (OUT/'watchdog.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
  if snapshot['free_bytes']<30*2**30:raise RuntimeError('disk floor')
  if (OUT/'STOP').exists():raise RuntimeError('owner STOP')
  if time.monotonic()-begun>3300:raise TimeoutError('training40min plus bounded initialization/final evaluation overhead')
  if pending:time.sleep(30)

def main():
 p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['C1','C2','C3','C4-source1','C4-source10','C4-source0','C5']);a=p.parse_args()
 lock=OUT/'DIAGNOSIS_V2_ONE_SHOT.lock';fd=os.open(str(lock),os.O_EXCL|os.O_CREAT|os.O_WRONLY,0o664);os.write(fd,str(os.getpid()).encode());os.close(fd)
 active={}
 try:
  gpus=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True);assert not gpus.strip(),'GPU process preflight failed'
  arms=list(a.arms)
  while arms:
   arm=arms.pop(0)
   if arm=='C1' and read(OUT/'C1-support-match.json')['match_fraction']<.95:continue
   if arm=='C4-source1' and arms[:2]==['C4-source10','C4-source0']:
    arms=arms[2:];active={x:launch(x,str(g)) for x,g in [('C4-source1',0),('C4-source10',1),('C4-source0',2)]}
   else:
    if not (OUT/arm/'recipe-v2.json').exists():
     subprocess.run([PY,str(TASK/'tools/run/prepare_diagnosis_v2.py')],cwd=ROOT,env=environment(''),check=True)
    active={arm:launch(arm,'0,1,2,3' if not arm.startswith('C4') else '0')}
   watch(active);active={}
  write(OUT/'driver-result.json',dict(status='COMPLETED',automatic_restart=False))
 except BaseException as e:
  for proc in active.values():stop(proc)
  write(OUT/'driver-failure.json',dict(reason=repr(e),automatic_restart=False,at=time.time()),immutable=True);raise
if __name__=='__main__':main()
