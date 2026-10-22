"""Bounded real-physics action/PD/q/dq and indexed-reset acceptance. No learning."""
import argparse,json,os,socket,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
ACTIVE=(0,1,2,3,4,5,6,8,10,12,14,15)
COUPLING=((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8))

def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def couple(x):
 x=np.array(x,copy=True)
 for dst,src,ratio in COUPLING:x[...,dst]=x[...,src]*ratio
 return x
def expected_pd(action,q,offset,scale):
 mapped=action.copy();mapped[:,6:]=(1+mapped[:,6:])/2
 target=offset+scale*mapped;target[:,:6]+=q[:,:6]
 return couple(target)
def invert(target,q,offset,scale):
 value=(target-offset)/scale;value[:,:6]=(target[:,:6]-q[:,:6]-offset[:6])/scale[:6]
 value[:,6:]=2*value[:,6:]-1
 out=np.zeros_like(value);out[:,ACTIVE]=value[:,ACTIVE]
 if np.abs(out).max()>1+1e-6:raise ValueError('axis target outside canonical action bounds')
 return out.astype(np.float32)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=4);a=p.parse_args()
 if a.gpu!=4:raise ValueError('only physical GPU4 allocated')
 a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
 sources=['src/task/ptv3-shared-wam-v6/tools/audit/validate_native_action_reset.py','src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py','src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/continuous_wrist.py','src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/inspire_reset_submission.py','src/task/consequence-evaluator/src/consequence_evaluator/native_reset.py','src/task/consequence-evaluator/src/consequence_evaluator/reset_kinematics.py','src/task/trajectory-policy/src/trajectory_policy/pd_retargeter.py']
 source_index=[]
 for relative in sources:
  source=ROOT/relative;target=a.output/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);source_index.append(dict(path=relative,mtime_ns=source.stat().st_mtime_ns,size=source.stat().st_size))
 write(a.output/'source-index.json',source_index)
 private=ROOT/'tmp'/a.output.name;private.mkdir(parents=True,exist_ok=False)
 os.environ.update(CUDA_VISIBLE_DEVICES='',TMPDIR=str(private),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
 sys.path[:0]=[str(ROOT/'src/task/trajectory-policy/src'),str(ROOT/'src/task/wam-rl-policy-v3/src')]
 from trajectory_policy.wam_ipc import send,receive,peer_pid
 from wam_rl_policy_v3.inspire_state_codec import unpack_state
 panel=json.loads(a.panel.read_text());names=panel['sequences'][:2];fixed=private/'motions.json';write(fixed,dict(sequences=names))
 gpu=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 if int(gpu.split(',')[1])>64:raise RuntimeError('GPU4 occupied: '+gpu)
 sock=private/'rpc.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(180);server.bind(str(sock));server.listen(1)
 nativeout=ROOT/'outputs/wam-rl-policy-v3'/a.output.name
 command=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(ROOT/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'),'--cache',str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),'--output',str(nativeout),'--socket',str(sock),'--server-pid',str(os.getpid()),'--envs','2','--gpu','4','--seed','195','--seconds','540','--fixed-motions-json',str(fixed),'--state-transport','packed','--extended-state','--synchronize-setters']
 report=dict(status='STARTING',experiment_id='P-20261022-teacher-action-reset-validation',run_id=a.output.name,python=sys.executable,gpu_before=gpu,native_command=command,sequences=names,physics_controls=0,environment_controls=0,axis_cases=[],reset_checks=[],mutants={},policy_updates=0)
 write(a.output/'manifest.json',report);log=(a.output/'native.log').open('x');child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(a.output/'process.json',dict(pid=os.getpid(),native_pid=child.pid,command=command))
 connection=None;trace=[];snapshots={};started=time.monotonic()
 try:
  connection,_=server.accept();connection.settimeout(120)
  if peer_pid(connection)!=child.pid:raise ValueError('unexpected native peer')
  hello=receive(connection);state=unpack_state(hello['state']);offset=np.asarray(hello['audit']['controller']['offset'],np.float32);scale=np.asarray(hello['audit']['controller']['scale'],np.float32)
  report.update(status='RUNNING',native_initial_audit=hello['audit'],controller=dict(offset=offset.tolist(),scale=scale.tolist()))
  def snapshot(s):return {k:np.asarray(s[k]).copy() for k in ['q','dq','objects','hand','palm_pose','physical','tick','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','episodes']}
  def reset(ids,label):
   nonlocal state
   before=snapshot(state);send(connection,dict(op='reset',ids=ids));state=unpack_state(receive(connection)['state']);after=snapshot(state);untouched=[i for i in range(2) if i not in ids]
   checks={k:float(np.max(np.abs(after[k][untouched].astype(float)-before[k][untouched].astype(float)))) if untouched else 0. for k in before}
   valid=all(v==0 for v in checks.values()) and np.all(after['episodes'][ids]==before['episodes'][ids]+1) and np.all(after['tick'][ids]==0) and not after['contact_force_valid'][ids].any()
   report['reset_checks'].append(dict(label=label,ids=ids,untouched_max_abs=checks,passed=bool(valid)))
   for when,s in [('before',before),('after',after)]:
    for k,v in s.items():snapshots[label+'_'+when+'_'+k]=v
   if not valid:raise ValueError('indexed reset invariant failed: '+label)
  def step(action,label):
   nonlocal state
   if time.monotonic()-started>540 or report['physics_controls']>=500:raise TimeoutError('registered engineering budget')
   before=snapshot(state);want=expected_pd(action,before['q'],offset,scale)
   send(connection,dict(op='step',action=action.tolist(),episodes=list(state['episodes']),return_pd=True));packet=receive(connection);state=unpack_state(packet['state']);after=snapshot(state);pd=np.asarray(packet['pd_targets'],np.float32)
   error=float(np.abs(pd-want).max());same=np.array_equal(after['episodes'],before['episodes']) and np.array_equal(after['tick'],before['tick']+1)
   report['physics_controls']+=1;report['environment_controls']+=2
   trace.append(dict(q_before=before['q'],q_after=after['q'],dq_after=after['dq'],palm_pose_before=before['palm_pose'],palm_pose_after=after['palm_pose'],action=action,pd=pd,expected_pd=want,tick_before=before['tick'],tick_after=after['tick'],episodes=after['episodes']))
   if error>1e-6 or not same or not np.isfinite(after['q']).all() or not np.isfinite(after['dq']).all() or not after['contact_force_valid'].all():raise ValueError('native action/time/force invariant: '+label+' '+str(error))
   return error
  # A measured reset q can lie outside the controller's legal finger target
  # interval. Reset isolation needs a legal command, not inverse(q_reset).
  fixture_action=np.zeros((2,18),np.float32)
  for t in range(4):
   step(fixture_action,'reset-fixture')
   if t==0:
    from scipy.spatial.transform import Rotation
    last=trace[-1];angles=(Rotation.from_quat(last['palm_pose_before'][:,3:7]).inv()*Rotation.from_quat(last['palm_pose_after'][:,3:7])).magnitude();translation=np.linalg.norm(last['palm_pose_before'][:,:3]-last['palm_pose_after'][:,:3],axis=-1)
    report['neutral_first_tick_measured_palm_rotation_rad']=angles.tolist();report['neutral_first_tick_measured_palm_translation_m']=translation.tolist()
    if angles[1]>=.1:raise ValueError('production normalized cubemedium neutral first-tick rotation >=0.1rad')
   if t==1:reset([0],'step2_env0')
   if t==3:reset([1],'step4_env1')
  for axis in ACTIVE:
   reset([0,1],'axis%d_reset'%axis);q=np.asarray(state['q'],np.float32);target=q.copy();target[:,6:]=scale[6:]/2+offset[6:];target=couple(target)
   for _ in range(20):step(np.zeros((2,18),np.float32),'preposition')
   baseline=np.asarray(state['q'],np.float32).copy();target=couple(baseline)
   amplitude=.003 if axis<3 else .03 if axis<6 else .08
   signs=np.array([1.,-1.]);target[:,axis]+=signs*amplitude;target=couple(target);errors=[]
   for _ in range(12):errors.append(step(invert(target,np.asarray(state['q']),offset,scale),'axis%d'%axis))
   observed=np.asarray(state['q']);dq=np.asarray(state['dq']);signed=signs*(observed[:,axis]-baseline[:,axis]);fraction=signed/amplitude;residual=np.abs(observed[:,axis]-target[:,axis])/amplitude
   for env,sign in enumerate(signs):report['axis_cases'].append(dict(axis=axis,env=env,sign=int(sign),amplitude=amplitude,baseline_q=float(baseline[env,axis]),target_q=float(target[env,axis]),observed_q=float(observed[env,axis]),observed_dq=float(dq[env,axis]),signed_response_fraction=float(fraction[env]),target_error_fraction=float(residual[env]),maximum_pd_error=max(errors),passed=bool(fraction[env]>=.25 and residual[env]<=.75)))
   write(a.output/'progress.json',report)
   if not all(row['passed'] for row in report['axis_cases'][-2:]):raise ValueError('measured axis response failed: '+str(axis))
  pd=np.stack([r['pd'] for r in trace]);actions=np.stack([r['action'] for r in trace]);qs=np.stack([r['q_before'] for r in trace]);wrong=offset+scale*actions
  wrong[...,:6]+=qs[...,:6];wrong=couple(wrong)
  double=couple(pd.reshape(-1,18)).reshape(pd.shape)
  # The native fixed map is idempotent. Applying an additive distal update is
  # a detectable wrong coupling mutant; merely calling couple twice is not.
  additive=pd.copy()
  for dst,src,ratio in COUPLING:additive[...,dst]+=pd[...,src]*ratio
  missing_current=pd.copy();missing_current[...,:6]-=qs[...,:6]
  report['mutants']={name:dict(maximum_pd_difference=float(np.abs(value-pd).max()),killed=bool(np.abs(value-pd).max()>1e-6)) for name,value in [('wrong_finger_minus1_to1_mapping',wrong),('missing_current_q_wrist',missing_current),('additive_duplicate_coupling',additive)]}
  report['mutants']['pure_double_assignment_coupling']=dict(maximum_pd_difference=float(np.abs(double-pd).max()),killed=False,reason='fixed assignment coupling is mathematically idempotent; this mutant cannot be detected and is not claimed killed')
  send(connection,dict(op='close'));receive(connection);child.wait(timeout=20)
  if child.returncode:raise RuntimeError('native child exit '+str(child.returncode))
  report.update(status='COMPLETED',axis_cases_passed=sum(r['passed'] for r in report['axis_cases']),elapsed_s=time.monotonic()-started)
 except BaseException as e:
  report.update(status='FAILED',error=repr(e),elapsed_s=time.monotonic()-started)
  raise
 finally:
  if trace:np.savez_compressed(a.output/'trace.npz',**{k:np.stack([r[k] for r in trace]) for k in trace[0]})
  if snapshots:np.savez_compressed(a.output/'reset-snapshots.npz',**snapshots)
  write(a.output/'manifest.json',report)
  if connection:
   try:send(connection,dict(op='close'));receive(connection)
   except Exception:pass
   connection.close()
  server.close();log.close()
  if child.poll() is None:
   child.terminate()
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=='__main__':main()
