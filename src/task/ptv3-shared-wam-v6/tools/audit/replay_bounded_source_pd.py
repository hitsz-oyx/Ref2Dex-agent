"""Bounded source-PD upper control using freshly captured official targets.

The replay uses target[t], live measured q[t] inverse, then native PD. It never
uses a learned executor, predicted tau, teacher future observation or reward.
Raw physical evidence is stored for both frozen task scorers offline.
"""
import argparse,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
ACTIVE=(0,1,2,3,4,5,6,8,10,12,14,15)
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=4);p.add_argument('--sequence');p.add_argument('--branch-aware',action='store_true');p.add_argument('--neutral-controls',type=int,choices=[4]);a=p.parse_args()
 if a.gpu!=4:raise ValueError('GPU4 only')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);private=ROOT/'tmp'/out.name;private.mkdir(parents=True,exist_ok=False)
 os.environ.update(CUDA_VISIBLE_DEVICES='',TMPDIR=str(private),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
 sys.path[:0]=[str(ROOT/'src/task/trajectory-policy/src'),str(ROOT/'src/task/wam-rl-policy-v3/src')]
 from trajectory_policy.wam_ipc import send,receive,peer_pid
 from wam_rl_policy_v3.inspire_state_codec import unpack_state
 from scipy.spatial.transform import Rotation
 teacher=json.loads((a.teacher/'manifest.json').read_text())
 if teacher['status']!='COMPLETED' or len(teacher['records'])!=6:raise ValueError('fresh completed fixed six source required')
 if a.branch_aware and a.sequence!='s7_cubemedium_pass_1':raise ValueError('registered branch probe is cubemedium only')
 records=[r for r in teacher['records'] if not a.sequence or r['sequence']==a.sequence]
 if not records:raise ValueError('sequence missing')
 names=[r['sequence'] for r in records];sources=[]
 for r in records:
  with np.load(a.teacher/r['path']/'trajectory.npz') as z:sources.append({k:z[k].copy() for k in ('q','dq','hand','obj','pd_targets')})
 lengths=[len(s['pd_targets']) for s in sources];controls=600 if a.branch_aware else min(600,max(lengths));n=len(names)
 if a.neutral_controls:
  if not a.branch_aware or n!=1:raise ValueError('only registered normalized single-case neutral diagnostic')
  controls=a.neutral_controls
 gpu=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 if int(gpu.split(',')[1])>64:raise RuntimeError('GPU4 occupied '+gpu)
 fixed=private/'motions.json';write(fixed,dict(sequences=names));sock=private/'rpc.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(180);server.bind(str(sock));server.listen(1)
 nativeout=ROOT/'outputs/wam-rl-policy-v3'/out.name
 cmd=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(ROOT/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'),'--cache',str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),'--output',str(nativeout),'--socket',str(sock),'--server-pid',str(os.getpid()),'--envs','6','--gpu','4','--seed','195','--seconds','540','--fixed-motions-json',str(fixed),'--state-transport','packed','--extended-state','--synchronize-setters']
 cmd[cmd.index('--envs')+1]=str(n)
 if a.branch_aware:cmd[2]=str(ROOT/'src/task/ptv3-shared-wam-v6/tools/audit/inspire_branch_reset_host.py')
 report=dict(status='STARTING',teacher=str(a.teacher.resolve()),source_lengths=lengths,controls=controls,native_command=cmd,policy_updates=0,gpu_before=gpu,physics_controls=0,environment_controls=0,maximum_pd_inverse_error=0.,clipped_envcontrols=0,sequence_order=names,reference_end_policy='hold final recorded PD target; post-reference suffix reported separately; not observed teacher continuation')
 write(out/'manifest.json',report);log=(out/'native.log').open('x');child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(out/'process.json',dict(client_pid=os.getpid(),native_pid=child.pid));connection=None;trace=[];applied=[];start=time.monotonic()
 report['branch_aware']=a.branch_aware
 report['neutral_controls']=a.neutral_controls
 try:
  connection,_=server.accept();connection.settimeout(120)
  if peer_pid(connection)!=child.pid:raise ValueError('native pid mismatch')
  hello=receive(connection);state=unpack_state(hello['state']);offset=np.asarray(hello['audit']['controller']['offset'],np.float32);scale=np.asarray(hello['audit']['controller']['scale'],np.float32);report['native_initial_audit']=hello['audit']
  if a.branch_aware and any(hello['audit']['native_dof_properties']['hasLimits'][3:6]):raise ValueError('native continuous wrists unexpectedly limited')
  if list(state['sequences'])!=names:raise ValueError('motion ordering mismatch')
  comparisons={}
  for k in ('q','dq','hand'):
   expected=np.stack([s[k][0] for s in sources]);observed=np.asarray(state[k]);comparisons[k]=dict(exact=bool(np.array_equal(expected,observed)),maximum_absolute_error=float(np.abs(expected-observed).max()))
  expected=np.stack([s['obj'][0] for s in sources]);objects=np.asarray(state['objects']);observed=np.repeat(np.eye(4)[None],n,0);observed[:,:3,:3]=Rotation.from_quat(objects[:,3:7]).as_matrix();observed[:,:3,3]=objects[:,:3];comparisons['obj_pose']=dict(exact=bool(np.array_equal(expected,observed)),maximum_absolute_error=float(np.abs(expected-observed).max()))
  report['initial_teacher_comparison']=comparisons
  # Identity differences are evidence, never silently relabeled paired equality.
  for t in range(controls+1):
   if time.monotonic()-start>540:raise TimeoutError('bounded replay')
   numeric={k:np.asarray(state[k]).copy() for k in ('q','dq','hand','objects','physical','tick','episodes','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force')};trace.append(numeric)
   if t==controls:break
   raw_target=np.stack([s['pd_targets'][min(t,len(s['pd_targets'])-1)] for s in sources]);target=raw_target.copy();q=numeric['q']
   if a.branch_aware:target[:,3:6]+=2*np.pi*np.round((q[:,3:6]-target[:,3:6])/(2*np.pi))
   value=(target-offset)/scale;value[:,:6]=(target[:,:6]-q[:,:6]-offset[:6])/scale[:6];value[:,6:]=2*value[:,6:]-1;intended=np.zeros_like(value);intended[:,ACTIVE]=value[:,ACTIVE];action=np.clip(intended,-1,1).astype(np.float32);clipped=(np.abs(intended-action)>1e-6).any(-1);report['clipped_envcontrols']+=int(clipped.sum())
   if a.neutral_controls:
    action[:]=0;intended[:]=0;clipped[:]=False;target=np.broadcast_to(offset+scale/2,q.shape).copy();target[:,:6]=q[:,:6]+offset[:6]
    for dst,src,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)):target[:,dst]=target[:,src]*ratio
    raw_target=target.copy()
   send(connection,dict(op='step',action=action.tolist(),episodes=list(state['episodes']),return_pd=True));reply=receive(connection);state=unpack_state(reply['state']);pd=np.asarray(reply['pd_targets'],np.float32)
   err=float(np.abs(pd-target).max());report['maximum_pd_inverse_error']=max(report['maximum_pd_inverse_error'],err);report['physics_controls']+=1;report['environment_controls']+=n
   applied.append(dict(target=target,raw_target=raw_target,actual_pd=pd,action=action,intended=intended,clipped=clipped,within_reference=np.asarray([t<l for l in lengths])))
   if not np.array_equal(np.asarray(state['episodes']),numeric['episodes']) or not np.array_equal(np.asarray(state['tick']),numeric['tick']+1):raise ValueError('unexpected episode/tick')
   if not np.isfinite(np.asarray(state['q'])).all() or not np.isfinite(np.asarray(state['dq'])).all():raise ValueError('nonfinite measured state')
   if t%60==0:write(out/'progress.json',report)
  send(connection,dict(op='close'));receive(connection);child.wait(timeout=20)
  if child.returncode:raise RuntimeError('native exit '+str(child.returncode))
  report.update(status='COMPLETED',elapsed_s=time.monotonic()-start)
 except BaseException as e:report.update(status='FAILED',error=repr(e),elapsed_s=time.monotonic()-start);raise
 finally:
  if trace:np.savez_compressed(out/'trajectory.npz',**{k:np.stack([s[k] for s in trace]) for k in trace[0]})
  if applied:np.savez_compressed(out/'controls.npz',**{k:np.stack([s[k] for s in applied]) for k in applied[0]})
  write(out/'manifest.json',report)
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
