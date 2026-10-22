"""Paired frozen executor qualification on robot DEV references; no fitting."""
import argparse,json,os,socket,subprocess,sys,time
from pathlib import Path
from collections import deque
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src')]
def write(p,v):
    p=Path(p);part=p.with_suffix('.part');part.write_text(json.dumps(v,indent=2)+'\n');part.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['replay','gt','healthy','flow'],required=True);p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--native-gpu',type=int,default=3);p.add_argument('--controls',type=int,default=300);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r4/latest.pt')
    p.add_argument('--wam',type=Path);p.add_argument('--stats',type=Path,required=True)
    a=p.parse_args();assert a.gpu in range(4) and a.native_gpu in range(4)
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    scratch=ROOT/'tmp'/('exqual-'+a.arm+'-'+a.output.parent.name+'-'+str(os.getpid()));scratch.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    import numpy as np,torch
    from scipy.spatial.transform import Rotation
    from trajectory_policy.wam_ipc import send,receive
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.robot_adapter import InspireAdapter
    from wam_rl_policy_v3.inspire_shadow import current_state
    from wam_rl_policy_v3.inspire_contract import expand_action,ACTIVE,WAM_TO_NATIVE
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.measured_execution import online_support
    from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime
    torch.set_num_threads(2);torch.manual_seed(185);np.random.seed(185)
    panel=json.loads(a.panel.read_text());rows=panel['records'];names=[r['sequence'] for r in rows];n=len(rows)
    fixed=scratch/'motions.json';write(fixed,dict(sequences=names));sock=scratch/'rpc.sock'
    model=None;runtime=None
    if a.arm!='replay':
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        model=MeasuredInspirePolicy(with_tau=True).cuda().eval();model.load_state_dict(saved['model'],strict=True)
        tauweights=model.state_dict()['actor.net.0.weight'] if 'actor.net.0.weight' in model.state_dict() else None
    if a.arm in ('healthy','flow'):
        runtime=OnlineV6Runtime(a.wam,a.stats,a.output.parent/'geometry')
    contexts=[];points=[];bounds=[];refs=[]
    manifest=Path(panel['source_manifest'])
    for row in rows:
        side=manifest.parent/row['record']['sidecar'];source=Path(row['source'])
        with np.load(source/'trajectory.npz') as z:refs.append({k:z[k].copy() for k in ('hand','action','obj')})
        with np.load(source/'canonical.npz') as z:pts=z['points'].copy()
        points.append(pts);contexts.append(TaskController(TaskSpec.from_packet(json.loads((side/'task.json').read_text())),pts))
        with np.load(side/'admission.npz') as z:bounds.append(z['table_bounds'].copy())
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.settimeout(300);listener.bind(str(sock));listener.listen(1)
    native_out=ROOT/'outputs/wam-rl-policy-v3'/('executor-qualification-20261022-'+a.arm+'-'+a.output.parent.name+'-'+str(os.getpid()))
    cmd=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python',str(ROOT/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'),
         '--cache',str(a.output.parent/'geometry'),'--output',str(native_out),'--socket',str(sock),
         '--server-pid',str(os.getpid()),'--envs',str(n),'--gpu',str(a.native_gpu),'--seed','185','--seconds',str(a.seconds),
         '--fixed-motions-json',str(fixed),'--extended-state','--state-transport','packed','--synchronize-setters']
    start=time.monotonic();state=dict(status='STARTING',arm=a.arm,python=sys.executable,torch=torch.__version__,numpy=np.__version__,seed=185,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checkpoint=str(a.checkpoint),wam=str(a.wam),
        panel=str(a.panel.resolve()),native_command=cmd,samples=n,controls=a.controls,policy_updates=0,native_output=str(native_out),
        gt_contract='world(reference_hand[t+1:t+8] - live_measured_hand[t]); held last reference beyond end',
        adapter='v6 sparse11 observed in joint21; remaining joints/skin invalid; first8 of24 to frozen Scheduled686')
    write(a.output/'manifest.json',state);log=(a.output/'native.log').open('x');proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    connection=None;trace=[];workspace_failed=np.zeros(n,bool);workspace_first=np.full(n,-1,int);previous=np.zeros((n,12),np.float32);hist=[deque(maxlen=4) for _ in rows];hands=[deque(maxlen=4) for _ in rows]
    try:
        connection,_=listener.accept();connection.settimeout(300);hello=receive(connection);snapshot=unpack_state(hello['state'])
        if list(snapshot['sequences'])!=names:raise ValueError('native sequence ordering mismatch')
        np.savez_compressed(a.output/'initial-state.npz',**{k:np.asarray(snapshot[k]) for k in ('q','dq','physical','objects','hand','tick','table_pose')})
        for step in range(a.controls+1):
            if time.monotonic()-start>a.seconds:raise TimeoutError('qualification deadline')
            if __import__('shutil').disk_usage(a.output).free<20*2**30:raise RuntimeError('disk20GiB floor')
            measured=[];tasks=[];poses=[];events=[];goals=[];supports=[];phases=[];progress=[];complete=[]
            for i,row in enumerate(rows):
                obj=np.asarray(snapshot['objects'])[i];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
                hand=InspireAdapter.canonical_hand(np.asarray(snapshot['hand'])[i]);tick=int(np.asarray(snapshot['tick'])[i])
                support=online_support(snapshot,i,pose,points[i],bounds[i]);contact=np.asarray(snapshot['contact_forces'])[i]
                result=contexts[i].observe(pose,hand,obj[7:13],tick/30.,contact_forces=contact,
                    force_valid=bool(np.asarray(snapshot['contact_force_valid'])[i]),supported=support)
                decision=result['decision'];goals.append(decision.local_goal_pose);events.append(decision.next_event.wam_condition())
                endpoint=contexts[i].scheduler.spec.object_path[-1,:3,3]
                base=current_state(np.asarray(snapshot['q'])[i:i+1],np.asarray(snapshot['dq'])[i:i+1],np.asarray(snapshot['physical'])[i:i+1],
                    np.asarray(snapshot['objects'])[i:i+1],endpoint[None],previous[i:i+1])[0]
                dv=bool(np.asarray(snapshot['dof_force_valid'])[i]);cv=bool(np.asarray(snapshot['contact_force_valid'])[i])
                measured.append(np.r_[base,np.asarray(snapshot['dof_force'])[i] if dv else np.zeros(18),contact.flatten() if cv else np.zeros(15),dv,cv])
                tasks.append(result['task']);hist[i].append(obj[:7]);hands[i].append(hand);poses.append(pose)
                supports.append(support);phases.append(int(decision.phase));progress.append(float(decision.path_progress));complete.append(bool(decision.task_complete))
            taus=np.zeros((n,8,11,3),np.float32);valid=np.zeros(n,bool)
            if a.arm=='gt':
                for i,ref in enumerate(refs):
                    tick=int(np.asarray(snapshot['tick'])[i]);idx=np.minimum(tick+np.arange(1,9),len(ref['hand'])-1)
                    taus[i]=(InspireAdapter.canonical_hand(ref['hand'][idx])-hands[i][-1])[:,WAM_TO_NATIVE];valid[i]=True
            elif runtime and step>=3:
                objects4=np.asarray(hist);hands4=np.asarray(hands)
                admissible=runtime.workspace_valid(objects4,hands4)
                newly=~admissible & ~workspace_failed;workspace_first[newly]=step;workspace_failed|=~admissible
                active=np.array([step<int(r['reference_controls']) for r in rows]) & ~workspace_failed
                chosen=np.flatnonzero(active)
                if len(chosen):
                    full=runtime.predict(objects4[chosen],hands4[chosen],np.asarray(goals)[chosen],
                        [events[i] for i in chosen],['dexplore/'+rows[i]['object'] for i in chosen])
                    taus[chosen]=full[:,:8][:,:,WAM_TO_NATIVE];valid[chosen]=True
            if a.arm=='replay':
                command=np.stack([r['action'][min(step,len(r['action'])-1),ACTIVE] for r in refs])
            else:
                with torch.inference_mode():command=model(torch.as_tensor(np.asarray(measured),dtype=torch.float32,device='cuda'),
                    torch.as_tensor(np.asarray(tasks),dtype=torch.float32,device='cuda'),torch.as_tensor(np.asarray(points),dtype=torch.float32,device='cuda'),
                    tau=torch.as_tensor(taus,device='cuda'),tau_valid=torch.as_tensor(valid,device='cuda')).clamp(-1,1).cpu().numpy()
            trace.append(dict(objects=np.asarray(snapshot['objects']).copy(),hand=np.asarray(snapshot['hand']).copy(),q=np.asarray(snapshot['q']).copy(),
                contact=np.asarray(snapshot['contact_forces']).copy(),contact_force_valid=np.asarray(snapshot['contact_force_valid']).copy(),tick=np.asarray(snapshot['tick']).copy(),dq=np.asarray(snapshot['dq']).copy(),supported=np.asarray(supports),phase=np.asarray(phases),progress=np.asarray(progress),
                complete=np.asarray(complete),tau=taus.copy(),tau_valid=valid.copy(),workspace_failed=workspace_failed.copy(),command=command.copy()))
            if step%30==0:
                state.update(status='RUNNING',controls_completed=step,elapsed_s=time.monotonic()-start);write(a.output/'progress.json',state);print(json.dumps(dict(arm=a.arm,step=step,elapsed_s=state['elapsed_s'])),flush=True)
            if step==a.controls:break
            previous[:]=command;send(connection,dict(op='step',action=expand_action(command).tolist(),episodes=list(snapshot['episodes'])))
            reply=receive(connection);next_snapshot=unpack_state(reply['state'])
            if list(next_snapshot['episodes'])!=list(snapshot['episodes']):raise ValueError('unexpected reset')
            snapshot=next_snapshot
        send(connection,dict(op='close'));receive(connection);proc.wait(timeout=60)
        state.update(status='COMPLETED',controls_completed=a.controls,elapsed_s=time.monotonic()-start,
            workspace_failure_first_tick=workspace_first.tolist(),inference_seconds=runtime.seconds if runtime else 0,inference_samples=runtime.samples if runtime else 0)
    except BaseException as e:
        state.update(status='FAILED',error=repr(e),controls_completed=max(0,len(trace)-1),workspace_failure_first_tick=workspace_first.tolist(),elapsed_s=time.monotonic()-start);raise
    finally:
        if trace:np.savez_compressed(a.output/'trajectory.npz',**{k:np.stack([r[k] for r in trace]) for k in trace[0]})
        write(a.output/'manifest.json',state)
        if connection is not None:connection.close()
        listener.close();log.close()
if __name__=='__main__':main()
