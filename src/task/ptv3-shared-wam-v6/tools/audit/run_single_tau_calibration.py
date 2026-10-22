"""Bounded, single-trajectory same-tau executor calibration; no RL or WAM training."""
import argparse,copy,json,os,shutil,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=2000);p.add_argument('--train-seconds',type=int,default=1800);a=p.parse_args()
    if not 1<=a.updates<=2000 or not 1<=a.train_seconds<=1800:raise ValueError('registered upper fitting budget')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);private=ROOT/'tmp'/out.name;private.mkdir(parents=True,exist_ok=False)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[line.split(',') for line in gpu.splitlines()]
    if any(int(row[1])>128 for row in rows if int(row[0]) in (4,5)):raise ValueError('registered GPU4/5 occupied')
    os.environ.update(CUDA_VISIBLE_DEVICES='5',TMPDIR=str(private),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path[:0]=[str(ROOT/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src'),str(ROOT/'src/task/ptv3-shared-wam-v6/src')]
    import torch
    from scipy.spatial.transform import Rotation
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.inspire_contract import ACTIVE,NATIVE_TO_WAM,WAM_TO_NATIVE,expand_action
    from wam_rl_policy_v3.inspire_shadow import current_state
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_scheduler import TaskSchedulerConfig
    from wam_rl_policy_v3.goal_scheduler import SchedulerConfig
    from wam_rl_policy_v3.measured_execution import online_support
    from wam_rl_policy_v3.executor_contract import task_condition
    from wam_rl_policy_v3.physical_safety import physical_safety
    from ptv3_shared_wam_v6.execution_contract import LiftHoldWitness,grasp_lift_task,encoder_domain_available
    from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime
    torch.set_num_threads(2);torch.manual_seed(196);np.random.seed(196)
    name='s1_cylindersmall_pass_1';teacher=ROOT/'outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r1'
    meta=json.loads((teacher/'manifest.json').read_text());source=next(r for r in meta['records'] if r['sequence']==name)
    source_path=teacher/source['path']
    with np.load(source_path/'trajectory.npz') as z:pd_reference=z['pd_targets'].copy()
    with np.load(source_path/'canonical.npz') as z:points=z['points'].copy()
    side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/name
    with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
    base_path=ROOT/'outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r4/latest.pt'
    packet=torch.load(base_path,map_location='cpu',weights_only=False)
    base=MeasuredInspirePolicy(with_tau=True).cuda().eval();base.load_state_dict(packet['model'],strict=True);base.requires_grad_(False)
    calibrated=copy.deepcopy(base);shape=base.shape(torch.tensor(points[None],dtype=torch.float32,device='cuda'))
    fixed=private/'motions.json';write(fixed,dict(sequences=[name]));sock=private/'rpc.sock'
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(180);server.bind(str(sock));server.listen(1)
    nativeout=ROOT/'outputs/wam-rl-policy-v3'/out.name
    command=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(ROOT/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'),
        '--cache',str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),'--output',str(nativeout),'--socket',str(sock),
        '--server-pid',str(os.getpid()),'--envs','1','--gpu','4','--seed','195','--seconds','3600','--fixed-motions-json',str(fixed),
        '--state-transport','packed','--extended-state','--synchronize-setters']
    report=dict(status='STARTING',experiment_id='P-20261022-single-tau-calibration',run_id=out.name,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sequence=name,source_pd_reference=str(source_path),base_checkpoint=str(base_path),gpu_before=gpu,native_command=command,
        task_contract=LiftHoldWitness.contract,source_full_task_evaluation=False,synthetic8cm_endpoint_success_not_claimed=True,
        update_limit=a.updates,train_seconds_limit=a.train_seconds,fit_updates=0,environment_controls=0,maximum_environment_controls=2100,
        runs=[],fit_curve=[],scope='one seen trajectory, repeated resets; no held-out/generalization/sample-efficiency claim')
    write(out/'manifest.json',report)
    freeze=out/'code';freeze.mkdir()
    for rel in ['src/task/ptv3-shared-wam-v6/tools/audit/run_single_tau_calibration.py','src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py',
                'src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/execution_contract.py','src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/physical_safety.py']:
        dest=freeze/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    (out/'source.diff').write_bytes(subprocess.check_output(['git','diff','HEAD','--','src/task/wam-rl-policy-v3','src/task/ptv3-shared-wam-v6']))
    log=(out/'native.log').open('x');child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'process.json',dict(client_pid=os.getpid(),native_pid=child.pid));connection=None;started=time.monotonic();reference=None;wam=None;first_initial=None
    try:
        connection,_=server.accept();connection.settimeout(120)
        if peer_pid(connection)!=child.pid:raise ValueError('native peer mismatch')
        hello=receive(connection);report['native_initial_audit']=hello['audit'];state=unpack_state(hello['state'])
        if state['sequences']!=[name] or 'physical_joint_limits' not in hello['audit']:raise ValueError('fixed sequence and physical safety metadata required')
        offset=np.asarray(hello['audit']['controller']['offset'],np.float32);scale=np.asarray(hello['audit']['controller']['scale'],np.float32)
        def reset():
            nonlocal state
            send(connection,dict(op='reset',ids=[0]));state=unpack_state(receive(connection)['state'])
        def run(label,model=None,channel=None,collect=False):
            nonlocal state,first_initial
            reset();initial={k:np.asarray(state[k]).copy() for k in ('q','dq','hand','objects','physical','tick','table_pose','contact_force_valid','dof_force_valid')}
            comparison={k:float(np.max(np.abs(value.astype(float)-first_initial[k].astype(float)))) for k,value in initial.items()} if first_initial else {}
            if first_initial is None:first_initial=initial
            if any(error>1e-6 for error in comparison.values()):raise ValueError('common physical initial state differs: '+str(comparison))
            obj=np.asarray(state['objects'])[0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
            target=grasp_lift_task(pose);context=TaskController(target,points,scheduler_config=TaskSchedulerConfig(episode_timeout_s=20.),path_config=SchedulerConfig(stall_timeout_s=30.))
            witness=LiftHoldWitness(obj[2]);history=[];hands=[];trace=[];features=[];commands=[];tau_records=[];valid_records=[];source_clipped=0;safety_abort=None;coverage=0;now=0.;decision=None
            for t in range(301):
                if time.monotonic()-started>3540 or report['environment_controls']>=2100 or shutil.disk_usage(ROOT).free<20*2**30:raise TimeoutError('registered physical wall/steps/storage boundary')
                obj=np.asarray(state['objects'])[0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
                native_hand=np.asarray(state['hand'])[0];hand=native_hand[NATIVE_TO_WAM];contact=np.asarray(state['contact_forces'])[0];now=float(np.asarray(state['tick'])[0])/30
                safe=physical_safety(state,0,points,bounds,hello['audit']['physical_joint_limits']);safety_abort=safe['reason'] if safe['abort'] else None
                if decision is None or not decision.terminated:
                    result=context.observe(pose,hand,obj[7:13],now,contact_forces=contact,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds),workspace_ok=not safe['abort'])
                    decision=result['decision'];feedback=result['feedback']
                else:
                    feedback=context.observer.update(pose,hand,obj[7:13],now,contact_forces=contact,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds))
                witness.update(obj[2],now,feedback,contact,physical_abort=safe['abort'])
                history.append(obj[:7].copy());hands.append(hand.copy());history=history[-4:];hands=hands[-4:]
                task=task_condition(pose,decision);dv=bool(state['dof_force_valid'][0]);cv=bool(state['contact_force_valid'][0])
                previous=np.asarray(commands[-1])[None] if commands else np.zeros((1,12),np.float32)
                current=current_state(np.asarray(state['q']),np.asarray(state['dq']),np.asarray(state['physical']),np.asarray(state['objects']),target.object_path[-1,:3,3][None],previous)[0]
                measured=np.r_[np.clip(current,-5,5),np.asarray(state['dof_force'])[0] if dv else np.zeros(18),contact.reshape(-1) if cv else np.zeros(15),dv,cv].astype(np.float32)
                robot=np.r_[measured[:345],shape[0].cpu().numpy(),measured[345:]]
                tau=np.zeros((8,11,3),np.float32);valid=False;available=bool(len(history)==4 and encoder_domain_available(points,hand,pose))
                if channel=='gt' and available and t+8<len(reference['hand']):
                    # A fixed known reference displacement, never live/future robot feedback.
                    tau=(reference['hand'][t+np.arange(1,9)]-reference['hand'][t]).astype(np.float32);valid=True
                elif channel=='wam' and available:
                    with torch.random.fork_rng(devices=[0]):
                        tau=wam.predict(np.asarray(history)[None],np.asarray(hands)[None],decision.local_goal_pose[None],[decision.next_event.wam_condition()],['dexplore/cylindersmall'])[0,:8][:,WAM_TO_NATIVE]
                    valid=True
                observation=np.r_[robot,task,tau.reshape(-1),float(valid)].astype(np.float32)
                assert observation.shape==(686,) and np.isfinite(observation).all()
                trace.append({k:np.asarray(state[k]).copy() for k in ('q','dq','objects','hand','physical','tick','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','palm_pose')})
                features.append(observation);tau_records.append(tau);valid_records.append(valid)
                if witness.final_success or safe['abort'] or t==300:break
                if collect:
                    raw_target=pd_reference[min(t,len(pd_reference)-1)][None].copy();q=np.asarray(state['q']);target_pd=raw_target.copy()
                    target_pd[:,3:6]+=2*np.pi*np.round((q[:,3:6]-target_pd[:,3:6])/(2*np.pi))
                    value=(target_pd-offset)/scale;value[:,:6]=(target_pd[:,:6]-q[:,:6]-offset[:6])/scale[:6];value[:,6:]=2*value[:,6:]-1
                    full=np.zeros_like(value);full[:,ACTIVE]=value[:,ACTIVE];source_clipped+=int((np.abs(full)>1.000001).any())
                    action=np.clip(full[:,ACTIVE],-1,1).astype(np.float32)
                else:
                    with torch.inference_mode():action=model.actor(torch.tensor(observation[None],device='cuda')).clamp(-1,1).cpu().numpy()
                commands.append(action[0].copy())
                before=list(state['episodes']);tick=np.asarray(state['tick']).copy()
                send(connection,dict(op='step',action=expand_action(action).tolist(),episodes=before));state=unpack_state(receive(connection)['state'])
                if state['episodes']!=before or not np.array_equal(np.asarray(state['tick']),tick+1):raise ValueError('crossed episode/tick')
                report['environment_controls']+=1;coverage+=int(valid)
            entry=dict(label=label,channel=channel or 'source_pd',controls=len(commands),ever_success=witness.ever_success,final_success=witness.final_success,
                longest_hold_elapsed_s=witness.longest,physical_safety_abort=safety_abort,encoder_covered_controls=coverage,coverage=coverage/max(1,len(commands)),
                source_clipped_controls=source_clipped,initial_max_errors=comparison,last_tick=t,source_full_task_success=None,task_contract=witness.contract)
            report['runs'].append(entry);folder=out/label;folder.mkdir()
            np.savez_compressed(folder/'trajectory.npz',**{k:np.stack([s[k] for s in trace]) for k in trace[0]},observation=np.stack(features),action=np.asarray(commands),tau=np.asarray(tau_records),tau_valid=np.asarray(valid_records))
            write(folder/'result.json',entry);write(out/'manifest.json',report);print(json.dumps(entry),flush=True)
            return dict(observation=np.asarray(features)[:len(commands)],action=np.asarray(commands),hand=np.stack([s['hand'][0] for s in trace]))
        report['status']='SOURCE_PD';write(out/'manifest.json',report)
        reference=run('source-pd',collect=True)
        if not report['runs'][-1]['final_success']:raise ValueError('known-controller narrow positive gate failed; do not interpret R failures')
        # Repack the same real states with their fixed GT tau; keep current/task/CAD inputs identical.
        x=reference['observation'].copy();n=len(x)
        for t in range(n):
            valid=t>=3 and t+8<len(reference['hand'])
            x[t,421:685]=(reference['hand'][t+np.arange(1,9)]-reference['hand'][t]).reshape(-1) if valid else 0.
            x[t,685]=float(valid)
        train_x=torch.tensor(x,device='cuda');train_y=torch.tensor(reference['action'],device='cuda')
        calibrated.actor.requires_grad_(True);optimizer=torch.optim.Adam(calibrated.actor.parameters(),lr=1e-4)
        with torch.no_grad():initial_rmse=float((calibrated.actor(train_x)-train_y).square().mean().sqrt())
        fit_start=time.monotonic();report.update(status='CALIBRATING',training_samples=n,initial_action_rmse=initial_rmse)
        for step in range(a.updates):
            if time.monotonic()-fit_start>=a.train_seconds:break
            ids=torch.randint(n,(64,),device='cuda');prediction=calibrated.actor(train_x[ids]);loss=((prediction-train_y[ids])/calibrated.actor.y_scale.clamp(min=.05)).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite fitting loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(calibrated.actor.parameters(),1.);optimizer.step();report['fit_updates']=step+1
            if step==0 or (step+1)%100==0:
                torch.cuda.synchronize();row=dict(update=step+1,loss=float(loss),elapsed_s=time.monotonic()-fit_start,cuda_mib=torch.cuda.max_memory_allocated()/2**20)
                report['fit_curve'].append(row);write(out/'manifest.json',report);print(json.dumps(row),flush=True)
        with torch.no_grad():final_rmse=float((calibrated.actor(train_x)-train_y).square().mean().sqrt())
        report.update(fit_seconds=time.monotonic()-fit_start,final_action_rmse=final_rmse,action_rmse_ratio=final_rmse/max(initial_rmse,1e-12),status='EVALUATING')
        torch.save(dict(schema='ref2dex.single-tau-calibration.v1',model=calibrated.state_dict(),updates=report['fit_updates'],source_sequence=name,base_checkpoint=str(base_path),task_contract=LiftHoldWitness.contract),out/'calibrated.pt')
        calibrated.eval().requires_grad_(False)
        for repeat in range(2):run('baseline-gt-'+str(repeat),base,'gt');run('calibrated-gt-'+str(repeat),calibrated,'gt')
        wam=OnlineV6Runtime(ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry',device='cuda:0',forecast_only=True)
        run('baseline-wam',base,'wam');run('calibrated-wam',calibrated,'wam')
        report.update(status='COMPLETED',elapsed_s=time.monotonic()-started,wam_step=wam.step,wam_seconds=wam.seconds)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started);raise
    finally:
        write(out/'manifest.json',report)
        if connection:
            try:send(connection,dict(op='close'));receive(connection)
            except Exception:pass
            connection.close()
        server.close();log.close()
        if child.poll() is None:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=20)
        write(out/'exit-receipt.json',dict(native_pid=child.pid,native_returncode=child.returncode,client_pid=os.getpid(),environment_controls=report['environment_controls']))
    return 0


if __name__=='__main__':raise SystemExit(main())
