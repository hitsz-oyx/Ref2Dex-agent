"""V7: measured three-stage grasp interface, one matched 2x5 improvement batch."""
import argparse,io,json,os,shutil,socket,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path.insert(0,str(TASK/'tools/audit'))
from repeat_frozen_tau_chains import freeze,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--original-run',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1')
    p.add_argument('--native-gpu',type=int,choices=[4,5,6],default=6);p.add_argument('--model-gpu',type=int,choices=[4,5,6],default=5)
    p.add_argument('--carry-run',type=Path,help='Retain completed unchanged baseline, all earlier budget, and failed-prototype evidence')
    a=p.parse_args();out=a.output.resolve();oldrun=a.original_run.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents or a.native_gpu==a.model_gpu:raise ValueError('fresh task-local output and distinct allowed GPUs required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if any(int(row.split(',')[1])>64 for row in gpu.splitlines() if int(row.split(',')[0]) in (a.native_gpu,a.model_gpu)):raise RuntimeError('registered GPUs occupied')
    out.mkdir(parents=True);frozen=freeze(out,oldrun)
    shutil.copyfile(TASK/'src/wam_rl_policy_v3/pose_consumer.py',frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/pose_consumer.py')
    shutil.copyfile(ROOT/'src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py',frozen/'src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py')
    shutil.copyfile(Path(__file__),out/'runner.py')
    private=ROOT/'tmp'/out.name;private.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.model_gpu),REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_V6_V5_CODE_ROOT=str(frozen),REF2DEX_V5_BASE_CODE_ROOT=str(frozen),TMPDIR=str(private),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path[:0]=[str(frozen/'src/task/wam-rl-policy-v3/src'),str(frozen/'src/task/trajectory-policy/src'),str(frozen/'src/task/ptv3-shared-wam-v6/src')]
    import torch
    from scipy.spatial.transform import Rotation
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.inspire_contract import ACTIVE,NATIVE_TO_WAM,WAM_TO_NATIVE,expand_action
    from wam_rl_policy_v3.inspire_shadow import current_state
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_scheduler import TaskSchedulerConfig,SCHEDULER_SCHEMA
    from wam_rl_policy_v3.goal_scheduler import SchedulerConfig
    from wam_rl_policy_v3.measured_execution import online_support
    from wam_rl_policy_v3.executor_contract import task_condition
    from wam_rl_policy_v3.physical_safety import physical_safety
    from wam_rl_policy_v3.tau_replay_contract import reference_tau,pd_targets
    from ptv3_shared_wam_v6.execution_contract import LiftHoldWitness,grasp_lift_task,encoder_domain_available
    from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime
    from wam_rl_policy_v3.pose_consumer import (MeasuredGraspConsumer,pose7,reduce_to_wrist_plan,grasp_intent_from_retargeter,GRASP_INTERFACE)
    from wam_rl_policy_v3.measured_reward_bridge import reward_features
    from ptv3_shared_wam_v6.staged_progress import StagedProgress,STAGE_CONTRACT
    torch.set_num_threads(2);torch.manual_seed(196);np.random.seed(196);started=time.monotonic()
    original=json.loads((oldrun/'manifest.json').read_text());name=original['source_sequence'];source=Path(original['source_pd_reference'])
    with np.load(source/'trajectory.npz') as z:source_pd=z['pd_targets'].copy()
    with np.load(source/'canonical.npz') as z:points=z['points'].copy()
    with np.load(oldrun/'source-pd/trajectory.npz') as z:reference_hand=z['hand'][:,0].copy();reference_initial={k:z[k][0].copy() for k in ('q','dq','hand','objects','physical','tick','table_pose','contact_force_valid','dof_force_valid')}
    side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/name
    with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
    models={}
    for key,path in [('baseline',Path(original['base_checkpoint'])),('calibrated',oldrun/'calibrated.pt')]:
        model=MeasuredInspirePolicy(with_tau=True).cuda().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);models[key]=model.requires_grad_(False)
    with torch.inference_mode():shape=models['baseline'].shape(torch.tensor(points[None],dtype=torch.float32,device='cuda'))[0].cpu().numpy()
    wam=OnlineV6Runtime(ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry',device='cuda:0',forecast_only=True)
    keys=['a-original-wam','b-grasp-intent-wam'];order=[keys[b%2:]+keys[:b%2] for b in range(5)]
    report=dict(status='STARTING',experiment_id='P-20261022-single-tau-calibration',run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol='grasp-intent.measured-feedback.2x5.v2',consumer=GRASP_INTERFACE,stages=STAGE_CONTRACT,scheduler_schema=SCHEDULER_SCHEMA,original_run=str(oldrun),source_sequence=name,source_reference_states=len(reference_hand),
        fixed_reference_regenerated=False,base_checkpoint=original['base_checkpoint'],calibrated_checkpoint=str(oldrun/'calibrated.pt'),wam_step=wam.step,flow_eval_seed=wam.config.get('flow_eval_seed'),
        native_gpu=a.native_gpu,model_gpu=a.model_gpu,gpu_before=gpu,native_seed=195,model_seed=196,envs=1,repeats_per_chain=5,order=order,
        maximum_environment_controls=3000,maximum_wall_s=1800,target_pose_wam_successes=4,decision_rule='main: historical2/5 to new b/5; target b>=4/5; paired current a reported, no attribution chain',learning_updates=0,environment_controls=0,runs=[],maximum_pd_reconstruction_error=0.,task_contract=LiftHoldWitness.contract,
        initial_tolerance=1e-6,unit='same seen trajectory/initial state, balanced repeated resets; not independent tasks or training seeds')
    report['current_environment_controls']=0;report['carried_elapsed_s']=0.
    if a.carry_run:
        previous=json.loads((a.carry_run/'manifest.json').read_text())
        if previous['protocol']!=report['protocol'] or previous['base_checkpoint']!=report['base_checkpoint'] or previous['wam_step']!=report['wam_step']:
            raise ValueError('carry must be the same base policy/task inputs')
        report['environment_controls']=previous['environment_controls']
        report['carried_budget_controls']=previous['environment_controls']
        report['carried_elapsed_s']=previous.get('elapsed_s',0.)
        report['engineering_attempts_source']=str(a.carry_run.resolve())
        report['excluded_changed_prototype_records']=[r for r in previous['runs'] if r['chain']!='a-original-wam']
        report['partial_engineering_episode']=previous.get('partial_episode_saved')
        for row in previous['runs']:
            if row['chain']=='a-original-wam':
                copied=dict(row,carried_unchanged_baseline=True,source_run=str(a.carry_run.resolve()))
                report['runs'].append(copied)
                (out/row['label']).symlink_to(a.carry_run.resolve()/row['label'],target_is_directory=True)
    write(out/'manifest.json',report)
    fixed=private/'motions.json';write(fixed,dict(sequences=[name]));sock=private/'rpc.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(2);server.bind(str(sock));server.listen(1)
    cmd=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(frozen/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'),
        '--cache',str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),'--output',str(out/'physics'),'--socket',str(sock),'--server-pid',str(os.getpid()),
        '--envs','1','--gpu',str(a.native_gpu),'--seed','195','--seconds','1860','--fixed-motions-json',str(fixed),'--state-transport','packed','--extended-state','--synchronize-setters']
    report['native_command']=cmd;log=(out/'native.log').open('x');child=subprocess.Popen(cmd,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'process.json',dict(client_pid=os.getpid(),native_pid=child.pid));connection=None;first_initial=None
    try:
        while connection is None:
            if child.poll() is not None:raise RuntimeError('native exited before handshake: '+str(child.returncode))
            if time.monotonic()-started>180:raise TimeoutError('native startup')
            try:connection,_=server.accept()
            except socket.timeout:continue
        connection.settimeout(120)
        if peer_pid(connection)!=child.pid:raise ValueError('native peer mismatch')
        hello=receive(connection);report['native_initial_audit']=hello['audit'];offset=np.asarray(hello['audit']['controller']['offset'],np.float32);scale=np.asarray(hello['audit']['controller']['scale'],np.float32)
        for block,chains in enumerate(order):
            for chain in chains:
                if any(r['block']==block and r['chain']==chain for r in report['runs']):continue
                episode_saved=False;report['active_episode']=dict(block=block,chain=chain,tick=0)
                send(connection,dict(op='reset',ids=[0]));state=unpack_state(receive(connection)['state'])
                initial={k:np.asarray(state[k]).copy() for k in reference_initial}
                errors={k:float(np.abs(v.astype(float)-first_initial[k].astype(float)).max()) for k,v in initial.items()} if first_initial else {}
                original_errors={k:float(np.abs(v.astype(float)-reference_initial[k].astype(float)).max()) for k,v in initial.items()}
                if first_initial is None:first_initial=initial
                if any(e>1e-6 for e in [*errors.values(),*original_errors.values()]):raise ValueError('visible initial state differs: '+str((errors,original_errors)))
                if state['sequences']!=[name] or int(np.asarray(state['tick'])[0])!=0 or state['contact_force_valid'][0]:raise ValueError('reset ownership/force validity failed')
                obj=np.asarray(state['objects'])[0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
                target=grasp_lift_task(pose);context=TaskController(target,points,scheduler_config=TaskSchedulerConfig(episode_timeout_s=20.),path_config=SchedulerConfig(stall_timeout_s=30.));witness=LiftHoldWitness(obj[2])
                trace=[];observations=[];actions=[];raw_actions=[];expanded=[];actual_targets=[];input_episodes=[];taus=[];valids=[];history=[];hands=[];decision=None;covered=0;clipped=0;source_clipped=0;safety_reason=None
                model=models['baseline'];stages=StagedProgress(obj[2]);consumer=MeasuredGraspConsumer(np.asarray(state['q'])[0],pose7(np.asarray(state['palm_pose'])[0]),offset,scale)
                cursor=0;control_info=[];stage_info=[];forecast_valids=[];full_forecasts=[];forecast_count=0
                for t in range(301):
                    report['active_episode']['tick']=t
                    if time.monotonic()-started+report['carried_elapsed_s']>1800 or report['environment_controls']>3000 or shutil.disk_usage(ROOT).free<20*2**30:raise TimeoutError('registered wall/control/storage budget')
                    obj=np.asarray(state['objects'])[0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
                    hand=np.asarray(state['hand'])[0,NATIVE_TO_WAM];forces=np.asarray(state['contact_forces'])[0]
                    safe=physical_safety(state,0,points,bounds,hello['audit']['physical_joint_limits']);safety_reason=safe['reason'] if safe['abort'] else None
                    if decision is None or not decision.terminated:
                        value=context.observe(pose,hand,obj[7:13],t/30,contact_forces=forces,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds),workspace_ok=not safe['abort']);decision=value['decision'];feedback=value['feedback']
                    else:feedback=context.observer.update(pose,hand,obj[7:13],t/30,contact_forces=forces,force_valid=bool(state['contact_force_valid'][0]),supported=online_support(state,0,pose,points,bounds))
                    features=reward_features(pose,hand,points,decision,obj[7:13],forces)
                    stage=stages.update(obj[2],t/30,features,feedback,forces,physical_abort=safe['abort'])
                    witness.update(obj[2],t/30,feedback,forces,physical_abort=safe['abort']);history.append(obj[:7].copy());hands.append(hand.copy());history=history[-4:];hands=hands[-4:]
                    previous=np.asarray(actions[-1])[None] if actions else np.zeros((1,12),np.float32)
                    current=current_state(np.asarray(state['q']),np.asarray(state['dq']),np.asarray(state['physical']),np.asarray(state['objects']),target.object_path[-1,:3,3][None],previous)[0]
                    dv=bool(state['dof_force_valid'][0]);cv=bool(state['contact_force_valid'][0]);measured=np.r_[np.clip(current,-5,5),np.asarray(state['dof_force'])[0] if dv else np.zeros(18),forces.reshape(-1) if cv else np.zeros(15),dv,cv].astype(np.float32)
                    robot=np.r_[measured[:345],shape,measured[345:]];tau=np.zeros((8,11,3),np.float32);valid=False;available=bool(len(history)==4 and encoder_domain_available(points,hand,pose))
                    future=None;plan=None;forecast_valid=False
                    if available and (chain=='a-original-wam' or consumer.phase=='approach' or (consumer.phase=='close' and consumer.contact_since is None)):
                        with torch.random.fork_rng(devices=[0]):prediction=wam.predict(np.asarray(history)[None],np.asarray(hands)[None],decision.local_goal_pose[None],[decision.next_event.wam_condition()],['dexplore/cylindersmall'])[0][:,WAM_TO_NATIVE]
                        forecast_count+=1;future=np.asarray(state['hand'])[0][None]+prediction;forecast_valid=True
                        tau=prediction[:8];valid=True
                    if future is not None and chain!='a-original-wam':
                        plan=reduce_to_wrist_plan(np.asarray(state['hand'])[0],pose7(np.asarray(state['palm_pose'])[0]),future,pose,points)
                    full_forecasts.append(np.zeros((24,11,3),np.float32) if future is None else future.astype(np.float32))
                    forecast_valids.append(forecast_valid);stage_info.append(stage)
                    observation=np.r_[robot,task_condition(pose,decision),tau.reshape(-1),float(valid)].astype(np.float32)
                    if observation.shape!=(686,) or not np.isfinite(observation).all():raise ValueError('observation contract')
                    trace.append({k:np.asarray(state[k]).copy() for k in ('q','dq','objects','hand','physical','tick','episodes','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','palm_pose')});observations.append(observation);taus.append(tau.copy());valids.append(valid)
                    if witness.final_success or safe['abort'] or t==300:break
                    with torch.inference_mode():raw_action=model.actor(torch.tensor(observation[None],device='cuda')).cpu().numpy()
                    if chain!='a-original-wam':
                        intent=grasp_intent_from_retargeter(np.asarray(state['q'])[0],pose7(np.asarray(state['palm_pose'])[0]),pose,target.object_path[-1],np.clip(raw_action[0],-1,1),offset,scale,plan=plan,tau_valid=valid)
                        evidence=context.observer.last_evidence
                        raw,info=consumer.step(q=np.asarray(state['q'])[0],dq=np.asarray(state['dq'])[0],palm=pose7(np.asarray(state['palm_pose'])[0]),object_pose=pose,object_velocity=obj[7:13],forces=forces,feedback=feedback,near_fingers=evidence['near_fingers'],fingertip_gap_m=evidence['fingertip_gap_m'],intent=intent,now=t/30)
                        raw_action=raw[None]
                    else:info=dict(phase='original-r',plan_mode=None)
                    control_info.append(info)
                    action=np.clip(raw_action,-1,1).astype(np.float32);clipped+=int((np.abs(raw_action-action)>1e-6).any());full=expand_action(action)
                    if report['environment_controls']>=3000:raise TimeoutError('total budget exhausted, including earlier failed attempts')
                    before=list(state['episodes']);tick=np.asarray(state['tick']).copy();predicted_pd=pd_targets(action,np.asarray(state['q']),offset,scale)
                    send(connection,dict(op='step',action=full.tolist(),episodes=before,return_pd=True));reply=receive(connection);report['environment_controls']+=1;report['current_environment_controls']+=1;next_state=unpack_state(reply['state']);actual_pd=np.asarray(reply['pd_targets'])
                    error=float(np.abs(predicted_pd-actual_pd).max());report['maximum_pd_reconstruction_error']=max(report['maximum_pd_reconstruction_error'],error)
                    if error>2e-5:raise ValueError('actual PD differs from independently reconstructed target')
                    if next_state['episodes']!=before or not np.array_equal(np.asarray(next_state['tick']),tick+1):raise ValueError('action episode/tick mismatch')
                    actions.append(action[0]);raw_actions.append(raw_action[0]);expanded.append(full[0]);actual_targets.append(actual_pd[0]);input_episodes.append(before[0]);covered+=int(forecast_valid);state=next_state
                label='block'+str(block)+'-'+chain;folder=out/label;folder.mkdir()
                np.savez_compressed(folder/'trajectory.npz',**{k:np.stack([row[k] for row in trace]) for k in trace[0]},observation=np.stack(observations),action=np.asarray(actions),raw_action=np.asarray(raw_actions),expanded_action=np.asarray(expanded),pd_targets_actual=np.asarray(actual_targets),episodes_input=np.asarray(input_episodes),tau=np.asarray(taus),tau_valid=np.asarray(valids),forecast_valid=np.asarray(forecast_valids),forecast_hand=np.asarray(full_forecasts))
                write(folder/'consumer.json',control_info);write(folder/'stages.json',stage_info)
                record=dict(label=label,chain=chain,block=block,order_position=chains.index(chain),controls=len(actions),ever_success=witness.ever_success,final_success=witness.final_success,longest_hold_elapsed_s=witness.longest,physical_safety_abort=safety_reason,encoder_covered_controls=covered,clipped_controls=clipped,source_clipped_controls=source_clipped,initial_max_errors=errors,old_reference_initial_max_errors=original_errors,terminal_tick=t,wam_calls=forecast_count,phase_switch_time_s=consumer.contact_time,final_stage=stage,final_object_goal_error_m=float(np.linalg.norm(pose[:3,3]-target.object_path[-1,:3,3])),phase_counts={name:sum(info['phase']==name for info in control_info) for name in ('approach','close','lift','hold')})
                episode_saved=True;report['runs'].append(record);report['status']='RUNNING';write(folder/'result.json',record);write(out/'manifest.json',report);print(json.dumps(record),flush=True)
        send(connection,dict(op='close'));receive(connection);child.wait(timeout=20)
        if child.returncode:raise RuntimeError('native exit '+str(child.returncode))
        report.update(status='COMPLETED',elapsed_s=time.monotonic()-started+report['carried_elapsed_s'],current_elapsed_s=time.monotonic()-started,wam_seconds=wam.seconds,native_returncode=child.returncode);write(out/'manifest.json',report)
    except BaseException as error:
        if 'episode_saved' in locals() and not episode_saved and 'trace' in locals() and trace:
            partial=out/'partial-episode';partial.mkdir(exist_ok=True)
            np.savez_compressed(partial/'trajectory.npz',**{k:np.stack([row[k] for row in trace]) for k in trace[0]},observation=np.asarray(observations),action=np.asarray(actions),raw_action=np.asarray(raw_actions),pd_targets_actual=np.asarray(actual_targets),tau=np.asarray(taus),tau_valid=np.asarray(valids))
            report['partial_episode_saved']=str(partial)
        report.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started+report['carried_elapsed_s']);write(out/'manifest.json',report);raise
    finally:
        if connection:connection.close()
        server.close();log.close()
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        write(out/'exit-receipt.json',dict(native_pid=child.pid,native_returncode=child.returncode,client_pid=os.getpid(),environment_controls=report['environment_controls']))
    return 0


if __name__=='__main__':raise SystemExit(main())
