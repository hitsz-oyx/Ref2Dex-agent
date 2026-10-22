"""V8: frozen consumer; absent-future, fixed-template, new-initial and live-GT controls."""
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
    p.add_argument('--suite',choices=['ablation','robust','gt'],required=True)
    p.add_argument('--sequence',default='s1_cylindersmall_pass_1')
    p.add_argument('--carry-run',type=Path,help='Resume completed unchanged arms; retain every partial control and elapsed second')
    p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args();out=a.output.resolve();oldrun=a.original_run.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents or a.native_gpu==a.model_gpu:raise ValueError('fresh task-local output and distinct allowed GPUs required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if any(int(row.split(',')[1])>64 for row in gpu.splitlines() if int(row.split(',')[0]) in (a.native_gpu,a.model_gpu)):raise RuntimeError('registered GPUs occupied')
    out.mkdir(parents=True);frozen=freeze(out,oldrun)
    shutil.copyfile(TASK/'src/wam_rl_policy_v3/pose_consumer.py',frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/pose_consumer.py')
    shutil.copyfile(ROOT/'src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py',frozen/'src/task/ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/staged_progress.py')
    shutil.copyfile(TASK/'src/wam_rl_policy_v3/evaluation_sources.py',frozen/'src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/evaluation_sources.py')
    shutil.copyfile(TASK/'tools/audit/measure_s2d_controls.py',out/'s2d_mapping.py')
    shutil.copyfile(TASK/'tools/audit/audit_first_tau_loss.py',out/'physical_scoring.py')
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
    from wam_rl_policy_v3.evaluation_sources import absent_tau,freeze_template,template_intent,gt_future,continuous_hold
    sys.path.insert(0,str(out))
    from physical_scoring import physical_rows
    from s2d_mapping import object_anchored_targets
    torch.set_num_threads(2);torch.manual_seed(196);np.random.seed(196);started=time.monotonic()
    original=json.loads((oldrun/'manifest.json').read_text());name=a.sequence
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())
    rows=catalog['records'] if 'records' in catalog else catalog['sequences']
    catalog_record=next(r for r in rows if r['sequence']==name);source=Path(catalog_record['source'])
    with np.load(source/'canonical.npz') as z:points=z['points'].copy()
    reference_path=ROOT/'src/task/ptv3-shared-wam-v6/outputs/s2b-reference-r2/reference.npz'
    with np.load(reference_path) as z:reference={k:z[k].copy() for k in z.files}
    if name==original['source_sequence']:
        with np.load(oldrun/'source-pd/trajectory.npz') as z:reference_initial={k:z[k][0].copy() for k in ('q','dq','hand','objects','physical','tick','table_pose','contact_force_valid','dof_force_valid')}
    else:
        with np.load(source/'trajectory.npz') as z:reference_initial={k:z[k][0][None].copy() for k in ('q','dq','hand')}
        reference_initial['q'][:,3:6]=(reference_initial['q'][:,3:6]+np.pi)%(2*np.pi)-np.pi
    side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/name
    with np.load(side/'admission.npz') as z:bounds=z['table_bounds'].copy()
    models={}
    for key,path in [('baseline',Path(original['base_checkpoint'])),('calibrated',oldrun/'calibrated.pt')]:
        model=MeasuredInspirePolicy(with_tau=True).cuda().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);models[key]=model.requires_grad_(False)
    with torch.inference_mode():shape=models['baseline'].shape(torch.tensor(points[None],dtype=torch.float32,device='cuda'))[0].cpu().numpy()
    wam=OnlineV6Runtime(ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry',device='cuda:0',forecast_only=True)
    keys={'ablation':['wam','zero-valid','no-tau','template'],'robust':['wam'],'gt':['wam','gt']}[a.suite]
    if not 1<=a.repeats<=5:raise ValueError('registered repeat bound 1..5')
    if a.suite!='robust' and (a.repeats!=5 or name!=original['source_sequence']):raise ValueError('matched arms require registered five seen initial resets')
    if a.suite=='robust' and a.repeats!=1:raise ValueError('one trial per registered new initial')
    order=[keys[b%len(keys):]+keys[:b%len(keys)] for b in range(a.repeats)];maximum_controls=300*len(keys)*a.repeats
    report=dict(status='STARTING',experiment_id='P-20261022-single-tau-calibration',run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol='grasp-intent.v8.frozen-consumer.'+a.suite,consumer=GRASP_INTERFACE,stages=STAGE_CONTRACT,scheduler_schema=SCHEDULER_SCHEMA,original_run=str(oldrun),source_sequence=name,source_object=catalog_record['object'],source_split=catalog_record['split'],source_reference=str(reference_path),
        base_checkpoint=original['base_checkpoint'],calibrated_checkpoint_loaded_unused=str(oldrun/'calibrated.pt'),wam_step=wam.step,flow_eval_seed=wam.config.get('flow_eval_seed'),
        native_gpu=a.native_gpu,model_gpu=a.model_gpu,gpu_before=gpu,native_seed=195,model_seed=196,envs=1,repeats_per_chain=a.repeats,order=order,
        maximum_environment_controls=maximum_controls,maximum_wall_s=1800,learning_updates=0,environment_controls=0,runs=[],maximum_pd_reconstruction_error=0.,task_contract=LiftHoldWitness.contract,
        initial_tolerance=2e-6,unit='repeated seen initial for ablation/GT; one new initial per robust case, not independent training seeds',
        template_provenance='single source-PD physical narrow-success demonstration, fixed before trials; not data-matched pretraining claim',
        current_environment_controls=0,carried_elapsed_s=0.)
    if a.carry_run:
        previous=json.loads((a.carry_run/'manifest.json').read_text())
        for key in ('protocol','source_sequence','base_checkpoint','wam_step','order','maximum_environment_controls'):
            if previous[key]!=report[key]:raise ValueError('resume input differs: '+key)
        report['environment_controls']=previous['environment_controls'];report['carried_elapsed_s']=previous['elapsed_s']
        report['interrupted_run']=str(a.carry_run.resolve());report['partial_controls_retained_in_budget']=previous['environment_controls']-sum(r['controls'] for r in previous['runs'])
        for row in previous['runs']:
            report['runs'].append(dict(row,carried_unchanged=True,source_run=str(a.carry_run.resolve())))
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
        source_side=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/original['source_sequence']
        with np.load(Path(original['source_pd_reference'])/'canonical.npz') as z:source_points=z['points'].copy()
        with np.load(source_side/'admission.npz') as z:source_bounds=z['table_bounds'].copy()
        source_rows=physical_rows(reference,source_points,source_bounds,hello['audit']['physical_joint_limits'])
        source_score=continuous_hold(np.arange(len(source_rows))/30,[r['held'] for r in source_rows])
        if not source_score['ever_success']:raise ValueError('GT/template source lacks physical continuous success')
        contact_frame=next(i for i,r in enumerate(source_rows) if r['force_count']>=2)
        held_frame=next(i for i,r in enumerate(source_rows) if r['held'])
        template=freeze_template(reference,contact_frame,held_frame)
        report['reference_qualification']=dict(**source_score,states=len(source_rows),first_contact_frame=contact_frame,first_held_frame=held_frame)
        write(out/'template.json',dict(**{k:v.tolist() for k,v in vars(template).items()},contact_frame=contact_frame,held_frame=held_frame,source=str(reference_path)))
        for block,chains in enumerate(order):
            for chain in chains:
                if any(r['block']==block and r['chain']==chain for r in report['runs']):continue
                episode_saved=False;report['active_episode']=dict(block=block,chain=chain,tick=0)
                send(connection,dict(op='reset',ids=[0]));state=unpack_state(receive(connection)['state'])
                initial={k:np.asarray(state[k]).copy() for k in reference_initial}
                errors={k:float(np.abs(v.astype(float)-first_initial[k].astype(float)).max()) for k,v in initial.items()} if first_initial else {}
                original_errors={k:float(np.abs(v.astype(float)-reference_initial[k].astype(float)).max()) for k,v in initial.items()}
                if first_initial is None:first_initial=initial
                if any(e>2e-6 for e in [*errors.values(),*original_errors.values()]):raise ValueError('visible initial state differs: '+str((errors,original_errors)))
                if state['sequences']!=[name] or int(np.asarray(state['tick'])[0])!=0 or state['contact_force_valid'][0]:raise ValueError('reset ownership/force validity failed')
                obj=np.asarray(state['objects'])[0];pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
                target=grasp_lift_task(pose);context=TaskController(target,points,scheduler_config=TaskSchedulerConfig(episode_timeout_s=20.),path_config=SchedulerConfig(stall_timeout_s=30.));witness=LiftHoldWitness(obj[2])
                trace=[];observations=[];actions=[];raw_actions=[];expanded=[];actual_targets=[];input_episodes=[];taus=[];valids=[];history=[];hands=[];decision=None;covered=0;clipped=0;source_clipped=0;safety_reason=None
                model=models['baseline'];stages=StagedProgress(obj[2]);consumer=MeasuredGraspConsumer(np.asarray(state['q'])[0],pose7(np.asarray(state['palm_pose'])[0]),offset,scale)
                cursor=0;control_info=[];stage_info=[];forecast_valids=[];full_forecasts=[];forecast_count=0;actor_calls=0;adapter_clipped=0;adapter_actions=[]
                for t in range(301):
                    report['active_episode']['tick']=t
                    if time.monotonic()-started+report['carried_elapsed_s']>1800 or report['environment_controls']>maximum_controls or shutil.disk_usage(ROOT).free<20*2**30:raise TimeoutError('registered wall/control/storage budget')
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
                    if available and chain=='wam' and consumer.phase in ('approach','close'):
                        with torch.random.fork_rng(devices=[0]):prediction=wam.predict(np.asarray(history)[None],np.asarray(hands)[None],decision.local_goal_pose[None],[decision.next_event.wam_condition()],['dexplore/'+catalog_record['object']])[0][:,WAM_TO_NATIVE]
                        forecast_count+=1;future=np.asarray(state['hand'])[0][None]+prediction;forecast_valid=True
                        tau=prediction[:8];valid=True
                    if chain=='gt' and consumer.phase in ('approach','close'):
                        future,forecast_valid=gt_future(object_anchored_targets,reference,t,obj,available)
                        if forecast_valid:tau=(future[:8]-np.asarray(state['hand'])[0]).astype(np.float32);valid=True
                    if chain in ('zero-valid','no-tau'):tau,valid=absent_tau(chain,available and consumer.phase in ('approach','close'))
                    if future is not None:
                        plan=reduce_to_wrist_plan(np.asarray(state['hand'])[0],pose7(np.asarray(state['palm_pose'])[0]),future,pose,points)
                    full_forecasts.append(np.zeros((24,11,3),np.float32) if future is None else future.astype(np.float32))
                    forecast_valids.append(forecast_valid);stage_info.append(stage)
                    observation=np.r_[robot,task_condition(pose,decision),tau.reshape(-1),float(valid)].astype(np.float32)
                    if observation.shape!=(686,) or not np.isfinite(observation).all():raise ValueError('observation contract')
                    trace.append({k:np.asarray(state[k]).copy() for k in ('q','dq','objects','hand','physical','tick','episodes','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','palm_pose')});observations.append(observation);taus.append(tau.copy());valids.append(valid)
                    if witness.final_success or safe['abort'] or t==300:break
                    if chain=='template':
                        adapter_action=np.zeros((1,12),np.float32)
                        intent=template_intent(template,np.asarray(state['q'])[0],pose7(np.asarray(state['palm_pose'])[0]),pose,target.object_path[-1],consumer.phase)
                    else:
                        with torch.inference_mode():adapter_action=model.actor(torch.tensor(observation[None],device='cuda')).cpu().numpy()
                        actor_calls+=1;adapter_clipped+=int((np.abs(adapter_action)>1+1e-6).any())
                        intent=grasp_intent_from_retargeter(np.asarray(state['q'])[0],pose7(np.asarray(state['palm_pose'])[0]),pose,target.object_path[-1],np.clip(adapter_action[0],-1,1),offset,scale,plan=plan,tau_valid=valid)
                    evidence=context.observer.last_evidence
                    raw,info=consumer.step(q=np.asarray(state['q'])[0],dq=np.asarray(state['dq'])[0],palm=pose7(np.asarray(state['palm_pose'])[0]),object_pose=pose,object_velocity=obj[7:13],forces=forces,feedback=feedback,near_fingers=evidence['near_fingers'],fingertip_gap_m=evidence['fingertip_gap_m'],intent=intent,now=t/30)
                    raw_action=raw[None];adapter_actions.append(adapter_action[0])
                    control_info.append(info)
                    action=np.clip(raw_action,-1,1).astype(np.float32);clipped+=int((np.abs(raw_action-action)>1e-6).any());full=expand_action(action)
                    if report['environment_controls']>=maximum_controls:raise TimeoutError('total budget exhausted, including earlier failed attempts')
                    before=list(state['episodes']);tick=np.asarray(state['tick']).copy();predicted_pd=pd_targets(action,np.asarray(state['q']),offset,scale)
                    send(connection,dict(op='step',action=full.tolist(),episodes=before,return_pd=True));reply=receive(connection);report['environment_controls']+=1;report['current_environment_controls']+=1;next_state=unpack_state(reply['state']);actual_pd=np.asarray(reply['pd_targets'])
                    error=float(np.abs(predicted_pd-actual_pd).max());report['maximum_pd_reconstruction_error']=max(report['maximum_pd_reconstruction_error'],error)
                    if error>2e-5:raise ValueError('actual PD differs from independently reconstructed target')
                    if next_state['episodes']!=before or not np.array_equal(np.asarray(next_state['tick']),tick+1):raise ValueError('action episode/tick mismatch')
                    actions.append(action[0]);raw_actions.append(raw_action[0]);expanded.append(full[0]);actual_targets.append(actual_pd[0]);input_episodes.append(before[0]);covered+=int(forecast_valid);state=next_state
                label='block'+str(block)+'-'+chain;folder=out/label;folder.mkdir()
                np.savez_compressed(folder/'trajectory.npz',**{k:np.stack([row[k] for row in trace]) for k in trace[0]},observation=np.stack(observations),action=np.asarray(actions),raw_action=np.asarray(raw_actions),adapter_raw_action=np.asarray(adapter_actions),expanded_action=np.asarray(expanded),pd_targets_actual=np.asarray(actual_targets),episodes_input=np.asarray(input_episodes),tau=np.asarray(taus),tau_valid=np.asarray(valids),forecast_valid=np.asarray(forecast_valids),forecast_hand=np.asarray(full_forecasts))
                write(folder/'consumer.json',control_info);write(folder/'stages.json',stage_info)
                record=dict(label=label,chain=chain,block=block,order_position=chains.index(chain),controls=len(actions),ever_success=witness.ever_success,final_success=witness.final_success,longest_hold_elapsed_s=witness.longest,physical_safety_abort=safety_reason,encoder_covered_controls=covered,clipped_controls=clipped,source_clipped_controls=source_clipped,initial_max_errors=errors,old_reference_initial_max_errors=original_errors,terminal_tick=t,wam_calls=forecast_count,actor_calls=actor_calls,adapter_clipped_controls=adapter_clipped,phase_switch_time_s=consumer.contact_time,final_stage=stage,final_object_goal_error_m=float(np.linalg.norm(pose[:3,3]-target.object_path[-1,:3,3])),phase_counts={name:sum(info['phase']==name for info in control_info) for name in ('approach','close','lift','hold')})
                data={k:np.stack([row[k] for row in trace]) for k in trace[0]};data['tau_valid']=np.asarray(valids)
                physical=physical_rows(data,points,bounds,hello['audit']['physical_joint_limits'])
                independent=continuous_hold(np.arange(len(physical))/30,[row['held'] for row in physical])
                write(folder/'independent-score.json',independent)
                if independent['ever_success']!=witness.ever_success or independent['final_success']!=witness.final_success or abs(independent['longest_hold_s']-witness.longest)>1e-8:raise ValueError('physical rescore disagrees with online success')
                record['independent_continuous_score']=independent
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
