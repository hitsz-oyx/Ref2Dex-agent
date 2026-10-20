"""Bounded native Shadow loop using the formal current565 control boundary.

No teacher actor or oracle rollout. Fixed-horizon goal outcomes are descriptive,
not evidence that the spatial scheduler certifies reachability.
"""
import argparse,json,os,random,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get("REF2DEX_RUN_ROOT",TASK.parents[2])).resolve()
sys.path.insert(0,str(TASK/'src'))
from native_helpers import DEFAULT_OBJECT,load_replay_helpers,write_json


def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',type=Path,required=True)
 p.add_argument('--gpu',type=int,default=6);p.add_argument('--seed',type=int,default=0)
 p.add_argument('--ticks',type=int,default=96);p.add_argument('--episodes',type=int,default=2)
 p.add_argument('--seconds',type=int,default=300);p.add_argument('--object-code',default=DEFAULT_OBJECT)
 p.add_argument('--scale',type=float,default=.08)
 p.add_argument('--ppo-smoke',action='store_true',help='One on-policy rollout and bounded PPO update; engineering only')
 p.add_argument('--envs',type=int,default=1);p.add_argument('--ppo-updates',type=int,default=0)
 p.add_argument('--official-ppo',action='store_true')
 p.add_argument('--adaptive-kl',choices=['actual','legacy'],help='Scheduled training defaults actual; native published evaluation defaults legacy')
 p.add_argument('--native-checkpoint-iteration',type=int,default=None,help='Known local finetuning iteration; omit for published weights')
 p.add_argument('--native-evaluation',action='store_true',help='Evaluate the historical official 300D policy without WAM conditioning')
 p.add_argument('--evaluate',action='store_true',help='Deterministic first-episode evaluation; no optimizer updates')
 p.add_argument('--bc-evaluation',action='store_true',help='Evaluate only --policy BC weights; never load an RL checkpoint or construct PPO')
 p.add_argument('--task-controller',action='store_true',help='Use TaskController goals; trained310/575 BC loads directly, legacy565 migrates explicitly')
 p.add_argument('--object-manifest',type=Path);p.add_argument('--rl-checkpoint',type=Path)
 p.add_argument('--wam-checkpoint',type=Path);p.add_argument('--wam-surface-manifest',type=Path)
 p.add_argument('--allow-wam-geometry-replacement',action='store_true',help='Evaluation only: explicitly substitute the local geometry manifest')
 p.add_argument('--checkpoint-migration-reason',help='Explicitly acknowledge a legacy/unbound or changed execution contract; recorded in output')
 p.add_argument('--execution-protocol',choices=['corrected','legacy'],help='Default corrected; native official reproduction defaults legacy')
 p.add_argument('--goal-type',choices=['position_only','se3_pose'],default='position_only')
 p.add_argument('--termination-policy',choices=['native','wam_workspace','common_workspace'],help='Use common_workspace for matched native/WAM comparisons')
 p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
 p.add_argument('--rendezvous',default='tcp://127.0.0.1:29671')
 p.add_argument('--save-every',type=int,default=20)
 a=p.parse_args();a.output=a.output.resolve();a.policy=a.policy.resolve()
 if not 1<=a.episodes<=4 or not 1<=a.ticks<=180:raise ValueError('Bounded native probe required')
 if a.ppo_smoke and (a.episodes!=1 or a.ticks%4):raise ValueError('PPO smoke needs one episode and ticks divisible by4')
 if not 1<=a.envs<=(256 if a.official_ppo else 32) or not 0<=a.ppo_updates<=(10000 if a.official_ppo else 100):raise ValueError('Bounded batch pilot required')
 if a.native_evaluation and not a.evaluate:raise ValueError('Native checkpoint comparison is evaluation-only')
 if a.allow_wam_geometry_replacement and not a.evaluate:raise ValueError('Geometry replacement is evaluation-only')
 if a.evaluate and (not a.official_ppo or (not a.rl_checkpoint and not a.bc_evaluation) or a.ppo_updates or a.world_size!=1):raise ValueError('Evaluation requires a declared BC or official checkpoint, single rank and zero updates')
 if a.bc_evaluation and (not a.evaluate or a.rl_checkpoint or a.native_evaluation):raise ValueError('BC evaluation forbids RL checkpoints and native published policies')
 if a.task_controller and (not a.evaluate or a.native_evaluation or a.goal_type!='position_only' or not a.checkpoint_migration_reason):raise ValueError('Task probe needs explicit position-only evaluation migration')
 if a.envs>1 and not (a.ppo_updates or a.evaluate):raise ValueError('Batched execution requires PPO pilot')
 if a.ppo_updates and (a.ppo_smoke or a.ticks*a.envs%4):raise ValueError('Equal PPO minibatches required')
 if not 1<=a.world_size<=6 or not 0<=a.rank<a.world_size or a.save_every<1:raise ValueError('Bounded distributed rank required')
 if a.object_manifest:a.object_manifest=a.object_manifest.resolve()
 if a.rl_checkpoint:a.rl_checkpoint=a.rl_checkpoint.resolve()
 if a.wam_checkpoint:a.wam_checkpoint=a.wam_checkpoint.resolve()
 if a.wam_surface_manifest:a.wam_surface_manifest=a.wam_surface_manifest.resolve()
 if a.output.exists():raise ValueError('Fresh output required')
 a.output.mkdir(parents=True);(a.output/'tmp').mkdir()
 os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(a.output/'tmp'))
 if 'float' not in np.__dict__:np.float=float
 from isaacgym import gymapi,gymutil
 from isaacgym.torch_utils import quat_apply
 import torch,yaml
 from wam_rl_policy_v3.bc_mlp import CurrentMLP,SCHEMA,SCHEDULED_SCHEMA
 from wam_rl_policy_v3.control_contracts import MeasuredShadowState,TaskIntent,scheduled_conditioning_contract
 from wam_rl_policy_v3.control_pipeline import ScheduledShadowControl
 from wam_rl_policy_v3.goal_scheduler import OrderedGoalScheduler
 from wam_rl_policy_v3.scheduled_executor import BCWarmStartActor
 from wam_rl_policy_v3.wam_v4_runtime import V4Runtime,object_poses
 from wam_rl_policy_v3.hold_metrics import summarize_realtime_hold
 from wam_rl_policy_v3.local_goal_metrics import local_goal_outcomes
 torch.set_num_threads(2)
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('native loop deadline')));signal.alarm(a.seconds)
 commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 manifest=dict(status='RUNNING',git_commit=commit,python=sys.executable,torch=torch.__version__,gpu=a.gpu,
    rank=a.rank,world_size=a.world_size,object_manifest=str(a.object_manifest),resume=str(a.rl_checkpoint),seed=a.seed,ticks=8 if a.official_ppo else a.ticks,episodes=a.episodes,envs=a.envs,ppo_updates=a.ppo_updates,policy=str(a.policy),scope='native interface engineering; not policy success validation')
 write_json(a.output/'manifest.json',manifest);started=time.monotonic();task=None
 try:
  vendor=ROOT/'third_party/UniDexGrasp2';assets=ROOT/'outputs/video-data-preparation/unidexgrasp-policy-assets-20261014-r1/assets'
  sys.path.insert(0,str(vendor/'dexgrasp'));replay=load_replay_helpers()
  Task=replay.task_class(vendor,a.output)
  cfg=yaml.safe_load((vendor/'dexgrasp/cfg/shadow_hand_grasp.yaml').read_text())
  from wam_rl_policy_v3.mixed_objects import read_objects,bind_environment_objects
  if a.object_manifest:assets=Path(json.loads(a.object_manifest.read_text())['asset_root'])
  objects=read_objects(a.object_manifest,a.envs) if a.object_manifest else [dict(object_code=a.object_code,scale=a.scale)]
  cfg['env'].update(numEnvs=a.envs,random_time=a.official_ppo,printNumSuccesses=False,
    object_code_dict={r['object_code']:[r['scale']] for r in objects})
  cfg['env']['asset']['assetRoot']=str(assets);cfg['graphics_device_id']=-1
  sim=gymapi.SimParams();sim.dt=1/60;sim.num_client_threads=0;sim.use_gpu_pipeline=True
  sim.physx.solver_type=1;sim.physx.num_position_iterations=4;sim.physx.num_velocity_iterations=0
  sim.physx.num_threads=4;sim.physx.use_gpu=True;sim.physx.num_subscenes=0;sim.physx.max_gpu_contact_pairs=8*1024*1024
  gymutil.parse_sim_config(cfg['sim'],sim)
  work=a.output/'runtime-work/dexgrasp';work.mkdir(parents=True)
  (work.parent/'assets').symlink_to(assets.resolve(),target_is_directory=True);os.chdir(work)
  if a.evaluate:
   random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
   cfg['env']['random_time']=False
  task=Task(cfg,sim,gymapi.SIM_PHYSX,'cuda',0,True)
  env_objects=bind_environment_objects(task,objects)
  write_json(a.output/'env-objects.json',env_objects)
  if not np.isclose(sim.dt*task.control_freq_inv,1/60):raise ValueError('Native control frequency is not60Hz')
  names=['robot0:palm','robot0:thhub','robot0:ffknuckle','robot0:mfknuckle','robot0:rfknuckle','robot0:lfknuckle',
         'robot0:thdistal','robot0:ffdistal','robot0:mfdistal','robot0:rfdistal','robot0:lfdistal']
  ids=[task.gym.find_actor_rigid_body_index(task.envs[0],task.shadow_hands[0],name,gymapi.DOMAIN_ACTOR) for name in names]
  if min(ids)<0:raise ValueError('Missing Shadow landmark')
  contact_tensor=None
  if a.task_controller:
   from isaacgym import gymtorch
   contact_tensor=gymtorch.wrap_tensor(task.gym.acquire_net_contact_force_tensor(task.sim)).view(a.envs,-1,3)
  offsets=torch.tensor([[0,0,.0275]]+[[0,0,.026]]*4,device='cuda')
  def snapshot():
   rigid=task.rigid_body_states[:,ids];hand=rigid[:,:,:3].clone()
   hand[:,6:]+=quat_apply(rigid[:,6:,3:7].reshape(-1,4),offsets).reshape(1,5,3)
   obj=task.object_pose.detach().cpu().numpy().copy()
   return dict(pose=object_poses(obj[:,None])[0,0],hand=hand[0].cpu().numpy().copy(),
     current=task.obs_buf[0].cpu().numpy().copy(),goal=task.goal_pos[0].cpu().numpy().copy())
  def sim_step(action):
   task.pre_physics_step(action)
   for _ in range(task.control_freq_inv):task.gym.simulate(task.sim);task.gym.fetch_results(task.sim,True)
   task.post_physics_step()
  packet=torch.load(a.policy,map_location='cpu',weights_only=False);old=None
  manifest.update(policy_step=packet.get('step'),policy_source_schema=packet.get('schema'))
  if packet.get('schema')==SCHEMA:
   old=CurrentMLP().cuda().eval();old.load_state_dict(packet['model'])
   expanded=CurrentMLP(565);expanded.x_scale[300:564]=.05;expanded.initialize_current(packet)
   packet=dict(schema=SCHEDULED_SCHEMA,model=expanded.state_dict(),config=dict(conditioning=scheduled_conditioning_contract()))
  actor=BCWarmStartActor(packet).cuda().eval()
  critic=None
  if a.ppo_smoke:
   from wam_rl_policy_v3.ppo import Critic
   from wam_rl_policy_v3.recipe_ppo import update_minibatches
   torch.manual_seed(a.seed);critic=Critic(obs_dim=565,value_scale=100.).cuda()
  cache=ROOT/'outputs/shadow-finetune/policy-data-full-20261015-r1';snap=ROOT/'outputs/ptv3-shared-wam-v4/local-control-snapshot-20261018-r1'
  wam_checkpoint=a.wam_checkpoint or snap/'step12000-local.pt'
  wam_surface=a.wam_surface_manifest or snap/'surface-manifest.json'
  runtime=V4Runtime(wam_checkpoint,env_objects,cache,dense_manifest=wam_surface,
    allow_geometry_replacement=a.allow_wam_geometry_replacement)
  manifest.update(wam_checkpoint=str(wam_checkpoint),wam_surface_manifest=str(wam_surface),
    wam_geometry_identity=runtime.geometry_identity,execution_protocol=a.execution_protocol,
    goal_type=a.goal_type,termination_policy=a.termination_policy,checkpoint_migration_reason=a.checkpoint_migration_reason)
  manifest['wam_step']=runtime.step
  if a.ppo_updates or a.evaluate:
   if a.bc_evaluation:
    from wam_rl_policy_v3.shadow_task_probe import run_bc as run
   elif a.task_controller:
    from wam_rl_policy_v3.shadow_task_probe import run
   elif a.official_ppo:
    from wam_rl_policy_v3.official_bridge import run
   else:
    from wam_rl_policy_v3.scheduled_ppo_loop import run
   def batch_snapshot():
    rigid=task.rigid_body_states[:,ids];hand=rigid[:,:,:3].clone()
    hand[:,6:]+=quat_apply(rigid[:,6:,3:7].reshape(-1,4),offsets.repeat(a.envs,1)).reshape(a.envs,5,3)
    forces=torch.zeros(a.envs,5,3,device='cuda')
    if contact_tensor is not None:
     task.gym.refresh_net_contact_force_tensor(task.sim)
     forces=contact_tensor[:,ids[6:]]
    measured=torch.cat((task.object_linvel,task.object_angvel,forces.flatten(1)),1)
    if a.official_ppo and os.environ.get('REF2DEX_DEEP_OPT') in ('sync','all'):
     # One device-to-host boundary; the official bridge already owns native obs.
     packed=torch.cat((task.object_pose,hand.flatten(1),task.goal_pos,measured),1).detach().cpu().numpy()
     return dict(pose=object_poses(packed[:,:7,None].transpose(0,2,1))[:,0],
       hand=packed[:,7:40].reshape(a.envs,11,3),goal=packed[:,40:43],velocity=packed[:,43:49],contact_forces=packed[:,49:64].reshape(a.envs,5,3))
    measured=measured.detach().cpu().numpy()
    return dict(pose=object_poses(task.object_pose.detach().cpu().numpy()[:,None])[:,0],
      hand=hand.cpu().numpy().copy(),current=task.obs_buf.cpu().numpy().copy(),goal=task.goal_pos.cpu().numpy().copy(),
      velocity=measured[:,:6],contact_forces=measured[:,6:].reshape(a.envs,5,3))
   manifest.update(envs=a.envs,ppo_updates=a.ppo_updates,scope='self-trained BC only; no PPO model or optimizer' if a.bc_evaluation else ('official UniDex PPO with scheduled565 BC/WAM' if a.official_ppo else 'custom truncated PPO pilot'))
   write_json(a.output/'manifest.json',manifest)
   if a.world_size>1:
    import torch.distributed as dist
    from datetime import timedelta
    dist.init_process_group('nccl',init_method=a.rendezvous,rank=a.rank,world_size=a.world_size,timeout=timedelta(seconds=300))
   reports=run(task,batch_snapshot,sim_step,actor,runtime,a)
   manifest.update(status='COMPLETED',reports=reports,elapsed_s=time.monotonic()-started)
   return
  summary=[];initial=[]
  for episode in range(a.episodes):
   for _ in range(2):
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    task.reset_buf[:]=1;task.step(torch.zeros((1,24),device='cuda'),-1)
   if task.reset_buf.any():raise ValueError('reset remains pending')
   first=snapshot();initial.append(first['current']);target=first['pose'].copy();target[:3,3]=first['goal']
   scheduler=OrderedGoalScheduler([first['pose'],target],[0,2],runtime.geometry[0][2],interaction_points=runtime.geometry[0][0],goal_type=a.goal_type)
   control=ScheduledShadowControl(runtime,scheduler,actor,TaskIntent('unidex-position-goal'));episode_id='native-%d'%episode;control.reset(episode_id)
   states=[first];commands=[];applied=[];parent_error=0.;decision_trace=[];reset_reason=None;rollout=[]
   with torch.inference_mode():
    for tick in range(a.ticks):
     measured=states[-1]
     state=MeasuredShadowState(episode_id,tick,tick/60,measured['current'],measured['pose'],measured['hand'],np.ones(11,bool))
     command=control.compute(state,sample=a.ppo_smoke)
     if critic is not None:
      x=torch.as_tensor(command['executor_observation'][None],device='cuda');dist=actor.distribution(x)
      value=float(critic(x)[0]);mean=dist.loc[0].cpu().numpy().copy();std=dist.scale[0].cpu().numpy().copy()
     if old is not None:
      x=torch.as_tensor(command['executor_observation'][None],device='cuda')
      parent_error=max(parent_error,float((actor(x)-old(x[:,:300])).abs().max()))
     before=command['canonical_action'].copy();control.execute(command,sim_step)
     np.testing.assert_array_equal(before,command['canonical_action'])
     if critic is not None:rollout.append(dict(value=value,mean=mean,std=std,reward=float(task.rew_buf[0]),done=bool(task.reset_buf[0])))
     applied.append(task.actions[0].detach().cpu().numpy().copy());commands.append(command);states.append(snapshot())
     d=command['scheduler'];decision_trace.append(dict(tick=tick,mode=d['mode'],reason=d['reason'],confirmed_progress=d['confirmed_progress'],
       arrival_witness=d['arrival_witness'],milestone_witnesses=d['milestone_witnesses'],path_complete=d['complete'],task_success=command['task_success']))
     if task.reset_buf.any():reset_reason='native_reset_requested';break
   poses=np.stack([s['pose'] for s in states]);goals=np.stack([c['goal']['goal_pose'] for c in commands])
   outcomes=local_goal_outcomes(poses,goals)
   np.savez_compressed(a.output/('episode-%d.npz'%episode),poses=poses,goal_pose=goals,
     observation=np.stack([s['current'] for s in states]),hand=np.stack([s['hand'] for s in states]),
     tau=np.stack([c['tau'] for c in commands]),executor_input=np.stack([c['executor_observation'] for c in commands]),
     canonical_action=np.stack([c['canonical_action'] for c in commands]),applied_action=np.stack(applied),
     **{'goal_audit_'+k:v for k,v in outcomes.items()})
   write_json(a.output/('decisions-%d.json'%episode),decision_trace)
   distances=np.linalg.norm(poses[:,:3,3]-first['goal'],axis=-1)
   metrics=summarize_realtime_hold(distances.tolist())
   report=dict(episode=episode,executed_ticks=len(commands),parent_mean_max_error=parent_error,
    reset_reason=reset_reason,startup_valid=[bool(c['tau_contract']['valid'].all()) for c in commands[:7]],
    hold=metrics,goal_audit_eligible=len(outcomes['ticks']),goal_audit_endpoint_error_mean_m=float(np.mean(outcomes['endpoint_position_error_m'])) if len(outcomes['ticks']) else None,
    goal_audit_scope='observed under this policy; no claim of goal controllability',final_distance_m=float(distances[-1]),max_lift_m=float(poses[:,2,3].max()-poses[0,2,3]))
   if old is not None and parent_error>1e-5:raise ValueError('Expanded parent BC behavior changed')
   if critic is not None:
    if len(commands)%4:raise ValueError('Early terminal not divisible by4; preserve rollout without update')
    if rollout[-1]['done']:bootstrap=torch.zeros(1,device='cuda')
    else:
     last=states[-1];final=MeasuredShadowState(episode_id,len(commands),len(commands)/60,last['current'],last['pose'],last['hand'],np.ones(11,bool))
     terminal_command=control.compute(final)
     with torch.no_grad():bootstrap=critic(torch.as_tensor(terminal_command['executor_observation'][None],device='cuda')).detach()
    batch={k:torch.as_tensor(np.asarray([r[k] for r in rollout]),device='cuda').unsqueeze(1) for k in ('value','mean','std','reward','done')}
    batch.update(observation=torch.as_tensor(np.stack([c['executor_observation'] for c in commands]),device='cuda').unsqueeze(1),
      latent=torch.as_tensor(np.stack([c['raw_action'] for c in commands]),device='cuda').unsqueeze(1),
      logp=torch.as_tensor([c['action_log_prob'] for c in commands],device='cuda').unsqueeze(1),bootstrap=bootstrap)
    batch['value']=batch['value'].float();batch['reward']=batch['reward'].float()
    before=actor.actor.net[0].weight.detach().clone();actor_opt=torch.optim.Adam(actor.parameters(),lr=1e-5)
    critic_opt=torch.optim.Adam(critic.parameters(),lr=1e-4)
    metrics=update_minibatches(actor,critic,batch,actor_opt,critic_opt,bc_weight=0.,target_kl=.01,epochs=1,minibatches=4)
    metrics.update(scope='one native on-policy engineering update; no policy improvement claim',
      reward='unmodified pinned UniDex native reward',bootstrap=float(bootstrap[0]),
      actor_first_layer_max_change=float((actor.actor.net[0].weight-before).abs().max()))
    if metrics['actor_steps']<1 or metrics['actor_first_layer_max_change']==0:raise ValueError('No actor update accepted')
    torch.save(dict(schema='ref2dex.scheduled-shadow-ppo-engineering.v1',actor=actor.state_dict(),critic=critic.state_dict(),
      actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),conditioning=scheduled_conditioning_contract()),a.output/'ppo-smoke.pt')
    torch.save({k:v.detach().cpu() for k,v in batch.items()},a.output/'rollout.pt')
    write_json(a.output/'ppo-update.json',metrics);report['ppo']=metrics
   summary.append(report);write_json(a.output/'episodes.json',summary);print(json.dumps(report),flush=True)
  manifest.update(status='COMPLETED',elapsed_s=time.monotonic()-started,episodes=summary,
    repeated_reset_max_observation_error=float(np.max(np.abs(np.stack(initial)-initial[0]))),peak_mib=torch.cuda.max_memory_allocated()/2**20,
    conditioning=scheduled_conditioning_contract(),warmstart_expansion_only=old is not None)
 except BaseException as error:manifest.update(status='FAILED',error=repr(error));raise
 finally:
  write_json(a.output/'manifest.json',manifest);signal.alarm(0)
  if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
  if task is not None:task.gym.destroy_sim(task.sim)

if __name__=='__main__':main()
