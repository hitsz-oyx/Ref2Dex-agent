"""Pinned UniDex official PPO on latest measured686 TaskController observations."""
import argparse,importlib.util,json,os,socket,sys,time,subprocess,shutil
from pathlib import Path
from types import SimpleNamespace
TASK=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src')]

def write(path,data):
 part=path.with_suffix(path.suffix+'.part');part.write_text(json.dumps(data,indent=2)+'\n');part.replace(path)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for arg in ['bc','audit-manifest','geometry-cache','output','socket']:p.add_argument('--'+arg,type=Path,required=True)
 p.add_argument('--gpu',type=int,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1);p.add_argument('--rendezvous')
 p.add_argument('--wam-batch',type=int,default=256);p.add_argument('--shared-tf32',action='store_true');p.add_argument('--updates',type=int,default=10000);p.add_argument('--seconds',type=int,default=172800);p.add_argument('--seed',type=int,default=246);p.add_argument('--reset-probe',action='store_true');p.add_argument('--resume',type=Path)
 a=p.parse_args();a.output=a.output.resolve();a.socket=a.socket.resolve()
 if a.output.exists() or ROOT/'outputs/wam-rl-policy-v3' not in a.output.parents or ROOT/'tmp' not in a.socket.parents or not 1<=a.updates<=10000 or not 30<=a.seconds<=172800:raise ValueError('fresh bounded PPO paths/budget required')
 a.output.mkdir(parents=True);a.socket.parent.mkdir(parents=True,exist_ok=True)
 os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',REF2DEX_DEEP_OPT='all',TMPDIR=str(a.socket.parent),PYTHONDONTWRITEBYTECODE='1',NCCL_P2P_DISABLE='1')
 import numpy as np,torch,yaml
 import torch.distributed as dist
 from trajectory_policy.wam_ipc import receive,send,peer_pid
 from wam_rl_policy_v3.inspire_state_codec import unpack_state
 from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
 from wam_rl_policy_v3.measured_official_env import MeasuredOfficialEnv
 from wam_rl_policy_v3.wam_v4_runtime import V4Runtime
 from wam_rl_policy_v3.official_bridge import prepare_official
 from wam_rl_policy_v3.ppo import reduce
 from wam_rl_policy_v3.scheduled_ppo_loop import parameter_sync_error
 from wam_rl_policy_v3.checkpoint_retention import prune_checkpoints
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(a.seed)
 if a.world_size>1:dist.init_process_group('nccl',init_method=a.rendezvous,rank=a.rank,world_size=a.world_size)
 packet=torch.load(a.bc.resolve(),map_location='cpu',weights_only=False)
 if packet.get('schema')!='ref2dex.inspire-measured-scheduled-bc.v2' or packet['config']['source_manifest']!=str(a.audit_manifest.resolve()):raise ValueError('latest same-source goal+WAM BC required')
 model=MeasuredInspirePolicy(with_tau=True).cuda().eval();model.load_state_dict(packet['model'],strict=True);model.requires_grad_(False)
 listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.settimeout(300);listener.bind(str(a.socket));listener.listen(1)
 commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 state=dict(status='WAITING_NATIVE',experiment_id='P-20261020-inspire-measured-official-ppo',run_id=a.output.parent.name,git_commit=commit,python=sys.executable,torch=torch.__version__,gpu=a.gpu,rank=a.rank,world_size=a.world_size,bc=str(a.bc.resolve()),bc_step=packet['step'],source_manifest=str(a.audit_manifest.resolve()),optimizer_steps=0,iteration=0,
  ppo_engine='pinned UniDexGrasp2 official; actual-KL/global advantage/gradient adapters reused from Shadow',observation_dim=686,cad_encoder='frozen BC encoder, part of checkpoint',reset_contract='step in source episode; terminal reward then eager indexed reset before next sampling',reset_probe=a.reset_probe)
 write(a.output/'manifest.json',state);print('READY',flush=True);connection=None;env=None;ppo=None;started=time.monotonic()
 try:
  connection,_=listener.accept();connection.settimeout(600);hello=receive(connection);state['native_peer_pid']=peer_pid(connection)
  if not hello['audit'].get('extended_state') or hello['audit'].get('state_transport')!='packed':raise ValueError('native measured packed state required')
  state['status']='INITIALIZING_MODELS';write(a.output/'manifest.json',state)
  native=unpack_state(hello['state']);objects=[dict(object_code='dexplore/'+s.split('_')[1],scale=1.) for s in native['sequences']]
  runtime=V4Runtime(packet['config']['wam'],objects,a.geometry_cache.resolve(),dense_manifest=a.geometry_cache/'surface-manifest.json',allow_geometry_replacement=True,device='cuda:0')
  env=MeasuredOfficialEnv(connection,hello,a.audit_manifest.resolve(),runtime,model,reset_probe=a.reset_probe,batch=a.wam_batch,shared_tf32=a.shared_tf32)
  official=prepare_official(ROOT,a.output,masked_resets=False,correct_distribution_kl=True);sys.path.insert(0,str(official))
  spec=importlib.util.spec_from_file_location('measured_official_process',official/'utils/process_sarl.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  cfg=yaml.safe_load((official/'cfg/ppo/config.yaml').read_text());cfg['seed']=a.seed;cfg['policy'].update(backbone_type='pn',freeze_backbone=False,obs_dim=686)
  from wam_rl_policy_v3.distribution_kl import OPTIMIZATION_CONTRACT
  cfg['learn']['adaptive_kl_contract']=OPTIMIZATION_CONTRACT
  torch.manual_seed(a.seed);ppo=module.process_ppo(SimpleNamespace(test=False,vision=False,model_dir=''),env,cfg,str(a.output/'models'))
  from wam_rl_policy_v3.bc_mlp import NormalizedCommandMLP
  actor=NormalizedCommandMLP(686,312,12).cuda();actor.load_state_dict(model.actor.state_dict(),strict=True);ppo.actor_critic.actor=actor
  ppo.optimizer=torch.optim.Adam(ppo.actor_critic.parameters(),lr=cfg['learn']['optim_stepsize'])
  identity=dict(source=str(a.audit_manifest.resolve()),bc=str(a.bc.resolve()),wam=packet['config']['wam'],geometry=str(a.geometry_cache.resolve()),world_size=a.world_size,envs=env.num_envs,physics_backend=hello['audit'].get('physics_backend','gpu'),wam_batch=a.wam_batch,shared_tf32=a.shared_tf32)
  completed=0
  if a.resume:
   saved=torch.load(a.resume,map_location='cuda',weights_only=False)
   if saved['schema']!='ref2dex.inspire-measured-official-ppo.v1' or saved['identity']!=identity:raise ValueError('PPO resume identity changed')
   ppo.actor_critic.load_state_dict(saved['model']);ppo.optimizer.load_state_dict(saved['optimizer']);ppo.step_size=saved['step_size'];completed=saved['iteration']
  if a.world_size>1:
   for value in ppo.actor_critic.state_dict().values():dist.broadcast(value,src=0)
  torch.manual_seed(a.seed+a.rank)
  state.update(status='RUNNING',physics_backend=identity['physics_backend'],envs_per_rank=env.num_envs,total_envs=env.num_envs*a.world_size,global_rollout=env.num_envs*a.world_size*8,config=cfg,wam=packet['config']['wam'],geometry=str(a.geometry_cache.resolve()),cad_frozen=True,wam_batch=a.wam_batch,shared_tf32=a.shared_tf32,native_object_names=hello['audit']['source_object_names'])
  write(a.output/'manifest.json',state)
  def save(_=None):
   if a.rank:return
   iteration=state['iteration'];dest=a.output/('official-iteration-%d.pt'%iteration)
   if dest.exists():return
   part=dest.with_suffix('.pt.part');torch.save(dict(schema='ref2dex.inspire-measured-official-ppo.v1',model=ppo.actor_critic.state_dict(),optimizer=ppo.optimizer.state_dict(),step_size=ppo.step_size,iteration=iteration,config=cfg,
    identity=identity,shape=model.shape.state_dict(),bc_step=packet['step'],resume='model/optimizer with fresh episodes'),part);part.replace(dest)
   prune_checkpoints(a.output,final_iteration=completed+a.updates)
  original_update=ppo.update;original_step=ppo.optimizer.step
  def step(*args,**kwargs):
   result=original_step(*args,**kwargs);state['optimizer_steps']+=1;return result
  ppo.optimizer.step=step
  def update():
   stop=(a.output.parent/'STOP').exists() or shutil.disk_usage(ROOT).free<20*2**30 or time.monotonic()-started>a.seconds-30
   stopping=reduce(torch.tensor(float(stop),device='cuda'),'max').item()
   if stopping:save();raise InterruptedError('owned STOP/time/storage boundary')
   with torch.no_grad():
    x=ppo.storage.observations[0];action=ppo.storage.actions[0];logp,_,_,mean,_=ppo.actor_critic.evaluate(x,None,action)
    error=(logp-ppo.storage.actions_log_prob[0].reshape(-1)).abs().max();error=float(reduce(error,'max'))
    if error>1e-4:raise ValueError('official behavior logprob mismatch '+str(error))
    state['behavior_logprob_error']=error
   result=original_update()
   if not np.isfinite(result).all():raise FloatingPointError('PPO loss not finite')
   return result
  ppo.update=update;ppo.save=save
  def log(locs):
   counts=reduce(torch.tensor([env.completed_episodes,env.successes,env.workspace_failures,env.reset_count],device='cuda',dtype=torch.float64),'sum')
   state.update(iteration=completed+locs['it']+1,elapsed_s=time.monotonic()-started,collection_s=locs['collection_time'],learning_s=locs['learn_time'],value_loss=float(locs['mean_value_loss']),surrogate_loss=float(locs['mean_surrogate_loss']),lr=ppo.step_size,completed_episodes=int(counts[0]),successes=int(counts[1]),workspace_failures=int(counts[2]),resets=int(counts[3]),reset_audits=env.reset_audits,source_mismatches=env.source_mismatches,
    tau_weight_norm=float(actor.net[0].weight[:,421:685].detach().norm()),task_weight_norm=float(actor.net[0].weight[:,411:421].detach().norm()),max_goal_error_m=env.max_goal_error,timings=env.timings,inference_audit=env.inference_audit,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
    eta_s=(locs['collection_time']+locs['learn_time'])*(a.updates-locs['it']-1))
   if state['iteration']==1 or state['iteration']%20==0:
    state['parameter_sync_error']=parameter_sync_error(ppo.actor_critic)
    if state['parameter_sync_error']!=0:raise ValueError('rank parameter mismatch')
   write(a.output/'progress.json',state)
   with (a.output/'curve.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
   print(json.dumps({k:state[k] for k in ['rank','iteration','collection_s','learning_s','successes','completed_episodes','resets','behavior_logprob_error','peak_cuda_mib']}),flush=True)
   if state['iteration']%100==0:save()
  ppo.log=log;ppo.run(a.updates,cfg['learn']['log_interval']);save()
  if a.reset_probe and (not any(0 in x['ids'] for x in env.reset_audits) or not any(1 in x['ids'] for x in env.reset_audits)):raise ValueError('partial reset fixture did not exercise both envs')
  state['status']='COMPLETED'
 except InterruptedError as error:state.update(status='STOPPED',error=repr(error))
 except BaseException as error:state.update(status='FAILED',error=repr(error));raise
 finally:
  if connection:
   try:send(connection,dict(op='close'));receive(connection)
   except Exception:pass
   connection.close()
  if env:state.update(reset_audits=env.reset_audits,source_mismatches=env.source_mismatches)
  write(a.output/'manifest.json',state);listener.close();a.socket.unlink(missing_ok=True)
  if a.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
