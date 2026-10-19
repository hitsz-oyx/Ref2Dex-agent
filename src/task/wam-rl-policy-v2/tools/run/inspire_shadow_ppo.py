"""Run the existing Shadow official PPO on reference-free native Inspire states."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2];ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src')]


def write(path,value):
    path=Path(path);part=path.with_suffix(path.suffix+'.part');part.write_text(json.dumps(value,indent=2)+'\n');part.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--bc',type=Path);group.add_argument('--current-parent',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--socket',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=6);p.add_argument('--updates',type=int,default=4);p.add_argument('--seed',type=int,default=246)
    p.add_argument('--seconds',type=int,default=1800);p.add_argument('--resume',type=Path);p.add_argument('--reset-probe',action='store_true')
    p.add_argument('--evaluate',action='store_true');p.add_argument('--evaluation-controls',type=int,default=300)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--rendezvous');p.add_argument('--stop-file',type=Path);p.add_argument('--save-every',type=int,default=20)
    a=p.parse_args();a.output=a.output.resolve();a.cache=a.cache.resolve();a.socket=a.socket.resolve()
    if a.output.exists() or ROOT/'outputs/wam-rl-policy-v2' not in a.output.parents or not 0<=a.updates<=10000:raise ValueError('fresh bounded Task output required')
    if a.current_parent and (a.updates>4 or a.resume):raise ValueError('expanded current BC is engineering-only, max4 updates')
    if a.evaluate and (a.current_parent or a.updates or not 1<=a.evaluation_controls<=1200):raise ValueError('bounded evaluation needs a BC/PPO checkpoint and zero updates')
    if not 1<=a.world_size<=6 or not 0<=a.rank<a.world_size or a.save_every<1:raise ValueError('bounded distributed rank required')
    if a.world_size>1 and (not a.rendezvous or a.evaluate or a.current_parent):raise ValueError('distributed training needs full BC and private rendezvous')
    stop_file=a.stop_file.resolve() if a.stop_file else a.output.parent/'STOP'
    if ROOT/'tmp' not in a.socket.parents or a.socket.exists() or len(os.fsencode(a.socket))>=108:raise ValueError('private project socket required')
    a.output.mkdir(parents=True);a.socket.parent.mkdir(parents=True,exist_ok=True);os.chmod(a.socket.parent,0o700)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(a.socket.parent),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',REF2DEX_DEEP_OPT='all',PYTHONDONTWRITEBYTECODE='1')
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('bounded Inspire Shadow PPO')));signal.alarm(a.seconds)
    import numpy as np
    import torch
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    from wam_rl_policy_v2.inspire_shadow import CONTRACT,SCHEMA,CURRENT_DIM,expanded_model,initialize_scheduled
    from wam_rl_policy_v2.inspire_ppo_bridge import RemoteInspireEnv,make_shadow_ppo
    from wam_rl_policy_v2.wam_v4_runtime import V4Runtime
    from wam_rl_policy_v2.execution_identity import validate_execution
    from wam_rl_policy_v2.checkpoint_retention import prune_checkpoints
    from wam_rl_policy_v2.ppo import reduce,distributed_ready
    from wam_rl_policy_v2.scheduled_ppo_loop import parameter_sync_error
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
    meta=json.loads((a.cache/'manifest.json').read_text())
    if a.current_parent:
        parent=torch.load(a.current_parent,map_location='cpu',weights_only=False);model=expanded_model(True)
        model.x_scale[CURRENT_DIM:-1]=.05;initialize_scheduled(model,parent)
        packet=dict(schema=SCHEMA,model=model.state_dict(),config=dict(conditioning=CONTRACT,engineering_current_expansion=True),source_cache=parent['source_cache'])
    else:
        if meta.get('scheduled_status')!='COMPLETED':raise ValueError('complete full-data WAM conditioning cache required')
        packet=torch.load(a.bc,map_location='cpu',weights_only=False)
        if packet['config']['conditioning']!=CONTRACT or packet['config'].get('engineering_current_expansion'):
            raise ValueError('full-data scheduled BC with current contract required')
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.settimeout(300);listener.bind(str(a.socket));listener.listen(1)
    state=dict(status='WAITING_NATIVE',git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,torch=torch.__version__,gpu=a.gpu,seed=a.seed,conditioning=CONTRACT,
        policy_source=str(a.bc or a.current_parent),engineering_current_expansion=bool(a.current_parent),
        ppo_engine='pinned UniDexGrasp2 official, reused through Shadow prepare_official; no DExplore PPO',
        rank=a.rank,world_size=a.world_size,updates_requested=a.updates,optimizer_steps=0,iteration=0)
    write(a.output/'manifest.json',state);print(json.dumps(dict(status='READY',pid=os.getpid())),flush=True)
    connection=None;ppo=None;env=None;started=time.monotonic()
    try:
        connection,_=listener.accept();connection.settimeout(300);hello=receive(connection)
        if hello.get('schema')!='ref2dex.inspire-shadow-rpc.v1':raise ValueError('wrong native protocol')
        state['native_peer_pid']=peer_pid(connection)
        objects=[dict(object_code='dexplore/'+s.split('_')[1],scale=1) for s in hello['state']['sequences']]
        runtime=V4Runtime(meta['wam'],objects,a.cache,dense_manifest=a.cache/'surface-manifest.json',allow_geometry_replacement=True)
        if not a.current_parent:
            bound=packet['config']['wam_identity']
            if bound['wam']!=runtime.conditioning_identity or any(bound['geometry'].get(k)!=v for k,v in runtime.geometry_bindings.items()):raise ValueError('BC WAM/geometry binding differs')
        env=RemoteInspireEnv(connection,hello,a.cache,runtime,reset_probe=a.reset_probe)
        identity=dict(conditioning=CONTRACT,wam=runtime.conditioning_identity,geometry=runtime.geometry_bindings,
            task_goal='ordered full object SE3 path',task_reward='shadow_approach_with_ordered_path_tracking.v1',
            scheduler=env.inputs.schedulers[0].config.__dict__,source=str(a.cache),
            reset='explicit_indexed_reset_then_observation_then_policy.v1',native_controller=hello['audit']['controller'],native_dt=hello['audit']['native_dt_s'])
        if a.world_size>1:
            import torch.distributed as dist
            from datetime import timedelta
            torch.cuda.set_device(0)
            dist.init_process_group('nccl',init_method=a.rendezvous,rank=a.rank,world_size=a.world_size,timeout=timedelta(seconds=300))
            geometries=[None]*a.world_size;dist.all_gather_object(geometries,identity['geometry'])
            merged={}
            for entries in geometries:
                for key,value in entries.items():
                    if key in merged and merged[key]!=value:raise ValueError('rank geometry conflict')
                    merged[key]=value
            identity['geometry']=merged
        write(a.output/'execution-identity.json',identity)
        ppo,cfg=make_shadow_ppo(ROOT,a.output,env,packet,a.seed);write(a.output/'training-config.json',cfg)
        completed=0
        if a.resume:
            saved=torch.load(a.resume,map_location=runtime.device,weights_only=False)
            if saved.get('schema')!='ref2dex.inspire-shadow-official-ppo.v1':raise ValueError('Inspire Shadow PPO checkpoint required')
            validate_execution(saved.get('execution_identity'),identity,evaluating=a.evaluate)
            ppo.actor_critic.load_state_dict(saved['model']);ppo.optimizer.load_state_dict(saved['optimizer']);ppo.step_size=saved['step_size'];completed=saved['iteration']
        if distributed_ready():
            for tensor in ppo.actor_critic.state_dict().values():dist.broadcast(tensor,src=0)
            torch.manual_seed(a.seed+a.rank)
        state.update(status='RUNNING',envs=env.num_envs,iteration=completed)
        original_step=ppo.optimizer.step
        def optimizer_step(*args,**kwargs):
            result=original_step(*args,**kwargs);state['optimizer_steps']+=1;return result
        ppo.optimizer.step=optimizer_step
        original_update=ppo.update
        def update():
            stopping=reduce(torch.tensor(float(stop_file.exists()),device=runtime.device),'max').item()
            if stopping:
                save(None)
                if distributed_ready():dist.barrier()
                raise InterruptedError('owned PPO STOP at complete optimizer boundary')
            with torch.no_grad():
                x=ppo.storage.observations[0];action=ppo.storage.actions[0]
                logp,_,_,mean,_=ppo.actor_critic.evaluate(x,None,action)
                state['mean_path_error']=float((mean-ppo.actor_critic.act_inference(x)).abs().max())
                state['logprob_replay_error']=float((logp-ppo.storage.actions_log_prob[0].reshape(-1)).abs().max())
                if state['mean_path_error']>1e-6 or state['logprob_replay_error']>1e-5:raise ValueError('official action/logprob path mismatch')
            result=original_update()
            if not np.isfinite(result).all():raise FloatingPointError('official PPO loss')
            return result
        ppo.update=update
        def save(_):
            if a.rank:return
            iteration=state['iteration'];destination=a.output/f'official-iteration-{iteration}.pt'
            if destination.exists():return
            part=destination.with_suffix('.pt.part');torch.save(dict(schema='ref2dex.inspire-shadow-official-ppo.v1',
                model=ppo.actor_critic.state_dict(),optimizer=ppo.optimizer.state_dict(),iteration=iteration,
                step_size=ppo.step_size,config=cfg,execution_identity=identity,bc_source=str(a.bc or a.current_parent),
                engineering_current_expansion=bool(a.current_parent),world_size=a.world_size,environments_per_rank=env.num_envs,
                resume_semantics='model/optimizer; fresh native episodes'),part);part.replace(destination)
            prune_checkpoints(a.output,final_iteration=completed+a.updates)
        ppo.save=save
        def log(locs):
            metrics=reduce(torch.tensor([float(locs['mean_reward']),float(locs['mean_value_loss']),float(locs['mean_surrogate_loss'])],device=runtime.device))
            counts=reduce(torch.tensor([env.completed_episodes,env.successes,env.ever_held,env.final_held,env.workspace_failures],dtype=torch.float64,device=runtime.device),'sum')
            state.update(iteration=completed+locs['it']+1,elapsed_s=time.monotonic()-started,
                env_steps=(locs['it']+1)*env.num_envs*cfg['learn']['nsteps']*a.world_size,
                collection_s=locs['collection_time'],learning_s=locs['learn_time'],reward_mean=float(metrics[0]),
                value_loss=float(metrics[1]),surrogate_loss=float(metrics[2]),
                effective_std=float(ppo.actor_critic.log_std.exp().square().mean()),lr=ppo.step_size,
                input_timings=env.input_timings,tau_input_weight_norm=float(ppo.actor_critic.actor.net[0].weight[:,CURRENT_DIM:-1].detach().norm()),
                completed_episodes=int(counts[0]),ordered_path_successes=int(counts[1]),
                ever_held1s=int(counts[2]),final_held1s=int(counts[3]),workspace_failures=int(counts[4]),
                resets=env.reset_count,peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                eta_s=(locs['collection_time']+locs['learn_time'])*(a.updates-locs['it']-1))
            if locs['it']==0 or state['iteration']%a.save_every==0 or locs['it']+1==a.updates:
                state['model_sync_iteration']=state['iteration'];state['model_sync_error']=parameter_sync_error(ppo.actor_critic)
                if state['model_sync_error']!=0:raise ValueError('rank parameters diverged')
            with (a.output/'curve.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
            write(a.output/'progress.json',state);print(json.dumps(state),flush=True)
            if state['iteration']%500==0:save(None)
        ppo.log=log
        if a.evaluate:
            ppo.actor_critic.eval()
            # A zero-expanded selected BC cannot consume tau or valid at all.
            # Elide only the irrelevant neural prediction, preserving physical
            # workspace checks and all episode/scheduler transitions.
            elided=not a.resume and bool(torch.count_nonzero(ppo.actor_critic.actor.net[0].weight[:,CURRENT_DIM:])==0)
            env.inputs.predict_trajectory=not elided
            state.update(evaluation_policy='ppo' if a.resume else 'bc',wam_prediction_elided_for_zero_condition_weights=elided)
            observation=env.reset();returns=np.zeros(env.num_envs);first_done=np.zeros(env.num_envs,bool);results={}
            np.savez(a.output/'evaluation-initial-state.npz',**{k:np.asarray(v) for k,v in env.native.items()})
            def evaluation_report(tick):
                return dict(optimizer_steps=state['optimizer_steps'],policy_source=state['policy_source'],
                    evaluation_policy=state['evaluation_policy'],deterministic=True,
                    wam_prediction_elided_for_zero_condition_weights=elided,
                    scope='TRAIN first episodes, physical lift/hold proxy; not a contact-force certificate or unseen-object evaluation',
                    controls=tick,complete_first_episodes=len(results),unfinished_envs=np.flatnonzero(~first_done).tolist(),
                    envs=env.num_envs,records=results,returns=returns.tolist())
            for tick in range(a.evaluation_controls):
                with torch.inference_mode():action=ppo.actor_critic.act_inference(observation)
                observation,reward,done,info=env.step(action);returns[~first_done]+=reward.cpu().numpy()[~first_done]
                for i,result in info['episode_results'].items():
                    if not first_done[i]:results[i]=result;first_done[i]=True
                if (tick+1)%30==0:write(a.output/'evaluation-progress.json',evaluation_report(tick+1))
                if first_done.all():break
            if state['optimizer_steps']:raise ValueError('evaluation updated optimizer')
            write(a.output/'evaluation.json',evaluation_report(tick+1))
        else:
            ppo.run(num_learning_iterations=a.updates,log_interval=a.save_every)
            if state['optimizer_steps']!=a.updates*20:raise ValueError('Shadow optimizer cadence changed')
        state.update(status='COMPLETED',elapsed_s=time.monotonic()-started)
        write(a.output/'reset-audit.json',env.reset_audits)
        write(a.output/'learning-audit.json',dict(state,status='PASS'))
        env.rpc('close')
    except InterruptedError as error:
        state.update(status='STOPPED',reason=str(error),elapsed_s=time.monotonic()-started)
        if env is not None:env.rpc('close')
    except BaseException as error:
        state.update(status='FAILED',error=repr(error));raise
    finally:
        if ppo is not None:ppo.writer.close()
        if connection is not None:connection.close()
        if distributed_ready():torch.distributed.destroy_process_group()
        listener.close();a.socket.unlink(missing_ok=True);write(a.output/'manifest.json',state)


if __name__=='__main__':main()
