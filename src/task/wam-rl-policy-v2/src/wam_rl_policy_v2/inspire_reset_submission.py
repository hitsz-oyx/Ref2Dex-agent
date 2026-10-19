"""Coalesce deferred Isaac setters; logical reset/FK observations stay immediate."""


def install_reset_submission(task_class, gymtorch):
    import torch
    if getattr(task_class,'_inspire_once_per_sim_reset',False):return
    original_pre=task_class.pre_physics_step
    def queue(task,env_ids):
        if not hasattr(task,'_inspire_pending_resets'):
            task._inspire_pending_resets=torch.zeros(task.num_envs,dtype=torch.bool,device=task.device)
            task._inspire_reset_setter_submissions=0
        task._inspire_pending_resets[env_ids]=True
        task.reset_buf[env_ids]=0;task._terminate_buf[env_ids]=0
    def before(task,actions):
        pending=getattr(task,'_inspire_pending_resets',None)
        if pending is not None:
            ids=pending.nonzero(as_tuple=False).flatten()
            if len(ids):
                humanoid=task._humanoid_actor_ids[ids]
                actors=[humanoid,task._tar_actor_ids[ids]]
                if task._motion_sampler is not None:actors.append(task._table_actor_ids[ids])
                # Persist both pointer owners through simulate, including when
                # several logical resets precede the very first control step.
                task._consequence_root_indices=torch.cat(actors).to(torch.int32).contiguous()
                task._consequence_dof_indices=humanoid.to(torch.int32).contiguous()
                task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),
                    gymtorch.unwrap_tensor(task._consequence_root_indices),len(task._consequence_root_indices))
                task.gym.set_dof_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._dof_state),
                    gymtorch.unwrap_tensor(task._consequence_dof_indices),len(task._consequence_dof_indices))
                pending.zero_();task._inspire_reset_setter_submissions+=1
        return original_pre(task,actions)
    task_class._reset_env_tensors=queue
    task_class.pre_physics_step=before
    task_class._inspire_once_per_sim_reset=True
