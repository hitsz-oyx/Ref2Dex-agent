"""Process-local reset compatibility with Isaac Gym's GPU tensor contract.

Do not refresh tensor caches after submitting deferred GPU setters. One root
setter submits every changed actor, with indices retained until simulation.
Reset observations use URDF FK at the submitted q; no hidden physics tick.
"""


def install_reset_patch():
    # Called only after Isaac Gym has imported, before constructing the task.
    from isaacgym import gymtorch
    import torch
    from env.tasks.base_dexplore_task import DexploreTask
    from .reset_kinematics import task_kinematics
    if getattr(DexploreTask, '_consequence_batched_reset', False):
        return

    def reset_tensors(task, env_ids):
        humanoid = task._humanoid_actor_ids[env_ids]
        actor_ids = [humanoid, task._tar_actor_ids[env_ids]]
        if task._motion_sampler is not None:
            actor_ids.append(task._table_actor_ids[env_ids])
        task._consequence_root_indices = torch.cat(actor_ids).to(torch.int32).contiguous()
        task._consequence_dof_indices = humanoid.to(torch.int32).contiguous()
        task.gym.set_actor_root_state_tensor_indexed(
            task.sim, gymtorch.unwrap_tensor(task._root_states),
            gymtorch.unwrap_tensor(task._consequence_root_indices),
            task._consequence_root_indices.numel())
        task.gym.set_dof_state_tensor_indexed(
            task.sim, gymtorch.unwrap_tensor(task._dof_state),
            gymtorch.unwrap_tensor(task._consequence_dof_indices),
            task._consequence_dof_indices.numel())
        task.reset_buf[env_ids] = 0
        task._terminate_buf[env_ids] = 0

    def reset_envs(task, env_ids):
        task._reset_default_env_ids = []
        task._reset_ref_env_ids = []
        if len(env_ids):
            # Native Inspire reset writes every humanoid root, even for a subset.
            # Keep unrelated environment caches exactly as measured.
            previous_roots = task._humanoid_root_states.clone()
            task._reset_actors(env_ids)
            selected_roots = task._humanoid_root_states[env_ids].clone()
            task._humanoid_root_states[:] = previous_roots
            task._humanoid_root_states[env_ids] = selected_roots
            fk = task_kinematics(task)
            bodies = task._rigid_body_state.view(task.num_envs, -1, 13)
            bodies[env_ids, :task.num_bodies] = fk.states(
                task._dof_pos[env_ids], task._dof_vel[env_ids],
                task._humanoid_root_states[env_ids])
            bodies[env_ids, task.num_bodies] = task._table_states[env_ids]
            bodies[env_ids, task.num_bodies+1] = task._target_states[env_ids]
            # Old contact caches cannot be an observation of the new reset.
            task._contact_forces[env_ids] = 0
            task._tar_contact_forces[env_ids] = 0
            task._reset_env_tensors(env_ids)
            task._compute_observations(env_ids)

    DexploreTask._reset_env_tensors = reset_tensors
    DexploreTask._reset_envs = reset_envs
    DexploreTask._consequence_batched_reset = True
