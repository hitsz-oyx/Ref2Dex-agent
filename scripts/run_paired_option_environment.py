"""Joint held requests after a common P0 prefix, with initial-only pairing."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build_source():
    source = (ROOT / 'scripts/run_continuous_critic_environment_v2.py').read_text()
    changes = {
        '    from isaacgym import gymapi,gymtorch':
        '    from src.task.CmResidual.paired_option_opportunity import matched_inputs\n    from isaacgym import gymapi,gymtorch',
        'offsets=placement_offsets(768,args.eval_seed).to(device);before_placement=task._target_states.clone()':
        'offsets,option_cluster,option_raw12=matched_inputs(motion,assignment,args.eval_seed);before_placement=task._target_states.clone()',
        "'normalized_context','request_noise')}":
        "'normalized_context','request_noise','hand_root','hand_body_position','hand_body_quaternion')}",
        'normalized_context=normalized_context,request_noise=request_noise)':
        'normalized_context=normalized_context,request_noise=request_noise,hand_root=task._humanoid_root_states,hand_body_position=task._rigid_body_pos[:,task._contact_body_ids],hand_body_quaternion=task._rigid_body_rot[:,task._contact_body_ids])',
        'physical_metadata=dict(native_dof_names=names,':
        'physical_metadata=dict(native_body_names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0]),contact_body_ids=task._contact_body_ids.cpu().tolist(),hand_body_com=[[p.com.x,p.com.y,p.com.z] for p in task.gym.get_actor_rigid_body_properties(task.envs[0],task.humanoid_handles[0])],native_dof_names=names,',
        'policy_group=assignment.cpu(),continuous_schema=CONTINUOUS_SCHEMA':
        'option_cluster=option_cluster.cpu(),option_raw12=option_raw12.cpu(),option_draw_seed=args.eval_seed+18000,paired_initial_offsets=True,policy_group=assignment.cpu(),continuous_schema=CONTINUOUS_SCHEMA',
        'result=dict(deterministic=args.deterministic,':
        'result=dict(paired_joint_option=True,request_policy_used=False,option_sampling_std=1.,option_held_until_task_deadline=True,deterministic=args.deterministic,',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    begin = source.index("                for group,variant", source.index('for tick in range(202)'))
    end = source.index('                current_root=', begin)
    source = source[:begin] + '''                for group in (1,2,3):
                    selected=assignment==group
                    active=(tick>=decision_steps[selected])&(tick<stops[motion[selected]]+30)
                    raw=torch.where(active[:,None],option_raw12[selected],torch.zeros_like(option_raw12[selected]))
                    target=executable_target(base_goal[selected],raw,task._dof_pos[selected],task.dof_limits_lower,task.dof_limits_upper,task._pd_action_scale)
                    goal[selected]=target;request[selected]=raw
                    executed_action12[selected]=(target[:,INDEPENDENT]-task._dof_pos[selected][:,INDEPENDENT])/scale12
''' + source[end:]
    return source


def main():
    exec(compile(build_source(), str(ROOT / 'scripts/run_continuous_critic_environment_v2.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT / 'scripts/run_continuous_critic_environment_v2.py')})


if __name__ == '__main__':
    main()
