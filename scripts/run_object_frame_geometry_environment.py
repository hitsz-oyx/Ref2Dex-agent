"""Capture SDK contact-body poses/root without altering the native controller."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/run_continuous_critic_environment_v2.py').read_text()
    changes={
        "'normalized_context','request_noise')}":"'normalized_context','request_noise','hand_root','hand_body_position','hand_body_quaternion')}",
        'normalized_context=normalized_context,request_noise=request_noise)':'normalized_context=normalized_context,request_noise=request_noise,hand_root=task._humanoid_root_states,hand_body_position=task._rigid_body_pos[:,task._contact_body_ids],hand_body_quaternion=task._rigid_body_rot[:,task._contact_body_ids])',
        'physical_metadata=dict(native_dof_names=names,':'physical_metadata=dict(native_body_names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0]),contact_body_ids=task._contact_body_ids.cpu().tolist(),hand_body_com=[[p.com.x,p.com.y,p.com.z] for p in task.gym.get_actor_rigid_body_properties(task.envs[0],task.humanoid_handles[0])],native_dof_names=names,',
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('exact native capture scaffold drift '+old)
        source=source.replace(old,new)
    exec(compile(source,str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})

if __name__=='__main__':main()
