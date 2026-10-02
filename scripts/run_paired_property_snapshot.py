"""Read all SDK actor properties after initial reset; no physics step."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    from scripts.run_paired_option_environment import build_source
    source = build_source()
    marker = "            (args.run_dir/'physical_metadata.json').write_text(json.dumps(physical_metadata,indent=2)+'\\n')"
    assert source.count(marker) == 1
    addition = '''
            records=[]
            for env_id in range(768):
                origin=task.gym.get_env_origin(task.envs[env_id])
                props=task.gym.get_actor_dof_properties(task.envs[env_id],task.humanoid_handles[env_id])
                def bodies(handle):
                    result=[]
                    for body in task.gym.get_actor_rigid_body_properties(task.envs[env_id],handle):
                        result.append(dict(mass=float(body.mass),flags=int(body.flags),com=[body.com.x,body.com.y,body.com.z],inertia=[[getattr(getattr(body.inertia,x),y) for y in ('x','y','z')] for x in ('x','y','z')]))
                    return result
                def shapes(handle):
                    result=[]
                    for shape in task.gym.get_actor_rigid_shape_properties(task.envs[env_id],handle):
                        result.append({key:float(getattr(shape,key)) for key in ('friction','rolling_friction','torsion_friction','restitution','compliance','thickness')})
                    return result
                records.append(dict(environment=env_id,motion=int(motion[env_id]),arm=int(assignment[env_id]),cluster=int(option_cluster[env_id]),origin=[origin.x,origin.y,origin.z],dof={name:props[name].tolist() for name in props.dtype.names},hand_bodies=bodies(task.humanoid_handles[env_id]),object_bodies=bodies(task._target_handles[env_id]),hand_shapes=shapes(task.humanoid_handles[env_id]),object_shapes=shapes(task._target_handles[env_id])))
            (args.run_dir/'sdk_properties.json').write_text(json.dumps(records,indent=2)+'\\n')
            torch.save(dict(q=task._dof_pos.cpu().clone(),dq=task._dof_vel.cpu().clone(),object_root=task._target_states.cpu().clone(),hand_root=task._humanoid_root_states.cpu().clone(),table_root=task._table_states.cpu().clone()),args.run_dir/'reset_states.pt')
            (args.run_dir/'results.json').write_text(json.dumps(dict(run_status='COMPLETED',engineering_only=True,physics_steps=0,model_forward_calls=0,source_actor_calls=0,metadata_records=len(records)))+'\\n')
            print('SDK_PROPERTY_SNAPSHOT_COMPLETE no physics steps or model forwards',flush=True)
            return
'''
    source = source.replace(marker, marker + addition)
    exec(compile(source, str(ROOT / 'scripts/run_continuous_critic_environment_v2.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(ROOT / 'scripts/run_continuous_critic_environment_v2.py')})


if __name__ == '__main__':
    # ROOT must be available before importing the generated controller.
    import sys
    sys.path.insert(0, str(ROOT))
    main()
