"""Fixed actor deployment with actual native hand/table filter contracts."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def build_source():
    from scripts.run_trained_option_environment import build_source as original
    source=original()
    marker='            physical_metadata=dict('
    assert source.count(marker)==1
    block='''            names_sdk=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0])
            ranges_sdk=task.gym.get_actor_rigid_body_shape_indices(task.envs[0],task.humanoid_handles[0])
            owners=[None]*len(task.gym.get_actor_rigid_shape_properties(task.envs[0],task.humanoid_handles[0]))
            for name_sdk,span in zip(names_sdk,ranges_sdk):
                start=int(span.start if hasattr(span,'start') else span['start'])
                count=int(span.count if hasattr(span,'count') else span['count'])
                for index in range(start,start+count):
                    if owners[index] is not None:raise ValueError('multiple shape owners')
                    owners[index]=name_sdk
            if any(owner is None for owner in owners):raise ValueError('missing shape owner')
            expected_filters=[3 if ('thumb' in name and 'distal' in name) or ('thumb' not in name and 'intermediate' in name) else 2 for name in owners]
            hand_filters=[];table_filters=[]
            for env_index in range(task.num_envs):
                hand_filters.append([int(p.filter) for p in task.gym.get_actor_rigid_shape_properties(task.envs[env_index],task.humanoid_handles[env_index])])
                table_filters.append([int(p.filter) for p in task.gym.get_actor_rigid_shape_properties(task.envs[env_index],task._table_handles[env_index])])
                if hand_filters[-1]!=expected_filters or any(value!=1 for value in table_filters[-1]):raise ValueError('actual native filter contract')
'''
    source=source.replace(marker,block+marker)
    source=source.replace('physical_metadata=dict(', 'physical_metadata=dict(native_shape_ownership=owners,hand_shape_filters=hand_filters,table_shape_filters=table_filters,correct_shape_ownership_verified=True,')
    return source


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})
