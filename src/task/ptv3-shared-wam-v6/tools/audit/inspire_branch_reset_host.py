"""Audit-only continuous wrist reset normalization; production host unchanged."""
from pathlib import Path
import difflib,json,sys
ROOT=Path(__file__).resolve().parents[5]
source=ROOT/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'
out=Path(sys.argv[sys.argv.index('--output')+1]);evidence=ROOT/'outputs/ptv3-shared-wam-v6'/('branch-host-'+out.name);evidence.mkdir(parents=True,exist_ok=False)
raw=source.read_text();anchor='    install_reset_submission(DexploreTask,gymtorch)\n'
patch='''    install_reset_submission(DexploreTask,gymtorch)
    from env.tasks.dexplore_inspire import Dexplore_Inspire
    original_set_state=Dexplore_Inspire._set_env_state
    def normalized_set_state(task,env_ids,dof_pos,dof_vel):
        original_set_state(task,env_ids,dof_pos,dof_vel)
        task._dof_pos[env_ids,3:6]=torch.remainder(task._dof_pos[env_ids,3:6]+np.pi,2*np.pi)-np.pi
    Dexplore_Inspire._set_env_state=normalized_set_state
'''
if raw.count(anchor)!=1:raise ValueError('host reset anchor changed')
modified=raw.replace(anchor,patch)
guard='not 2<=a.envs<=512'
if modified.count(guard)!=1:raise ValueError('host env count guard changed')
modified=modified.replace(guard,'not 1<=a.envs<=512')
anchor2='            env=create();task=env.task\n'
patch2='''            env=create();task=env.task
            props=task.gym.get_actor_dof_properties(task.envs[0],task.humanoid_handles[0])
            audit['audit_only_reset']='continuous wrist modulo 2pi; no production source changed'
            audit['native_dof_names']=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles[0])
            audit['native_dof_properties']={key:props[key].tolist() for key in ('hasLimits','lower','upper')}
'''
if modified.count(anchor2)!=1:raise ValueError('host create anchor changed')
modified=modified.replace(anchor2,patch2)
(evidence/'original.py').write_text(raw);(evidence/'instrumented.py').write_text(modified);(evidence/'diff.patch').write_text(''.join(difflib.unified_diff(raw.splitlines(True),modified.splitlines(True),fromfile=str(source),tofile='instrumented.py')))
exec(compile(modified,str(evidence/'instrumented.py'),'exec'),dict(__name__='__main__',__file__=str(source)))
