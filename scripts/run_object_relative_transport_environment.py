"""New closed-loop transport in the frozen native/P0 scaffold, no source actor calls."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def build_source():
    from scripts.run_self_trained_teacher_environment import build_source as original
    s=original()
    b=s.index('                teacher_observation=torch.zeros(');e=s.index('                for group,variant in ():',b)
    s=s[:b]+'''                transport_delta=torch.zeros_like(goal)
                reference_object=task.hoi_refs[task.data_id,task.ref_index,next_progress,106:109]
                dynamic=assignment==2;static=assignment==3
                transport_delta[dynamic,:2]=(task._target_states[dynamic,:2]-reference_object[dynamic,:2]).clamp(-.02,.02)
                transport_delta[static,:2]=offsets[static]
                goal[:,:2]+=transport_delta[:,:2]
                goal[:,:2]=torch.maximum(torch.minimum(goal[:,:2],task._dof_pos[:,:2]+task._pd_action_scale[:2].abs()),task._dof_pos[:,:2]-task._pd_action_scale[:2].abs())
                if not torch.equal(goal[:,2:],base_goal[:,2:]):raise ValueError('transport changes lift/rotation/fingers')
''' + s[e:]
    s=s.replace(",'teacher_observation','teacher_action','teacher_observation_valid','teacher_preprogress'",",'transport_delta'")
    s=s.replace(',teacher_observation=teacher_observation,teacher_action=teacher_action,teacher_observation_valid=teacher_observation_valid,teacher_preprogress=teacher_preprogress',',transport_delta=transport_delta')
    s=s.replace('no_source_actor_calls=False,source_actor_self_trained=True,source_actor_calls=201','no_source_actor_calls=True,source_actor_self_trained=True,source_actor_calls=0')
    s=s.replace('source_actor_first_tick=1','source_actor_first_tick=None,construction_actor_only=True,transport_contract=True')
    s=s.replace('native_reference_q=task.hoi_refs[:,0,:int(stops.max())+1,119:137].cpu().clone(),','native_reference_q=task.hoi_refs[:,0,:int(stops.max())+1,119:137].cpu().clone(),native_reference_object=task.hoi_refs[:,0,:int(stops.max())+1,106:109].cpu().clone(),')
    return s

def main():
    exec(compile(build_source(),str(ROOT/'scripts/run_continuous_critic_environment_v2.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_continuous_critic_environment_v2.py')})
if __name__=='__main__':main()
