"""Narrow controller replacement in the unchanged fully audited native scaffold."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/run_natural_retention_environment.py').read_text()
    changes={
        'from src.task.CmResidual.natural_retention_feedback import groups,NaturalRetentionFeedback':'from src.task.CmResidual.rotation_retention_feedback import groups,RotationRetentionFeedback as NaturalRetentionFeedback',
        'feedback=NaturalRetentionFeedback(assignment,task._target_states[:,2].clone(),lift_starts[motion],decision_steps)':'feedback=NaturalRetentionFeedback(assignment,task._target_states[:,2].clone(),lift_starts[motion],decision_steps,task._pd_action_scale)',
        'feedback_arrest_translation=feedback.arrest_translation.cpu()':'rotation_anchor=feedback.rotation_anchor.cpu(),rotation_anchor_tick=feedback.rotation_anchor_tick.cpu()',
        'physical_metadata=dict(object_body_properties=body_properties':'physical_metadata=dict(native_dof_names=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles[0]),object_body_properties=body_properties',
    }
    for old,new in changes.items():
        if source.count(old)!=1:raise ValueError('native scaffold drift')
        source=source.replace(old,new)
    exec(compile(source,str(ROOT/'scripts/run_natural_retention_environment.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/run_natural_retention_environment.py')})

if __name__=='__main__':main()
