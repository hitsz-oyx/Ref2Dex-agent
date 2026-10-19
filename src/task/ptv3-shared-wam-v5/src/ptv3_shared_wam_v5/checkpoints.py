"""Checkpoint scheduling and retention, restricted to the current owned run."""
def checkpoint_due(step,last,config):
    interval=config.get('checkpoint_interval',1000)
    milestone=config.get('checkpoint_milestone_interval',0)
    return (step==last or (step==1 and config.get('save_first_step',False))
            or (interval>0 and step%interval==0) or (milestone>0 and step%milestone==0))


def retained_steps(steps,config):
    keep=int(config.get('checkpoint_keep_recent',2))
    if keep<1:raise ValueError('at least one recent checkpoint required')
    recent=set(sorted(steps)[-keep:]);milestone=config.get('checkpoint_milestone_interval',0)
    return recent|{step for step in steps if (step==1 and config.get('save_first_step',False))
                    or (milestone>0 and step%milestone==0)}
