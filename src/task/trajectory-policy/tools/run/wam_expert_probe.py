"""Training-time reference-conditioned expert labels on actual learner observations."""


def expert_on_rows(player, observation, verify_official=False):
    """Each actual row uses64 identical copies; expose mean before player clipping.

    Calling the player's own preprocessing/model preserves its external normalizer.
    Native Inspire action bounds must be[-1,1], so explicit clamp equals rescaling.
    """
    import torch
    raw=observation['obs']
    if player.is_rnn or not player.has_batch_dimension or raw.ndim!=2:
        raise ValueError('nonrecurrent batched native observations required')
    if not ((player.actions_low==-1).all() and (player.actions_high==1).all()):
        raise ValueError('native normalized action bounds must be[-1,1]')
    original_clip=player.clip_actions
    if not original_clip:
        raise ValueError('qualified official player must clip actions')
    means=[]
    try:
        player.clip_actions=False
        for row in raw:
            padded=dict(observation,obs=row[None].repeat(64,1))
            means.append(player.get_action(padded,True)[:1].clone())
    finally:
        player.clip_actions=original_clip
    mean=torch.cat(means)
    if not torch.isfinite(mean).all():
        raise FloatingPointError('nonfinite queried expert mean')
    applied=mean.clamp(-1,1)
    if verify_official:
        for index,row in enumerate(raw):
            official=player.get_action(dict(observation,obs=row[None].repeat(64,1)),True)[:1]
            if not torch.equal(official,applied[index:index+1]):
                raise ValueError('explicit expert clamp differs from official player')
    return mean,applied,(mean-applied).abs().amax(-1)>1e-6


def paired_prefix_error(current):
    """Before intervention, equality must be measured, never reset/forked into place."""
    return {key:float((value[2].float()-value[3].float()).abs().max()) for key,value in current.items()}
