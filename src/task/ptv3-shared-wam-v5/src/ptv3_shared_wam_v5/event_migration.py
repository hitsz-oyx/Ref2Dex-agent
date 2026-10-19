"""Explicit named AdamW expansion, separate from strict same-contract resume."""
import copy


def _group_names(model, config):
    names = {id(p): name for name, p in model.named_parameters()}
    return [(g['name'], [names[id(p)] for p in g['params']])
            for g in model.parameter_groups(config['geometry_lr'], config['shared_lr'])]


def expand_event_optimizer(base, target, saved, config):
    """Keep original moments/steps/group settings, leave only event params fresh.

    Parameter IDs in torch optimizer checkpoints are positional, not names.
    Reconstruct the original group order from the original architecture first;
    never zip old states directly to the expanded model's parameter iterator.
    Returned tensors alias the saved state; this function does not mutate it.
    """
    old_groups, new_groups = _group_names(base, config), _group_names(target, config)
    old_names = {name for _, names in old_groups for name in names}
    new_names = {name for _, names in new_groups for name in names}
    added = new_names-old_names
    if (not added or old_names-new_names or any(not n.startswith('event_goal.') for n in added)):
        raise ValueError('only new event_goal parameters may be added')
    if len(saved['param_groups']) != len(old_groups) or len(old_groups) != len(new_groups):
        raise ValueError('original optimizer group count changed')
    old_id_by_name, new_id_by_name, groups = {}, {}, []
    cursor = 0
    for group, (old_group, old), (new_group, new) in zip(saved['param_groups'], old_groups, new_groups):
        if group.get('name') != old_group or old_group != new_group or len(group['params']) != len(old):
            raise ValueError('original optimizer parameter-group identity/order mismatch')
        if [n for n in new if n in old_names] != old:
            raise ValueError('original parameters moved between optimizer groups')
        old_id_by_name.update(zip(old, group['params']))
        ids = list(range(cursor, cursor+len(new))); cursor += len(new)
        new_id_by_name.update(zip(new, ids))
        settings = copy.deepcopy({k: v for k, v in group.items() if k != 'params'})
        groups.append(dict(settings, params=ids))
    if set(saved['state'])-set(old_id_by_name.values()):
        raise ValueError('unmapped original optimizer state')
    states = {new_id_by_name[name]: saved['state'][old_id]
              for name, old_id in old_id_by_name.items() if old_id in saved['state']}
    report = dict(kind='named-event-optimizer-expansion', original_parameters=len(old_names),
                  preserved_states=len(states), new_parameters=sorted(added),
                  new_optimizer_state='empty; initialize on first event-conditioned gradient',
                  original_steps_moments_and_group_settings='preserved without modification')
    return dict(state=states, param_groups=groups), report
