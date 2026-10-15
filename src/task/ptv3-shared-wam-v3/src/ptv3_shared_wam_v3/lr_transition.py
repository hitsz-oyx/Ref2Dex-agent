"""Explicit LR-only continuation; all data/model/execution contracts stay fixed."""
import math

LR_FIELDS = ('geometry_lr', 'shared_lr')


def validate_lr_resume(parent, config, stats, world):
    before = parent['config']
    changed = {key for key in set(before) | set(config) if before.get(key) != config.get(key)}
    if not changed or not changed <= set(LR_FIELDS):
        raise ValueError('LR continuation permits only a learning-rate decrease')
    if (parent['stats'] != stats or parent['provenance']['world_size'] != world
            or len(parent['rank_rng']) != world):
        raise ValueError('LR continuation requires identical statistics/recipe/world/rank RNG')
    for key in LR_FIELDS:
        if not math.isfinite(config[key]) or not 0 < config[key] <= before[key]:
            raise ValueError('LR must be finite, positive, and no larger than its parent')
    return dict(schema='ref2dex.lr-only-resume.v1', parent_step=parent['step'],
                before={k:before[k] for k in LR_FIELDS}, after={k:config[k] for k in LR_FIELDS},
                preserved='model, optimizer moments/steps, per-rank RNG, recipe, stats, panel, global step')


def apply_learning_rates(optimizer, config, step):
    scale = min(1., step / max(1, config.get('warmup_updates', 1)))
    for group in optimizer.param_groups:
        if group['name'] not in ('joint_geometry', 'shared_and_heads'):
            raise ValueError('unknown optimizer group')
        group['lr'] = config['geometry_lr' if group['name']=='joint_geometry' else 'shared_lr'] * scale
    return {group['name']:group['lr'] for group in optimizer.param_groups}
