"""Explicit optimizer-preserving recipe migration; distinct from exact resume."""
import math

from oakink_wm.distributed import restore_rng

ENTRY = 'tools/run/train_oakink2_pointworld_ddp.py'
SELF = 'src/oakink_wm/continuation.py'
# Frozen entry of the successfully completed original three-rank run.
LEGACY_ENTRY_SHA256 = '54e76b43d242ce1bcef320e2bd5b6a024a52c0d6cba394e3afcf32d687cfd670'


def continue_optimizer(state, model, optimizer, config, identity, rank, world):
    previous = state['identity']
    current_sources = dict(identity['implementation_sources'])
    legacy_sources = dict(current_sources)
    legacy_sources.pop(SELF)
    legacy_sources[ENTRY] = LEGACY_ENTRY_SHA256
    if (state.get('checkpoint_kind') != 'pointworld-temporal.ddp.v1'
            or state.get('world_size') not in (2, 3) or world != 2
            or state['dataset_hash'] != identity['dataset_hash']
            or previous['arm'] != identity['arm'] or identity['arm'] != 'action'
            or previous['stats_sha256'] != identity['stats_sha256']
            or previous['vendor_sources'] != identity['vendor_sources']
            or previous.get('performance_backend', 'reference') != identity.get('performance_backend', 'reference')
            or previous['implementation_sources'] not in (current_sources, legacy_sources)):
        raise ValueError('continuation checkpoint data/source/backend/world mismatch')
    recipe_keys = {'seed', 'accumulation', 'updates', 'warmup_updates', 'learning_rate',
                   'learning_rate_schedule', 'group_seconds'}
    for key in set(config) | set(state['config']):
        if key not in recipe_keys and config.get(key) != state['config'].get(key):
            raise ValueError('continuation changes fixed model/data/batch semantics: '+key)
    groups = state['optimizer']['param_groups']
    rates = {float(g['lr']) for g in groups}
    if (config.get('learning_rate_schedule') != 'constant' or config['warmup_updates'] != 0
            or len(rates) != 1 or not math.isfinite(config['learning_rate'])
            or config['learning_rate'] <= 0 or config['learning_rate'] != next(iter(rates))):
        raise ValueError('continuation must keep the saved endpoint learning rate')
    moments = state['optimizer']['state']
    steps = [int(v['step']) for v in moments.values()]
    if (not steps or min(steps) < 1 or len(moments) != sum(len(g['params']) for g in groups)
            or any(not {'step', 'exp_avg', 'exp_avg_sq'} <= set(v) for v in moments.values())):
        raise ValueError('continuation requires complete trained AdamW moments')
    if len(state['rank_rngs']) != state['world_size'] or not 0 <= rank < world:
        raise ValueError('invalid parent rank RNG state')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rank_rngs'][rank])
    return dict(parent_step=state['step'], parent_world_size=state['world_size'],
                parent_optimizer_steps=[min(steps), max(steps)], weights_only=False,
                optimizer_reset=False, stage_step_reset=True, draw_reset=True,
                schedule='constant saved endpoint learning rate', learning_rate=config['learning_rate'],
                rank_rng_migration='retain parent ranks0/1; discard any other rank',
                exact_rng_continuation=False)
