"""Explicit GPU topology transitions; ordinary resume remains strict."""
from .contracts import SCHEMA
from ptv3_shared_wam_v5.training_contract import execution_contract, validate_resume


def transition_checkpoint(saved, config, data_identity):
    if saved.get('schema') != SCHEMA or saved.get('world_size') != 4:
        raise ValueError('explicit transition requires an original world4 v6 checkpoint')
    validate_resume(saved, saved['config'], data_identity, 4)
    expected = dict(saved['config'], batch_size=config['batch_size'])
    if config != expected or not 1 <= config['batch_size'] <= saved['config']['batch_size']:
        raise ValueError('topology transition may change only local batch within original capacity')
    states = saved.get('random_state', [])
    if len(states) != 4:
        raise ValueError('four original RNG payloads required')
    result = dict(saved)
    result.update(config=dict(config), world_size=3,
                  execution_contract=execution_contract(config, 3), random_state=list(states[:3]),
                  topology_transition=dict(schema='ref2dex.wam-v6-topology-transition.v1',
                      original_world_size=4, target_world_size=3,
                      original_execution_contract=saved['execution_contract'],
                      target_execution_contract=execution_contract(config, 3),
                      retained_rng_ranks=[0, 1, 2], omitted_rng_rank=3,
                      original_step=saved['step'],
                      sampling_note='New batch/world changes future draws and gradient noise; not bitwise world4 replay',
                      state_policy='Model, normalizers, optimizer moments/steps, selection, elapsed and warmup retained'))
    validate_resume(result, config, data_identity, 3)
    return result


def restore_four_gpu_checkpoint(saved, config, data_identity, new_rank_rng, new_rank_seed):
    if saved.get('schema') != SCHEMA or saved.get('world_size') != 3:
        raise ValueError('four-GPU restoration requires a world3 v6 checkpoint')
    validate_resume(saved, saved['config'], data_identity, 3)
    previous=saved.get('topology_transition') or {}
    original_batch=previous.get('original_execution_contract',{}).get('effective_local_batch')
    if previous.get('original_world_size')!=4 or previous.get('target_world_size')!=3 or not original_batch:
        raise ValueError('original world4 capacity provenance required')
    expected=dict(saved['config'],batch_size=config['batch_size'])
    if config!=expected or not 1<=config['batch_size']<=original_batch:
        raise ValueError('four-GPU restoration may change only batch within original capacity')
    states=saved.get('random_state',[])
    if len(states)!=3 or set(new_rank_rng)!=set(('python','numpy','torch','cuda')) or new_rank_rng['cuda'] is None:
        raise ValueError('three retained RNG states and a fresh complete rank3 state required')
    result=dict(saved)
    result.update(config=dict(config),world_size=4,execution_contract=execution_contract(config,4),
        random_state=list(states)+[new_rank_rng],topology_transition=dict(
            schema='ref2dex.wam-v6-topology-transition.v1',
            original_world_size=3,target_world_size=4,original_step=saved['step'],
            original_execution_contract=saved['execution_contract'],
            target_execution_contract=execution_contract(config,4),
            retained_rng_ranks=[0,1,2],initialized_rng_ranks=[3],new_rank_seed=int(new_rank_seed),
            previous_transition=previous,
            sampling_note='World3 to world4 and batch change future draws; not bitwise replay',
            state_policy='Model, normalizers, optimizer moments/steps, selection, elapsed and warmup retained'))
    validate_resume(result,config,data_identity,4)
    return result
