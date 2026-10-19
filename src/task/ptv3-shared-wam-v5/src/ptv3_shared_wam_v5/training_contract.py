"""Effective-batch binding and explicit migration of pre-contract checkpoints."""
EXECUTION_SCHEMA = 'ref2dex.wam-v5-effective-batch.v1'


def execution_contract(config, world_size):
    local = int(config['batch_size'])
    if local < 1 or world_size < 1:
        raise ValueError('positive effective batch and world size required')
    return dict(schema=EXECUTION_SCHEMA, effective_local_batch=local,
                effective_global_batch=local*world_size, world_size=world_size)


def validate_resume(saved, config, data_identity, world_size, legacy_manifest=None):
    expected = execution_contract(config, world_size)
    if (saved.get('config') != config or saved.get('data_identity') != data_identity
            or saved.get('world_size') != world_size):
        raise ValueError('v5 resume config/data/world-size mismatch')
    if saved.get('execution_contract') is not None:
        if saved['execution_contract'] != expected:
            raise ValueError('v5 resume effective batch mismatch')
        return dict(kind='bound-effective-batch', contract=expected)
    # Old checkpoints did not bind CLI overrides. Their original run manifest
    # did record the actual local batch, so require that provenance to migrate.
    if legacy_manifest is None:
        raise ValueError('legacy resume requires original run-manifest.json batch provenance')
    if (legacy_manifest.get('config') != saved['config']
            or legacy_manifest.get('data_identity') != data_identity
            or legacy_manifest.get('world_size') != world_size
            or legacy_manifest.get('batch_size') != expected['effective_local_batch']
            or legacy_manifest.get('engineering_smoke')):
        raise ValueError('legacy effective batch/config/data provenance mismatch')
    return dict(kind='legacy-run-manifest-verified', contract=expected,
                source_git_commit=legacy_manifest.get('git_commit'))


def loader_options(workers, timeout=120, context='spawn'):
    if workers < 0 or context not in ('spawn', 'forkserver') or timeout <= 0:
        raise ValueError('nonnegative workers and bounded spawn/forkserver loader required')
    return (dict(multiprocessing_context=context, persistent_workers=True,
                 prefetch_factor=2, timeout=timeout) if workers else {})


def restore_optimizer_state(optimizer, saved):
    """Torch2.0 leaves fused AdamW step on CPU after CPU-mapped recovery.

    Move only fused/capturable scalar steps to their parameter device, keeping
    their values and all momentum tensors. Unfused CPU-step semantics remain.
    """
    import torch
    optimizer.load_state_dict(saved)
    moved=0
    for group in optimizer.param_groups:
        if not (group.get('fused') or group.get('capturable')):
            continue
        for parameter in group['params']:
            state=optimizer.state.get(parameter,{})
            step=state.get('step')
            if isinstance(step,torch.Tensor) and step.device!=parameter.device:
                state['step']=step.to(parameter.device)
                moved+=1
    return dict(fused_or_capturable_steps_moved=moved)
