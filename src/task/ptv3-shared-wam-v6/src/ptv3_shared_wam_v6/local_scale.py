"""Explicit local topology/data expansion retaining learned optimizer state."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from .local_subset import LOCAL_SCOPES,EXPANDED_SCOPE,index_identity
from .recipe import validate_recipe
from .contracts import SCHEMA
from ptv3_shared_wam_v5.training_contract import execution_contract,validate_resume
from ptv3_shared_wam_v5.storage import read_json


def data_identity(manifest,stats):
    return hashlib.sha256(Path(manifest).read_bytes()+Path(stats).read_bytes()).hexdigest()


def transition_local(saved,config,old_recipe_path,new_recipe_path,world,new_rng):
    old=validate_recipe(old_recipe_path);new=validate_recipe(new_recipe_path)
    old_identity=data_identity(old_recipe_path,old['stats']);new_identity=data_identity(new_recipe_path,new['stats'])
    if saved.get('schema')!=SCHEMA or saved.get('world_size') not in (1,3):raise ValueError('local v6 checkpoint required')
    validate_resume(saved,saved['config'],old_identity,saved['world_size'])
    if old.get('local_subset_scope') not in LOCAL_SCOPES or new.get('local_subset_scope')!=EXPANDED_SCOPE:
        raise ValueError('explicit local expansion required')
    if old['stats_identity']!=new['stats_identity']:raise ValueError('scale-up must retain exact normalizers')
    before=read_json(old['parent_manifest']);after=read_json(new['parent_manifest'])
    if before['sequences']!=after['sequences'] or before['cache_bindings']!=after['cache_bindings']:
        raise ValueError('only TRAIN window expansion over unchanged caches allowed')
    for key in ('indices','strata'):
        if index_identity(before[key]['dev'])!=index_identity(after[key]['dev']):
            raise ValueError('DEV panel must remain exactly unchanged')
    a=np.ascontiguousarray(np.load(before['indices']['train']),dtype=np.int32)
    b=np.ascontiguousarray(np.load(after['indices']['train']),dtype=np.int32)
    dtype=np.dtype([('sequence','<i4'),('slot','<i4'),('tick','<i4')])
    if not np.isin(a.view(dtype).ravel(),b.view(dtype).ravel()).all():
        raise ValueError('expanded TRAIN must retain every prior window')
    allowed={'local_subset_scope','batch_size','updates','validation_interval','checkpoint_interval',
             'validation_initial_step','validation_before_training','log_interval'}
    changed={k for k in set(config)|set(saved['config']) if config.get(k)!=saved['config'].get(k)}
    if changed-allowed or world not in (1,3) or not 1<=config['batch_size']<=128:
        raise ValueError('unapproved optimizer/model/data/normalization change')
    states=saved['random_state']
    if len(states)!=saved['world_size'] or len(new_rng)!=max(0,world-len(states)):
        raise ValueError('complete rank RNG payloads required')
    for state in new_rng:
        if set(state)!=set(('python','numpy','torch','cuda')) or state['cuda'] is None:
            raise ValueError('complete new CUDA rank RNG required')
    result=dict(saved,config=dict(config),world_size=world,data_identity=new_identity,
        execution_contract=execution_contract(config,world),random_state=list(states[:world])+list(new_rng),
        topology_transition=dict(schema='ref2dex.wam-v6-local-scale-transition.v1',
            previous_transition=saved.get('topology_transition'),parent_recipe=str(Path(old_recipe_path).resolve()),
            target_recipe=str(Path(new_recipe_path).resolve()),original_step=saved['step'],
            original_execution_contract=saved['execution_contract'],target_execution_contract=execution_contract(config,world),
            original_data_identity=old_identity,target_data_identity=new_identity,
            state_policy='retain model, optimizer moments/steps, normalizers, selection, elapsed and rank0 RNG; explicit new ranks',
            sampling_note='larger TRAIN pool and global batch change future draws; not bitwise single-GPU continuation'))
    validate_resume(result,config,new_identity,world)
    return result
