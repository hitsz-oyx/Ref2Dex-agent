"""Explicit v5→v6 expansion; normal resume never changes model/data contracts."""
import hashlib
import torch
from . import paths
from .contracts import SCHEMA, MIGRATION_SCHEMA
from .events import EVENT_ENCODING, initialize_event_model
from .optimizer_expansion import expand_event_optimizer
from ptv3_shared_wam_v5.contracts import SCHEMA as V5_SCHEMA
from ptv3_shared_wam_v5.model import SharedWAMV5
from ptv3_shared_wam_v5.training_contract import validate_resume, execution_contract


def validate_expansion_config(base, target):
    expected=dict(base,schema=SCHEMA,event_conditioning=True,
                  event_encoding=EVENT_ENCODING,event_dropout=.15)
    if target!=expected or base.get('event_conditioning'):
        raise ValueError('v5→v6 expansion may change only schema and declared event fields')


def migrate_checkpoint(saved, target, recipe_path, stats_path, recipe, config,
                       world_size, legacy_manifest=None):
    if saved.get('schema')!=V5_SCHEMA:
        raise ValueError('explicit expansion requires an original v5 checkpoint')
    validate_expansion_config(saved['config'],config)
    from pathlib import Path
    original_identity=hashlib.sha256(Path(recipe['parent_manifest']).read_bytes()+
                                     Path(recipe['stats']).read_bytes()).hexdigest()
    provenance=validate_resume(saved,saved['config'],original_identity,world_size,legacy_manifest)
    if len(saved.get('random_state',[]))!=world_size:
        raise ValueError('all original per-rank RNG states required')
    from ptv3_shared_wam_v5.storage import read_json
    base=SharedWAMV5(read_json(stats_path),saved['config'])
    weights=initialize_event_model(target,saved['model'])
    optimizer,report=expand_event_optimizer(base,target,saved['optimizer'],saved['config'])
    target_state=target.state_dict()
    if any(not torch.equal(v,target_state[k]) for k,v in saved['model'].items()):
        raise ValueError('original weights or normalization changed during expansion')
    result=dict(saved)
    result.update(schema=SCHEMA,config=config,model=target_state,optimizer=optimizer,
        data_identity=hashlib.sha256(Path(recipe_path).read_bytes()+Path(stats_path).read_bytes()).hexdigest(),
        execution_contract=execution_contract(config,world_size),selection={},
        migration=dict(schema=MIGRATION_SCHEMA,weights=weights,optimizer=report,
                       original_resume_binding=provenance,original_data_identity=original_identity,
                       original_weights_exact=True,old_selection='reset for the new conditioned DEV contract',
                       new_optimizer_state='fresh only for event_goal; original moments/steps/RNG retained'))
    return result
