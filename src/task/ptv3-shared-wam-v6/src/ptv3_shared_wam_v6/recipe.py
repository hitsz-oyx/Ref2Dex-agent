"""Admission binds immutable event sidecars to the original windows and stats."""
from pathlib import Path
from . import paths
from .contracts import DATA_SCHEMA
from .events import EVENT_ENCODING
from ptv3_shared_wam_v5.cache_contract import metadata_identity
from ptv3_shared_wam_v5.storage import read_json, load_array
from ptv3_shared_wam_v5.event_labels import EVENT_LABEL_SCHEMA
from ptv3_shared_wam_v5.contracts import STATS_SCHEMA


def validate_recipe(path, stats=None):
    path=Path(path); recipe=read_json(path)
    if (recipe.get('schema')!=DATA_SCHEMA or recipe.get('status')!='FROZEN'
            or recipe.get('training_allowed') is not True
            or recipe.get('event_encoding')!=EVENT_ENCODING):
        raise ValueError('explicit admitted v6 event recipe required')
    for key in ('parent_manifest','event_manifest','stats'):
        if metadata_identity(recipe[key])!=recipe[key+'_identity']:
            raise ValueError('v6 upstream changed: '+key)
    parent=read_json(recipe['parent_manifest']); events=read_json(recipe['event_manifest'])
    normalizer=read_json(recipe['stats'])
    from .local_subset import LOCAL_SCOPES, validate_local_parent
    if recipe.get('local_subset_scope') in LOCAL_SCOPES:
        validate_local_parent(parent, recipe, normalizer)
    elif (parent.get('status')!='FROZEN' or not parent.get('training_allowed')
            or parent.get('engineering_subset') or parent.get('pending_sequences',0)
            or sorted(parent['source_ids'])!=[0,1,2,4,5,6,8,9,10]):
        raise ValueError('complete nine-source v5 parent required')
    keys=[r['key'] for r in parent['sequences']]; event_keys=[r['key'] for r in events['sequences']]
    if (events.get('schema')!=EVENT_LABEL_SCHEMA or events.get('status')!='COMPLETED'
            or events['parent_manifest_identity']!=recipe['parent_manifest_identity']
            or len(set(event_keys))!=len(keys) or set(event_keys)!=set(keys)):
        raise ValueError('complete bound v2 proximity-event census required')
    if (normalizer.get('schema')!=STATS_SCHEMA or normalizer.get('split')!='train'
            or normalizer.get('horizon')!=24 or normalizer.get('hand_points')!=21):
        raise ValueError('original TRAIN-only joint21 K24 normalization required')
    if stats is not None and metadata_identity(stats)!=recipe['stats_identity']:
        raise ValueError('CLI stats differ from admitted original normalization')
    if not 0<=recipe['event_dropout']<1:
        raise ValueError('bounded event dropout required')
    if recipe['source_ids']!=parent['source_ids']:
        raise ValueError('source identities changed')
    for split in ('train','dev'):
        if len(load_array(parent['indices'][split],mmap_mode='r'))!=recipe['windows'][split]:
            raise ValueError('inherited window count changed')
    return recipe
