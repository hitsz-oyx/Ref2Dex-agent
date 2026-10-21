"""v6 optional decoded-geometry objective; legacy objective stays unchanged."""
from . import paths
from ptv3_shared_wam_v5.objectives import loss_components as base_components
from ptv3_shared_wam_v5.objectives import OBJECTIVE_SEMANTICS as BASE_SEMANTICS
from ptv3_shared_wam_v4.objectives import normalized_components,combine_components as base_combine
from .geometry_consistency import geometry_components,validate_geometry_config

GEOMETRY_OBJECTIVE = 'v6.global-masked-joint21-surface16.geometry-relative.v1'


def objective_semantics(config):
    return GEOMETRY_OBJECTIVE if config.get('geometry_consistency') else BASE_SEMANTICS


def active_geometry_weight(config, step=None):
    if not config.get('geometry_consistency'):return 0.
    warmup = config['geometry_warmup_updates']
    return config['geometry_weight']*(min(1., step/warmup) if step is not None else 1.)


class ComponentStatistics(dict):
    geometry_diagnostics = None


def loss_components(prediction,batch,config,normalizer,reduction_dtype=None):
    result = ComponentStatistics(base_components(prediction,batch,config,normalizer,reduction_dtype))
    if config.get('geometry_consistency'):
        extra, diagnostics = geometry_components(prediction,batch,config,reduction_dtype)
        result.update(extra);result.geometry_diagnostics=diagnostics
    return result


def combine_components(means,config,step=None):
    loss, terms = base_combine(means,config)
    if config.get('geometry_consistency'):
        validate_geometry_config(config)
        terms.update(geometry_contact=means['geometry_contact'],geometry_clearance=means['geometry_clearance'])
        loss = loss+active_geometry_weight(config,step)*(means['geometry_contact']+
                    config['geometry_clearance_weight']*means['geometry_clearance'])
    return loss, terms


def forecast_loss(prediction,batch,config,normalizer,distributed=False,step=None):
    components=loss_components(prediction,batch,config,normalizer)
    total,terms=combine_components(normalized_components(components,distributed),config,step)
    terms['components']=components
    return total,terms
