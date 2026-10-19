"""Masked joint21/flow/proximity/part losses with global DDP denominators."""
import torch
import torch.nn.functional as F
from . import paths
from ptv3_shared_wam_v4.objectives import normalized_components,combine_components

OBJECTIVE_SEMANTICS='v5.global-masked-joint21-surface16.v1'


def loss_components(prediction,batch,config,normalizer,reduction_dtype=None):
    result={}
    for name,key,mask in (
        ('hand','trajectory',batch['trajectory_valid']&batch['start_hand_valid'][:,None]),
        ('scene','scene',batch['scene_valid']&batch['object_valid'][:,:,None,None])):
        target=torch.where(mask[...,None],batch[key+'_target'].float(),0.)
        normalized=normalizer.normalize(name,target)
        error=torch.where(mask[...,None],prediction[key+'_normalized']-normalized,0.).square().mean(-1)
        weight=torch.ones_like(error)
        if name=='scene':
            floor,tau=config['motion_floor'],config['motion_tau_m']
            if tau<=0 or not 0<floor<=1:raise ValueError('positive motion scale and nonzero floor required')
            weight=floor+(1-floor)*torch.sigmoid(5*(target.norm(dim=-1)/tau-1))
        weight=torch.where(mask,weight,0.)
        result[name]=((error*weight).sum(dtype=reduction_dtype),weight.sum(dtype=reduction_dtype))
    valid=batch['interaction_valid']&prediction['interaction_prediction_valid']
    target=torch.where(valid,batch['contact_target'].float(),0.)
    logits=prediction['interaction_logits'].float()
    loss=F.binary_cross_entropy_with_logits(logits[...,0],target,reduction='none')
    loss=torch.where(valid,loss,0.)
    result['contact']=(loss.sum(dtype=reduction_dtype),valid.float().sum(dtype=reduction_dtype))
    part_valid=valid&(target>config.get('part_contact_threshold',.25))
    labels=torch.where(part_valid,batch['part_target'].long(),0)
    loss=F.cross_entropy(logits[...,1:].reshape(-1,16),labels.flatten(),reduction='none').reshape_as(labels)
    loss=torch.where(part_valid,loss,0.)
    result['part']=(loss.sum(dtype=reduction_dtype),part_valid.float().sum(dtype=reduction_dtype))
    return result


def forecast_loss(prediction,batch,config,normalizer,distributed=False):
    components=loss_components(prediction,batch,config,normalizer)
    total,terms=combine_components(normalized_components(components,distributed),config)
    terms['components']=components
    return total,terms
