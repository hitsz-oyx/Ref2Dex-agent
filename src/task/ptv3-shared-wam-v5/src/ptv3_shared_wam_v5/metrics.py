"""Descriptive DEV diagnostics; these do not change optimization weights."""
import torch
from .contracts import SPARSE11_IN_21


def forecast_metrics(prediction,batch):
    valid=batch['trajectory_valid']&batch['start_hand_valid'][:,None]
    target=torch.where(valid[...,None],batch['trajectory_target'].float(),0.)
    delta=prediction['trajectory'].float()-target
    epe=delta.norm(dim=-1)*1000
    result={}

    def measure(name,values,mask):
        result[name]=dict(sum=float(torch.where(mask,values,0.).sum()),count=int(mask.sum()),unit='mm')

    measure('hand/root',epe[...,0],valid[...,0])
    relative=(delta-delta[...,:1,:]).norm(dim=-1)*1000
    relative_valid=valid&valid[...,:1]
    relative_valid=relative_valid.clone();relative_valid[...,0]=False
    measure('hand/root_relative',relative,relative_valid)
    old=torch.zeros_like(valid)
    for i,sid in enumerate(batch['source_id'].tolist()):
        ids=(0,2,5,9,13,17,4,8,12,16,20) if sid==4 else SPARSE11_IN_21
        old[i,...,list(ids)]=True
    measure('hand/old11',epe,valid&old)
    measure('hand/new10',epe,valid&~old)
    tips=torch.zeros_like(valid);tips[...,[4,8,12,16,20]]=True
    measure('hand/tips',epe,valid&tips)
    contact_valid=batch['interaction_valid']&prediction['interaction_prediction_valid']
    confidence=torch.where(contact_valid,batch['contact_target'].float(),0.)
    near=(contact_valid&(confidence>.25)).any(1).any(-1)
    phase_known=contact_valid.any(1).any(-1)
    measure('hand/contact_phase',epe,valid&near[...,None,None])
    measure('hand/noncontact_phase',epe,valid&(phase_known&~near)[...,None,None])
    scene_valid=batch['scene_valid']&batch['object_valid'][:,:,None,None]
    scene_target=torch.where(scene_valid[...,None],batch['scene_target'].float(),0.)
    scene_error=(prediction['scene'].float()-scene_target).norm(dim=-1)*1000
    contact_error=(prediction['interaction'][...,0].float()-confidence).abs()
    for tick in range(24):
        measure('hand/h%02d'%(tick+1),epe[:,tick],valid[:,tick])
        measure('scene/h%02d'%(tick+1),scene_error[:,:,tick],scene_valid[:,:,tick])
    for name,mask in [('all',contact_valid),('near_gt025',contact_valid&(confidence>.25)),
                      ('core_approx3mm',contact_valid&(confidence>=1))]:
        result['contact/mae_'+name]=dict(sum=float(torch.where(mask,contact_error,0.).sum()),count=int(mask.sum()),unit='1')
    part_valid=contact_valid&(confidence>.25)
    target_part=torch.where(part_valid,batch['part_target'].long(),0)
    predicted_part=prediction['interaction'][...,1:].argmax(-1)
    confusion=torch.bincount((16*target_part[part_valid]+predicted_part[part_valid]),minlength=256).reshape(16,16)
    result['part/confusion_gt025']=dict(matrix=confusion.cpu().tolist(),unit='count')
    return result


def merge_metrics(total,current):
    for key,value in current.items():
        if key not in total:total[key]=value;continue
        if 'matrix' in value:
            total[key]['matrix']=[[a+b for a,b in zip(x,y)] for x,y in zip(total[key]['matrix'],value['matrix'])]
        else:
            total[key]['sum']+=value['sum'];total[key]['count']+=value['count']


def summarize_metrics(total):
    return {key:dict(value,mean=value['sum']/value['count'] if value['count'] else None)
            if 'sum' in value else value for key,value in total.items()}
