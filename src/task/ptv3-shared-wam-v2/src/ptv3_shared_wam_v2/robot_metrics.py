"""Physical first-control errors; all actors predict the same geometry-lift label."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from ptv3_shared_wam.action_targets import freeze_chunk,consume_target,GEOMETRY_LIFTED


def physical_errors(prediction,batch,controller=None):
    value=prediction['action'].detach().float().cpu()
    target=batch['action_target'].detach().float().cpu()
    a=value[:,0].numpy();b=target[:,0].numpy()
    rotation=(Rotation.from_rotvec(a[:,3:6]).inv()*Rotation.from_rotvec(b[:,3:6])).magnitude()
    result=dict(PD_xyz_mm=np.linalg.norm(a[:,:3]-b[:,:3],axis=-1)*1000,
        PD_rotation_rad=rotation,PD_finger_rad=np.abs(a[:,6:12]-b[:,6:12]).mean(-1),
        PD_lift_rad=np.abs(a[:,12:15]-b[:,12:15]).mean(-1))
    for key,prefix,mask in [('trajectory','hand',batch['trajectory_valid'] & batch['start_hand_valid'][:,None]),
                            ('scene','scene',batch['scene_valid'] & batch['object_valid'][:,:,None,None])]:
        error=(prediction[key].detach().float()-batch[key+'_target'].float()).norm(dim=-1)
        result[prefix+'_all24_mm']=error[mask].cpu().numpy()*1000
    if controller is not None:
        q=batch['q_real'][:,0].detach().float().cpu()
        frozen=freeze_chunk(value,q,GEOMETRY_LIFTED)
        _,_,clipped=consume_target(frozen[:,0],q,GEOMETRY_LIFTED,controller)
        result['first_control_clipped']=clipped.numpy().astype(float)
    return result


def summarize_errors(parts):
    return {key:dict(mean=float(np.mean(x)),p95=float(np.percentile(x,95)),count=len(x))
            for key,values in parts.items() for x in [np.concatenate(values)] if len(x)}
