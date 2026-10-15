"""Physical first-control errors in common units across action representations."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from ptv3_shared_wam.action_targets import freeze_chunk,consume_target,GEOMETRY_LIFTED,NATIVE
from trajectory_policy.retargeter_data import ACTIVE


def physical_errors(prediction,batch,controller=None,representation=GEOMETRY_LIFTED):
    value=prediction['action'].detach().float().cpu()
    if representation==NATIVE:
        if controller is None:raise ValueError('Native predictions need the physical PD controller')
        from trajectory_policy.pd_retargeter import pd_from_action
        q=batch['q_real'][:,0].detach().float().cpu()
        command=torch.zeros_like(q);command[:,list(ACTIVE)]=value[:,0]
        desired=pd_from_action(command,q,q.new_tensor(controller['offset']),q.new_tensor(controller['scale']))
        a=desired.numpy();b=batch['q_pd'][:,0].detach().float().cpu().numpy()
        rotation=(Rotation.from_euler('XYZ',a[:,3:6]).inv()*Rotation.from_euler('XYZ',b[:,3:6])).magnitude()
        fingers=list(ACTIVE)[6:]
        result=dict(PD_xyz_mm=np.linalg.norm(a[:,:3]-b[:,:3],axis=-1)*1000,
            PD_rotation_rad=rotation,PD_finger_rad=np.abs(a[:,fingers]-b[:,fingers]).mean(-1))
    elif representation==GEOMETRY_LIFTED:
        target=batch['action_target'].detach().float().cpu()
        a=value[:,0].numpy();b=target[:,0].numpy()
        rotation=(Rotation.from_rotvec(a[:,3:6]).inv()*Rotation.from_rotvec(b[:,3:6])).magnitude()
        result=dict(PD_xyz_mm=np.linalg.norm(a[:,:3]-b[:,:3],axis=-1)*1000,
            PD_rotation_rad=rotation,PD_finger_rad=np.abs(a[:,6:12]-b[:,6:12]).mean(-1),
            PD_lift_rad=np.abs(a[:,12:15]-b[:,12:15]).mean(-1))
    else:raise ValueError('Unsupported shared actor physical metric representation')
    for key,prefix,mask in [('trajectory','hand',batch['trajectory_valid'] & batch['start_hand_valid'][:,None]),
                            ('scene','scene',batch['scene_valid'] & batch['object_valid'][:,:,None,None])]:
        error=(prediction[key].detach().float()-batch[key+'_target'].float()).norm(dim=-1)
        result[prefix+'_all24_mm']=error[mask].cpu().numpy()*1000
    if controller is not None:
        q=batch['q_real'][:,0].detach().float().cpu()
        frozen=freeze_chunk(value,q,representation)
        intended,applied,clipped=consume_target(frozen[:,0],q,representation,controller)
        result['first_control_clipped']=clipped.numpy().astype(float)
        if 'q_pd' in batch:
            from trajectory_policy.pd_retargeter import pd_from_action
            desired=pd_from_action(intended,q,q.new_tensor(controller['offset']),q.new_tensor(controller['scale']))
            pd_target=batch['q_pd'][:,0].detach().float().cpu()
            result['PD_euler_rad']=(desired[:,3:6]-pd_target[:,3:6]).abs().mean(-1).numpy()
        if 'native_action' in batch:
            target=batch['native_action'][:,0].detach().float().cpu()
            result['command_first_mae']=(intended[:,list(ACTIVE)]-target).abs().mean(-1).numpy()
            result['applied_first_mae']=(applied[:,list(ACTIVE)]-target).abs().mean(-1).numpy()
    return result


def summarize_errors(parts):
    return {key:dict(mean=float(np.mean(x)),p95=float(np.percentile(x,95)),count=len(x))
            for key,values in parts.items() for x in [np.concatenate(values)] if len(x)}
