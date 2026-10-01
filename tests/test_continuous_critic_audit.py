"""Check independent audit derivatives before native learning, using CPU math."""
import numpy as np
import torch
from src.task.CmResidual.continuous_critic_cm import initialized_models,critic_loss
from scripts.audit_continuous_critic_update import mlp,backward,array

def test_independent_numpy_full_auxiliary_critic_gradient():
    g=torch.Generator().manual_seed(51)
    x=torch.randn(17,70,generator=g,dtype=torch.float64)
    action=torch.randn(17,12,generator=g,dtype=torch.float64)
    target=torch.randn(17,6,generator=g,dtype=torch.float64)
    ret=torch.randint(0,2,(17,),generator=g).double()
    for variant in ('cm','state_only','none'):
        _,critic=initialized_models();critic=critic.double()
        # Nonzero value weights exercise BOTH paths through shared encoder.
        with torch.no_grad():critic.value.weight.copy_(torch.randn(1,64,generator=g,dtype=torch.float64)*.02)
        value,pred=critic(x,action,variant);loss=critic_loss(value,pred,ret,target,variant);loss.backward();w=critic.state_dict()
        latent,ca=mlp(x.numpy(),w,['encoder.0','encoder.2']);latent=np.maximum(latent,0)
        v=(latent@array(w['value.weight']).T+array(w['value.bias'])).ravel()
        din=np.concatenate((latent,action.numpy() if variant=='cm' else np.zeros((17,12))),-1)
        prediction,da=mlp(din,w,['dynamics.0','dynamics.2']);weight=0 if variant=='none' else .05
        dv=(v-ret.numpy())/17;dp=2*weight*(prediction-target.numpy())/(17*6)
        gradients,dl=backward(dp,w,['dynamics.0','dynamics.2'],da)
        gradients['value.weight']=dv[None]@latent;gradients['value.bias']=np.array([dv.sum()])
        dl=dl[:,:64]+dv[:,None]*array(w['value.weight'])
        eg,_=backward(dl*(ca[-1]>0),w,['encoder.0','encoder.2'],ca);gradients.update(eg)
        assert set(gradients)==set(dict(critic.named_parameters()))
        for name,param in critic.named_parameters():np.testing.assert_allclose(gradients[name],param.grad.numpy(),atol=2e-12,rtol=2e-12)
