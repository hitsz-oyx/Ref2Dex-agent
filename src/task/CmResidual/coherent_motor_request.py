"""Fixed four-tick Gaussian REQUEST execution; likelihood belongs to decision state."""
import torch

class CoherentMotorRequest:
    period=4
    def __init__(self):
        self.last_tick=-1
        self.packet=None

    def choose(self,tick,mean,logstd,generator):
        if tick!=self.last_tick+1:raise ValueError('contiguous native clock required')
        if mean.ndim!=2 or mean.shape[-1]!=12 or logstd.shape!=(12,):raise ValueError('twelve independent Gaussian request coordinates')
        if not torch.isfinite(mean).all() or not torch.isfinite(logstd).all():raise ValueError('finite Gaussian parameters required')
        decision=tick%self.period==0
        if decision:
            mu=mean.detach().clone();ls=logstd.detach().clone();noise=torch.randn(mu.shape,dtype=mu.dtype,device=mu.device,generator=generator);raw=mu+ls.exp()*noise
            if not torch.isfinite(raw).all():raise ValueError('nonfinite Gaussian request')
            self.packet=dict(request=raw,request_mean=mu,request_logstd=ls,request_noise=noise,decision_tick=tick)
        else:
            if self.packet is None or self.packet['request'].shape!=mean.shape or self.packet['request'].device!=mean.device or self.packet['request'].dtype!=mean.dtype:raise ValueError('fixed rollout cohort required')
        self.last_tick=tick
        return {k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in self.packet.items()},decision
