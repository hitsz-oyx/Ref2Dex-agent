"""Small causal/likelihood/clock test; no simulator or scientific model updates."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.task.CmResidual.coherent_motor_request import CoherentMotorRequest
from src.task.CmResidual.continuous_critic_cm import request_log_probability

def main():
    torch.set_num_threads(2);gen=torch.Generator().manual_seed(4101);reference=torch.Generator().manual_seed(4101);cache=CoherentMotorRequest();mu=torch.zeros((2,12));ls=torch.full((12,),-3.)
    expected=torch.randn(mu.shape,generator=reference);p,decision=cache.choose(0,mu,ls,gen);assert decision and torch.equal(p['request_noise'],expected);first={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in p.items()};rng=gen.get_state().clone()
    logp=request_log_probability(p['request_mean'],p['request_logstd'],p['request']);mu.fill_(100);ls.fill_(2)
    for tick in (1,2,3):
        p,decision=cache.choose(tick,mu,ls,gen);assert not decision and p['decision_tick']==0
        assert all(torch.equal(p[k],first[k]) for k in ['request','request_mean','request_logstd','request_noise']) and torch.equal(gen.get_state(),rng)
        assert torch.equal(request_log_probability(p['request_mean'],p['request_logstd'],p['request']),logp)
        assert not torch.equal(request_log_probability(mu,ls,p['request']),logp)
        p['request'].zero_()  # Returned aliases must not corrupt held command.
    expected=torch.randn(mu.shape,generator=reference);p,decision=cache.choose(4,mu,ls,gen);assert decision and p['decision_tick']==4 and torch.equal(p['request_noise'],expected)
    try:cache.choose(6,mu,ls,gen)
    except ValueError:pass
    else:raise AssertionError('skipped native tick accepted')
    for tick in range(5,202):p,decision=cache.choose(tick,mu,ls,gen);assert decision==(tick%4==0) and p['decision_tick']==tick-tick%4
    assert p['decision_tick']==200;print('PASS causal four-tick holds, decision-state likelihood, private RNG, alias protection, exact202tick clock (CPU tiny engineering only)')

if __name__=='__main__':main()
