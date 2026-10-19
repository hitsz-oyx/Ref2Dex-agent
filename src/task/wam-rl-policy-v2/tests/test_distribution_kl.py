import torch
from wam_rl_policy_v2.distribution_kl import actual_policy_kl


def test_calibrated_kl_matches_the_actual_upstream_sampling_distribution():
    torch.manual_seed(246)
    old_mean=torch.randn(7,12,dtype=torch.float64);mean=old_mean+torch.randn_like(old_mean)*.03
    old=torch.randn(7,12,dtype=torch.float64)*.2-.8;new=old+torch.randn_like(old)*.02
    left=torch.distributions.MultivariateNormal(old_mean,scale_tril=torch.diag_embed(old.exp().square()))
    right=torch.distributions.MultivariateNormal(mean,scale_tril=torch.diag_embed(new.exp().square()))
    torch.testing.assert_close(actual_policy_kl(old_mean,old,mean,new),torch.distributions.kl_divergence(left,right),rtol=1e-10,atol=1e-10)


def test_legacy_proxy_can_request_opposite_learning_rate_change_at_real_late_scale():
    log_std=torch.tensor([.0924739316],dtype=torch.float64).sqrt().log()
    old=torch.zeros(1,dtype=torch.float64);new=torch.full((1,),.03,dtype=torch.float64)
    proxy=((old-new).square()/(2*log_std.exp().square())).sum()
    actual=actual_policy_kl(old,log_std,new,log_std)
    assert proxy<.016/2 and actual>.016*2
