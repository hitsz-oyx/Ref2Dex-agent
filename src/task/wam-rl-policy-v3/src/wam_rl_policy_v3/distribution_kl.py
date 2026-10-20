"""KL calibration for the preserved upstream squared-scale-tril policy."""
import torch

OPTIMIZATION_CONTRACT='ref2dex.official-squared-scale-tril-actual-kl.v1'


def actual_policy_kl(old_mean,old_log_std,mean,log_std):
    # Upstream receives exp(log_std)**2 as scale_tril. The actual log standard
    # deviation is 2*log_std; the covariance diagonal is exp(4*log_std).
    old_log_scale=2*old_log_std;log_scale=2*log_std
    return (log_scale-old_log_scale+(torch.exp(2*old_log_scale)+(old_mean-mean).square())
            /(2*torch.exp(2*log_scale))-.5).sum(-1)
