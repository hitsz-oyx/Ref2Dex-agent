"""Explicit native-only normalizer; main3's default validation remains strict."""
import numpy as np
import torch
from torch import nn

from .normalization import HorizonNormalizer


class NativeHorizonNormalizer(HorizonNormalizer):
    stats_schema='ref2dex.native-horizon-stats.v1'

    def __init__(self,stats):
        nn.Module.__init__(self)
        if (stats.get('schema')!=self.stats_schema or stats.get('split')!='train'
                or stats.get('training_allowed') is not False or stats.get('units')!='m'
                or stats.get('fps')!=30 or stats.get('history')!=4 or stats.get('horizon')!=24
                or stats.get('source_ids')!=[4,5,6,8,9,10]):
            raise ValueError('explicit native six-source engineering TRAIN statistics required')
        weights=stats.get('source_probabilities',{})
        if (set(weights)!={'4','5','6','8','9','10'} or not np.isfinite(list(weights.values())).all()
                or any(v<=0 for v in weights.values()) or not np.isclose(sum(weights.values()),1,atol=1e-12,rtol=0)):
            raise ValueError('native source probability contract required')
        for name in ('hand','scene'):
            mean=torch.tensor(stats[name]['mean'],dtype=torch.float32)
            std=torch.tensor(stats[name]['std'],dtype=torch.float32)
            if (mean.shape!=(24,3) or std.shape!=(24,3) or not torch.isfinite(mean).all()
                    or not torch.isfinite(std).all() or (std<=0).any()):
                raise ValueError('native finite24x3 positive normalization required')
            self.register_buffer(name+'_mean',mean)
            self.register_buffer(name+'_std',std)
