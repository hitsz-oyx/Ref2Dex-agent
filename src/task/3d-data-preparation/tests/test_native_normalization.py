"""Native normalization is explicit and does not relax the existing main3 gate."""
from copy import deepcopy

import pytest
import torch

from pose_mesh_data.native_normalization import NativeHorizonNormalizer
from ptv3_shared_wam_v2.normalization import HorizonNormalizer


def stats():
    return dict(schema='ref2dex.native-horizon-stats.v1',split='train',training_allowed=False,
        units='m',fps=30,history=4,horizon=24,source_ids=[4,5,6,8,9,10],
        source_probabilities={str(s):1/6 for s in (4,5,6,8,9,10)},
        hand={'mean':[[.1,.2,.3]]*24,'std':[[.01,.02,.03]]*24},
        scene={'mean':[[.2,.3,.4]]*24,'std':[[.02,.03,.04]]*24})


@pytest.mark.parametrize('name,shape',[('hand',(2,24,2,11,3)),('scene',(2,4,24,32,3))])
def test_native_roundtrip_and_old_default_remains_restricted(name,shape):
    normalizer=NativeHorizonNormalizer(stats())
    x=torch.randn(shape)
    torch.testing.assert_close(normalizer.denormalize(name,normalizer.normalize(name,x)),x)
    with pytest.raises(ValueError,match='v2 TRAIN-only'):
        HorizonNormalizer(stats())


def test_native_clock_pending_source_dev_or_degenerate_std_are_rejected():
    bad=deepcopy(stats());bad['source_ids'].append(7)
    with pytest.raises(ValueError,match='six-source'):
        NativeHorizonNormalizer(bad)
    bad=deepcopy(stats());bad['split']='dev'
    with pytest.raises(ValueError,match='TRAIN'):
        NativeHorizonNormalizer(bad)
    bad=deepcopy(stats());bad['hand']['std'][0][0]=0
    with pytest.raises(ValueError,match='positive'):
        NativeHorizonNormalizer(bad)
