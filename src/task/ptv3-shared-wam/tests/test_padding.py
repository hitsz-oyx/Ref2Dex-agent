import pytest
import torch

from ptv3_shared_wam.padding import vectorized_padding


class Point(dict):
    __getattr__=dict.__getitem__


@pytest.mark.parametrize('counts,patch', [([1,2,3,4],4),([129,130,255,256,257],128),
    ([4,5,6,7,8,9],4),([1,0,12,0,15],4),([512]*32,128)])
def test_vectorized_padding_preserves_every_serialized_index(counts,patch):
    from ptv3_shared_wam.scene import SceneHistoryEncoder
    from ptv3.ptv3 import SerializedAttention
    net=SerializedAttention(128,4,patch)
    offset=torch.tensor(counts).cumsum(0)
    a=Point(offset=offset,feat=torch.randn(sum(counts),128))
    b=Point(offset=offset,feat=a.feat)
    before=net.get_padding_and_inverse(a)
    after=vectorized_padding(net,b)
    assert all(torch.equal(x,y) for x,y in zip(before,after))
    cached=vectorized_padding(net,b)
    assert all(x.data_ptr()==y.data_ptr() for x,y in zip(cached,after))
    torch.testing.assert_close(after[0][after[1]],torch.arange(sum(counts)),rtol=0,atol=0)
