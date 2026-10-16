import numpy as np
import pytest
import torch
from ptv3_shared_wam_v2.wrist_shared_probe import MODEL_SCHEMA,WristSharedMLP
from ptv3_shared_wam_v2.wrist_shared_execution import load_matched_wrist,predict_matched_wrist,order_server_features


def stats():
    return {k:np.full(n,0. if k.endswith('mean') else 1.,np.float32) for k,n in
            [('xmean',811),('xstd',811),('ymean',6),('ystd',6),('zmean',2304),('zstd',2304)]}


def test_live_condition_and_server_row_order():
    torch.manual_seed(3);model=WristSharedMLP()
    torch.nn.init.normal_(model.net[-1].weight,std=.02)
    torch.nn.init.normal_(model.shared_input.weight,std=.02)
    s={k:torch.from_numpy(v) for k,v in stats().items()}
    x=torch.randn(4,811);z=torch.randn(4,2304)
    torch.testing.assert_close(predict_matched_wrist((model,s,'none'),x,z),model.net(x))
    assert not torch.allclose(predict_matched_wrist((model,s,'aligned'),x,z),model.net(x))
    source=torch.stack((torch.ones(2304)*2,torch.ones(2304)))
    ordered=order_server_features(source.tolist(),'cpu')
    assert (ordered[0]==1).all() and (ordered[1]==2).all()
    with pytest.raises(ValueError):order_server_features(source[:1],'cpu')


def test_loader_binds_frozen_feature_checkpoint_and_preserves_rng(tmp_path):
    model=WristSharedMLP();feature=tmp_path/'frozen.pt';feature.touch()
    path=tmp_path/'wrist.pt'
    torch.save(dict(schema=MODEL_SCHEMA,step=6000,mode='aligned',model=model.state_dict(),stats=stats(),
        feature_checkpoint=str(feature),feature_pooling='separate masked mean scene/hand/goal, concatenated2304D'),path)
    state=torch.random.get_rng_state().clone()
    load_matched_wrist(path,'aligned',feature,'cpu')
    assert torch.equal(torch.random.get_rng_state(),state)
    with pytest.raises(ValueError,match='frozen feature source'):load_matched_wrist(path,'aligned',tmp_path/'different.pt','cpu')
    with pytest.raises(ValueError,match='Fixed6000'):load_matched_wrist(path,'none',feature,'cpu')
