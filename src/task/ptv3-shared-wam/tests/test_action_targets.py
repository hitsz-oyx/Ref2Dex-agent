import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from ptv3_shared_wam.action_targets import NATIVE,PD,GEOMETRY,encode_targets,freeze_chunk,consume_target
from trajectory_policy.pd_retargeter import pd_from_action
from trajectory_policy.retargeter_data import ACTIVE


@pytest.mark.parametrize('representation',[NATIVE,PD,GEOMETRY])
def test_real_controller_roundtrip_with_different_future_feedback(representation):
    rng=np.random.default_rng(31);query=torch.tensor(rng.normal(0,.2,(5,18)),dtype=torch.float32)
    live=query[:,None].repeat(1,8,1)+torch.tensor(rng.normal(0,.02,(5,8,18)),dtype=torch.float32)
    raw=torch.tensor(rng.uniform(-.08,.08,(5,8,18)),dtype=torch.float32)
    raw[..., [i for i in range(18) if i not in ACTIVE]]=0
    controller=dict(offset=[.03]*18,scale=[1.]*3+[np.pi]*3+[.9]*12)
    actual=pd_from_action(raw,live,torch.tensor(controller['offset']),torch.tensor(controller['scale']))
    encoded=encode_targets(actual.numpy(),query.numpy(),representation,raw[...,list(ACTIVE)].numpy())
    frozen=freeze_chunk(torch.tensor(encoded),query,representation)
    _,applied,clipping=consume_target(frozen,live,representation,controller)
    replay=pd_from_action(applied,live,torch.tensor(controller['offset']),torch.tensor(controller['scale']))
    torch.testing.assert_close(replay,actual,atol=5e-7,rtol=1e-5)
    assert not clipping.any()


def test_frozen_target_is_not_reanchored_when_consuming_second_action():
    query=torch.zeros(1,18);label=torch.zeros(1,8,12);label[...,0]=.1
    frozen=freeze_chunk(label,query,PD);live=query.clone();live[:,0]=.06
    _,applied,_=consume_target(frozen[:,1],live,PD,dict(offset=[0.]*18,scale=[1.]*18))
    assert applied[0,0].item()==pytest.approx(.04)


def test_all_chunk_fingers_and_wrist_use_query_not_future_q():
    base=np.ones((2,18),np.float32);target=np.full((2,8,18),3.,np.float32)
    encoded=encode_targets(target,base,PD)
    np.testing.assert_array_equal(encoded,np.full((2,8,12),2.,np.float32))


def test_geometry_uses_urdf_intrinsic_xyz_and_short_relative_rotation():
    query=np.zeros((1,18),np.float32);query[0,3:6]=[.6,-.4,3.13]
    targets=np.repeat(query[:,None],8,axis=1);targets[...,5]=-3.13
    encoded=encode_targets(targets,query,GEOMETRY)
    expected=Rotation.from_euler('XYZ',query[0,3:6]).inv()*Rotation.from_euler('XYZ',targets[0,0,3:6])
    np.testing.assert_allclose(encoded[0,0,3:6],expected.as_rotvec(),atol=1e-6)
    assert np.linalg.norm(encoded[0,0,3:6])<.03


def test_geometry_wrist_chain_matches_checked_asset():
    from pathlib import Path
    import xml.etree.ElementTree as ET
    root=Path(__file__).resolve().parents[4]
    tree=ET.parse(root/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf')
    for i,axis in enumerate(['1 0 0','0 1 0','0 0 1']*2,1):
        joint=tree.find(".//joint[@name='joint%d']"%i)
        assert joint.find('axis').attrib['xyz']==axis
        assert joint.find('origin').attrib==dict(xyz='0 0 0',rpy='0 0 0')


def test_real_dataset_collate_and_train_only_normalization_use_physical_labels(tmp_path):
    from test_rl_robot import corpus
    from ptv3_shared_wam.rl_robot import RLRobotWindows
    from ptv3_shared_wam.data import shared_collate
    from ptv3_shared_wam.robot import initialize_retargeter
    from trajectory_policy.chunk_retargeter import ChunkRetargeter
    dataset=RLRobotWindows(corpus(tmp_path),'train',action_representation=PD)
    sample=dataset[3];batch=shared_collate([sample,dataset[0]])
    assert batch['action_target'].shape==(2,8,12)
    np.testing.assert_allclose(sample['action_target'],sample['pd_target'][...,list(ACTIVE)]-sample['native_state'][None,list(ACTIVE)])
    net=ChunkRetargeter();initialize_retargeter(net,dataset)
    expected=np.stack([dataset[i]['action_target'] for i in range(len(dataset))])
    np.testing.assert_allclose(net.action_mean.numpy(),expected.mean(0),atol=1e-7)
    with pytest.raises(ValueError,match='TRAIN'):RLRobotWindows(dataset.root,'val',action_representation=PD).normalization()


def test_physical_loss_uses_new_labels_and_forbids_native_unit_first_loss(monkeypatch):
    import ptv3_shared_wam.robot as robot
    from trajectory_policy.chunk_retargeter import ChunkRetargeter
    monkeypatch.setattr(robot,'forecast_loss',lambda *args:(torch.tensor(0.),{}))
    p=torch.ones(2,8,12,requires_grad=True);batch=dict(native_action=torch.full_like(p,9),action_target=torch.ones_like(p))
    prediction=dict(action=p,action_anchor=p)
    loss,_=robot.robot_loss(prediction,batch,dict(action_representation=PD),ChunkRetargeter())
    assert loss.item()==0
    with pytest.raises(ValueError,match='different units'):
        robot.robot_loss(prediction,batch,dict(action_representation=PD,action_objective='control-aware-first.v1'),ChunkRetargeter())


def test_physical_checkpoint_units_and_geometry_roundtrip_gate(tmp_path):
    import json
    from ptv3_shared_wam.native_inputs import load_execution_retargeter
    from ptv3_shared_wam.action_targets import R_ONLY_SCHEMA
    from trajectory_policy.chunk_retargeter import ChunkRetargeter
    net=ChunkRetargeter();controller=dict(offset=[0.]*18,scale=[1.]*18)
    pair=dict(config=dict(native_controller=controller,native_PD_scale=[1.]*12),
              model={'retargeter.'+k:v for k,v in net.state_dict().items()},provenance=dict(source=str(tmp_path/'corpus')))
    manifest=dict(status='COMPLETED',source=str(tmp_path/'corpus'),normalization='TRAIN-only per-horizon-coordinate.v1',representations=[PD,GEOMETRY])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    state={k:v.clone() for k,v in net.state_dict().items()};state['action_mean']+=.3
    packet=dict(schema=R_ONLY_SCHEMA,width=128,model=state,native_controller=controller,native_PD_scale=torch.ones(12),action_representation=PD)
    path=tmp_path/'PD.pt';torch.save(packet,path)
    restored=load_execution_retargeter(pair,path)
    assert restored.action_representation==PD and not restored.first_only
    torch.save(dict(packet,action_representation=GEOMETRY),path)
    with pytest.raises(ValueError,match='roundtrip'):load_execution_retargeter(pair,path)


def test_continuous_lift_restores_real_saturated_pi_servo_direction():
    from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
    # s6_airplane_fly_1, query201, chunk7: both poses agree but old
    # geometry decoding changes the native pitch command from +1 to -1.
    q=torch.zeros(1,18);q[0,3:6]=torch.tensor([4.6777715683,3.4784977436,-5.2264471054])
    pd=q[:,None].repeat(1,8,1);pd[...,3:6]=torch.tensor([5.1151351929,6.6200904846,-4.2884674072])
    controller=dict(offset=[0.]*18,scale=[1.]*3+[float(np.pi)]*3+[1.]*12)
    legacy=freeze_chunk(torch.tensor(encode_targets(pd.numpy(),q.numpy(),GEOMETRY)),q,GEOMETRY)
    _,wrong,_=consume_target(legacy,q[:,None].expand(-1,8,-1),GEOMETRY,controller)
    assert wrong[0,7,4]<-.99
    encoded=encode_targets(pd.numpy(),q.numpy(),GEOMETRY_LIFTED)
    frozen=freeze_chunk(torch.tensor(encoded),q,GEOMETRY_LIFTED)
    _,applied,clipped=consume_target(frozen,q[:,None].expand(-1,8,-1),GEOMETRY_LIFTED,controller)
    assert applied[0,7,4]>.99 and not clipped.any()
    recovered=pd_from_action(applied,q[:,None],torch.zeros(18),torch.tensor(controller['scale']))
    torch.testing.assert_close(recovered,pd,atol=2e-6,rtol=0)


@pytest.mark.parametrize('pitch',[np.pi/2,-np.pi/2,5*np.pi/2])
def test_continuous_lift_preserves_winding_at_gimbal_lock(pitch):
    from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
    q=torch.zeros(2,18);q[:,3:6]=torch.tensor([7.,pitch,-4.])
    target=q[:,None].repeat(1,8,1)
    encoded=torch.tensor(encode_targets(target.numpy(),q.numpy(),GEOMETRY_LIFTED))
    frozen=freeze_chunk(encoded,q,GEOMETRY_LIFTED)
    _,applied,clipped=consume_target(frozen,q[:,None].expand(-1,8,-1),GEOMETRY_LIFTED,dict(offset=[0.]*18,scale=[1.]*18))
    recovered=pd_from_action(applied,q[:,None],torch.zeros(18),torch.ones(18))
    torch.testing.assert_close(recovered,target,atol=2e-6,rtol=0)
    assert not clipped.any()


def test_continuous_lift_model_and_buffer_keep_extra_branch_coordinates():
    from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
    from ptv3_shared_wam.target_retargeter import make_retargeter
    from ptv3_shared_wam.execution import ActionBuffer
    net=make_retargeter(GEOMETRY_LIFTED)
    value=net(torch.zeros(1,87),torch.zeros(1,24,33))
    assert value.shape==(1,8,15)
    buffer=ActionBuffer(4,15);buffer.refill(value[0].detach().numpy(),0)
    assert buffer.pop()[0].shape==(15,)


def test_lifted_full_checkpoint_restores_explicit_output_units(tmp_path):
    from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
    from ptv3_shared_wam.target_retargeter import make_retargeter
    from ptv3_shared_wam.native_inputs import load_execution_retargeter
    net=make_retargeter(GEOMETRY_LIFTED)
    pair=dict(config=dict(action_representation=GEOMETRY_LIFTED),model={'retargeter.'+k:v for k,v in net.state_dict().items()})
    restored=load_execution_retargeter(pair)
    assert restored.action_representation==GEOMETRY_LIFTED and restored.head.out_features==15


def test_lifted_recipe_preserves_large_training_hyperparameters():
    import json
    from pathlib import Path
    configs=Path(__file__).resolve().parents[1]/'configs'
    old=json.loads((configs/'robot_660_large.json').read_text())
    new=json.loads((configs/'robot_660_geometry_lift.json').read_text())
    assert {k:new[k] for k in old}==old
    assert set(new)-set(old)=={'action_representation','action_objective','artifact_budget_bytes','min_free_disk_bytes'}


def test_new_storage_reserve_and_legacy_budget_are_separate(monkeypatch):
    import ptv3_shared_wam.checkpoint as checkpoint
    from types import SimpleNamespace
    monkeypatch.setattr(checkpoint,'directory_bytes',lambda *args:500_000_000_000)
    monkeypatch.setattr(checkpoint.shutil,'disk_usage',lambda *args:SimpleNamespace(free=30*2**30))
    checkpoint.check_shared_storage(dict(min_free_disk_bytes=20*2**30))
    with pytest.raises(RuntimeError,match='legacy'):checkpoint.check_shared_storage({})
    monkeypatch.setattr(checkpoint.shutil,'disk_usage',lambda *args:SimpleNamespace(free=19*2**30))
    with pytest.raises(RuntimeError,match='reserve'):checkpoint.check_shared_storage(dict(min_free_disk_bytes=20*2**30))
