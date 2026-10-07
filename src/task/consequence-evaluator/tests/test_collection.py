"""Exercise real collection mechanics and driver loop with a tiny CPU fake env."""
import ast
import json
from pathlib import Path
import sys
import os
import runpy
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from consequence_evaluator.collection import Episode, Perturbations, pose_matrix, smooth_residual, PHASES


def root_state(z=0.):
    state = np.zeros(13, dtype='float32')
    state[2], state[6] = z, 1
    return state


def test_native_xyzw_translation_and_rotation():
    state = root_state(.3)
    state[3:7] = [0, 0, np.sqrt(.5), np.sqrt(.5)]
    matrix = pose_matrix(state)
    assert np.allclose(matrix[:3, 3], [0, 0, .3])
    assert np.allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-6)
    with pytest.raises(ValueError, match='quaternion'):
        pose_matrix(np.zeros(13))


def test_chunk_is_reproducible_bounded_and_smooth_at_boundaries():
    chunk = smooth_residual(np.random.default_rng(5), .08)
    assert np.array_equal(chunk, smooth_residual(np.random.default_rng(5), .08))
    assert chunk.shape == (24, 18) and np.all(chunk[[0, -1]] == 0)
    assert np.abs(chunk[:, :3]).max() <= .02 and np.abs(chunk).max() <= .08
    assert np.all(chunk[:,[7,9,11,13,16,17]] == 0)
    # Four linear knots and a taper avoid independent per-frame jumps.
    assert np.abs(np.diff(chunk, axis=0)).max() < .04


def test_no_reset_perturbation_one_complete_chunk_and_clean_recovery():
    controller = Perturbations(6, seed=5)
    base = np.zeros((6, 18), dtype='float32')
    actual, perturbing = [], []
    for tick in range(60):
        action, phase, info = controller.apply(base, np.zeros(6), np.zeros(6, bool),
                                             np.zeros(6), tick, np.full(6, 100-tick), np.ones(6, bool))
        actual.append(action)
        perturbing.append(info['perturbing'])
    assert controller.started[0] == -1  # clean
    assert controller.started[1] == 1  # approach, after the first physical step
    assert np.asarray(perturbing)[:, 1].sum() == 24
    assert np.all(np.asarray(actual)[25:, 1] == 0)  # exact return to expert
    assert np.all(controller.started[2:] == -1)  # never reached assigned stages


def test_insufficient_remaining_clock_never_starts_partial_chunk():
    controller = Perturbations(6, seed=7)
    _, _, info = controller.apply(np.zeros((6,18)), np.zeros(6), np.zeros(6,bool), np.zeros(6),
                                  3, np.full(6,23), np.ones(6,bool))
    assert np.all(controller.started == -1) and not info['perturbing'].any()


def test_clipped_executed_action_and_residual_are_recorded_separately():
    controller = Perturbations(6, seed=7)
    controller.chunks[1] = .1
    base = np.full((6,18), .99)
    action, _, info = controller.apply(base, np.zeros(6), np.zeros(6,bool), np.zeros(6),
                                       1, np.full(6,100), np.ones(6,bool))
    assert np.all(action[1] == 1)
    assert np.allclose(info['actual_residual'][1], .01)
    assert info['clipped'][1].all() and np.allclose(info['residual'][1], .1)


def test_episode_alignment_and_no_second_episode_append():
    episode = Episode(np.zeros(6), root_state(), False,kinematics={'hand_keypoints':np.zeros((11,3))})
    for tick in range(24):
        episode.append(np.full(18,tick/100), 'approach', np.full(6,tick+1),
                       root_state((tick+1)/1000), False, np.zeros(18), np.zeros(18,bool), tick==23,
                       kinematics={'hand_keypoints':np.zeros((11,3))})
    arrays = episode.arrays()
    assert arrays['history'].shape == (25,6) and arrays['action'].shape == (24,18)
    assert arrays['object_pose'][1,2,3] == pytest.approx(.001)
    assert not arrays['progress_mask'].any() and np.isnan(arrays['progress']).all()
    assert set(arrays) == {'history','action','residual_plan','plan_known','hand_keypoints','object_pose','timestamps','phase','progress','progress_mask'}
    with pytest.raises(ValueError,match='second episode'):
        episode.append(np.zeros(18),'approach',np.zeros(6),root_state(),False,np.zeros(18),np.zeros(18,bool),False)


def test_actual_driver_never_resets_partial_done_envs_and_exports_full_episodes(tmp_path):
    """Run the actual Collector class extracted without GPU-only module imports."""
    path = TASK/'tools/run/collect_continuous.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=='Collector')
    n=6
    task = SimpleNamespace(dt=1/30, num_envs=n, device='cpu',
                           _target_states=torch.as_tensor(np.stack([root_state()]*n)),
                           _contact_forces=torch.zeros(n,5,3), _contact_body_ids=torch.arange(5),
                           _tar_contact_forces=torch.zeros(n,3), start_times=torch.zeros(n,dtype=torch.long),
                           _dof_pos=torch.zeros(n,18),_humanoid_root_states=torch.as_tensor(np.stack([root_state()]*n)),
                           progress_buf=torch.zeros(n,dtype=torch.long),data_id=torch.zeros(n,dtype=torch.long),
                           object_name=['fixture'],object_id=torch.zeros(1,dtype=torch.long),
                           motion_file=['fixture'],
                           max_episode_length=torch.full((1,),80,dtype=torch.long),rollout_length=80)
    def native_pd(actions):
        actions[:,6:]=(1+actions[:,6:])/2
    task.pre_physics_step=native_pd
    reset_calls, controls = [], []
    class Base:
        is_rnn=False
        observation_router=None
        expert_names=['engineering_fake_expert']
        device='cpu'
        def __init__(self):
            self.env=SimpleNamespace(task=task)
        def env_reset(self, ids):
            reset_calls.append(ids.clone())
            task.progress_buf.zero_()
            return {'obs':torch.zeros(n,6)}
        def get_action(self, obs, deterministic):
            self.last_teacher_choice=torch.zeros(n,dtype=torch.long)
            return torch.zeros(n,18)
        def env_step(self, env, action):
            action=action+.02  # Native domain action noise, after wrapper input.
            controls.append(action.clone())
            task.pre_physics_step(action)  # Actual pre-physics capture and in-place PD pattern.
            task.progress_buf+=1
            task._dof_pos+=.001
            tick=int(task.progress_buf[0])
            if tick>=5:
                task._contact_forces[:]=1
                task._tar_contact_forces[:]=1
            if tick>=20:
                task._target_states[:,2]=.04
            done=task.progress_buf>=torch.arange(65,71)
            return {'obs':torch.full((n,6),float(tick))},torch.zeros(n),done,{}
        def _post_step(self, info):
            pass
    a=SimpleNamespace(num_envs=n, waves=1,seed=7,amplitude=.08,max_steps=100,split='train')
    manifest={'episodes':[]}
    frozen={str((ROOT/'third_party/DExplore/dexplore/evaluate.py').resolve()):'engineering-only'}
    (tmp_path/'diagnostics').mkdir()
    def write(path,data):path.write_text(json.dumps(data))
    from hashlib import sha256
    class Geometry:
        def __init__(self,*args):pass
        def measure(self,task):return torch.zeros(n,11,3),torch.full((n,),.001)
    namespace=dict(torch=torch,np=np,Path=Path,router=SimpleNamespace(RoutedPlayer=Base),a=a,ROOT=ROOT,
                   PhysicalGeometry=Geometry,
                   DexploreTask=SimpleNamespace(StateInit=SimpleNamespace(Start='Start')),frozen=frozen,
                   output=tmp_path,check=lambda:None,manifest=manifest,Episode=Episode,Perturbations=Perturbations,
                   PHASES=PHASES,digest=lambda p:sha256(Path(p).read_bytes()).hexdigest(),write=write)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),namespace)
    namespace['Collector']().run()
    assert len(reset_calls)==1 and len(controls)==70
    assert len(manifest['episodes'])==6
    assert task._hybrid_init_prob==1.
    for env, record in enumerate(manifest['episodes']):
        with np.load(tmp_path/record['path']) as packet:
            assert len(packet['action'])==65+env
            assert len(packet['history'])==66+env
            assert packet['history'][-1,0]==65+env
            assert np.array_equal(packet['action'],np.stack([v[env].numpy() for v in controls[:65+env]]))
            assert not packet['progress_mask'].any()
        if env==0:
            assert record['perturbation_tick']==-1
        else:
            assert record['perturbation_tick']>=1
        with np.load(tmp_path/record['diagnostics']) as sidecar:
            assert sidecar['q'].shape == (66+env,18)
            assert sidecar['hand_root'].shape == (66+env,13)
            assert sidecar['q'][-1,0] == pytest.approx((65+env)*.001,abs=1e-6)
            assert not sidecar['contact_valid'][0] and sidecar['contact_valid'][1:].all()
    assert torch.all(torch.stack(controls)[65:,0]==.02)  # zero request, then native noise


def test_real_cli_rejects_missing_expert_before_gpu_or_isaac_import(tmp_path):
    output=ROOT/'outputs/consequence-evaluator'/('missing-expert-'+tmp_path.name)
    command=[sys.executable,str(TASK/'tools/run/collect_continuous.py'),
             '--route-config',str(ROOT/'src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json'),
             '--asset-root',str(tmp_path),'--motions',str(tmp_path/'missing-motion'),
             '--cfg-env',str(tmp_path/'missing-env.yaml'),'--cfg-train',str(tmp_path/'missing-train.yaml'),
             '--output',str(output),'--gpu','3','--seed','42','--split','train']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(tmp_path))
    result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=10)
    assert result.returncode!=0 and 'missing self-trained expert' in result.stderr
    assert 'AttributeError' not in result.stderr and not output.exists()


def test_real_training_entry_rejects_occupied_gpu_before_data_or_model(tmp_path,monkeypatch):
    output=ROOT/'outputs/consequence-evaluator'/('gpu-guard-'+tmp_path.name)
    monkeypatch.setattr(sys,'argv',['train_matched.py','--data',str(tmp_path/'missing-data'),
                        '--output',str(output),'--gpu','3','--seed','42'])
    monkeypatch.setattr(subprocess,'check_output',lambda *args,**kwargs:'12345\n')
    with pytest.raises(RuntimeError,match='GPU is occupied; no model is loaded'):
        runpy.run_path(str(TASK/'tools/run/train_matched.py'),run_name='__main__')
    assert not output.exists()
