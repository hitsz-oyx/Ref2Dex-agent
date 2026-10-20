import ast,importlib.util
from pathlib import Path
import torch
from wam_rl_policy_v3.official_bridge import prepare_official,normalize_advantages


def test_upstream_run_preserved_and_action_paths_agree(tmp_path):
    root=Path(__file__).resolve().parents[4]
    official=prepare_official(root,tmp_path)
    def method(path,name):
        tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PPO')
        return ast.dump(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name))
    assert method(official/'algorithms/rl/ppo/ppo.py','run')==method(root/'third_party/UniDexGrasp2-official-ppo/dexgrasp/algorithms/rl/ppo/ppo.py','run')
    spec=importlib.util.spec_from_file_location('official_test_actor',official/'algorithms/rl/ppo/module.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    config=dict(backbone_type='pn',freeze_backbone=False,pi_hid_sizes=[16,16],vf_hid_sizes=[16,16],activation='elu',obs_dim=565)
    model=m.ActorCritic((565,),(0,),(24,),.8,config)
    x=torch.randn(4,565)
    a,lp,value,mean,sigma=model.act(x,None)
    logp,entropy,value2,mean2,sigma2=model.evaluate(x,None,a)
    torch.testing.assert_close(mean,model.act_inference(x));torch.testing.assert_close(mean,mean2)
    torch.testing.assert_close(lp,logp);torch.testing.assert_close(value,value2)
    torch.testing.assert_close(model.log_std.exp().square(),torch.full((24,),.64))
    z=torch.randn(8,4,1)
    torch.testing.assert_close(normalize_advantages(z),(z-z.mean())/(z.std()+1e-8),rtol=0,atol=0)


def test_native_evaluation_never_uses_wam_inputs():
    import numpy as np
    from wam_rl_policy_v3.official_bridge import ScheduledOfficialEnv
    env=object.__new__(ScheduledOfficialEnv)
    env.native_only=True;env.inputs=None;env.num_envs=2
    env.termination_policy='native';env.workspace_invalid=np.zeros(2,bool)
    env.ticks=np.zeros(2,int);env.initial_z=np.zeros(2)
    env.streak=np.zeros(2,int);env.longest=np.zeros(2,int)
    env.dropped_after_hold=np.zeros(2,bool)
    poses=np.repeat(np.eye(4)[None],2,axis=0)
    env.snapshot=lambda:dict(pose=poses,goal=np.zeros((2,3)))
    native=torch.randn(2,300)
    torch.testing.assert_close(env.augment(native,[0,1]),native,rtol=0,atol=0)
    assert env.inputs is None
