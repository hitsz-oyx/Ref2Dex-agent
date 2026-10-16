"""A two-rank update must equal a single optimizer on the concatenated rollout."""
from pathlib import Path
import multiprocessing as mp
import torch
import pytest
from wam_rl_policy.ppo import Actor,Critic,update


def _fixture():
    torch.set_num_threads(1);torch.manual_seed(17)
    actor=Actor();critic=Critic();obs=torch.randn(4,6,327);obs[:,:,-24:]=0
    with torch.no_grad():
        dist=actor.distribution(obs);latent=dist.sample()
        batch=dict(observation=obs,latent=latent,logp=dist.log_prob(latent).sum(-1),
            mean=dist.loc,std=dist.scale,value=critic(obs),reward=torch.randn(4,6)*3,
            done=torch.zeros(4,6,dtype=torch.bool),bootstrap=torch.zeros(6))
        batch['done'][-1]=True
    return actor,critic,batch


def _worker(rank,folder,minibatches):
    actor,critic,batch=_fixture()
    batch={k:(v[rank*3:(rank+1)*3] if k=='bootstrap' else v[:,rank*3:(rank+1)*3]) for k,v in batch.items()}
    torch.distributed.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=2)
    if minibatches>1:torch.randperm=lambda n,device,**kwargs:torch.arange(n,device=device)
    stats=update(actor,critic,batch,torch.optim.Adam(actor.parameters(),lr=1e-4),torch.optim.Adam(critic.parameters(),lr=3e-4),minibatches=minibatches)
    torch.save(dict(actor=actor.state_dict(),critic=critic.state_dict(),stats=stats),Path(folder)/('rank%d.pt'%rank))
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize('minibatches',[1,2])
def test_distributed_update_matches_global_rollout(tmp_path,monkeypatch,minibatches):
    workers=[mp.get_context('spawn').Process(target=_worker,args=(rank,str(tmp_path),minibatches)) for rank in range(2)]
    try:
        for process in workers:process.start()
        for process in workers:process.join(30);assert process.exitcode==0
    finally:
        for process in workers:
            if process.is_alive():process.terminate();process.join()
    actor,critic,batch=_fixture()
    if minibatches>1:monkeypatch.setattr(torch,'randperm',lambda n,device,**kwargs:torch.arange(n,device=device))
    stats=update(actor,critic,batch,torch.optim.Adam(actor.parameters(),lr=1e-4),torch.optim.Adam(critic.parameters(),lr=3e-4),minibatches=minibatches)
    ranks=[torch.load(tmp_path/('rank%d.pt'%rank),weights_only=False) for rank in range(2)]
    for name,model in [('actor',actor),('critic',critic)]:
        for key,value in model.state_dict().items():
            torch.testing.assert_close(ranks[0][name][key],ranks[1][name][key],rtol=0,atol=0)
            torch.testing.assert_close(value,ranks[0][name][key],rtol=1e-4,atol=2e-6)
    for packet in ranks:assert packet['stats']['actor_steps']==stats['actor_steps'] and abs(packet['stats']['joint_kl']-stats['joint_kl'])<1e-5
