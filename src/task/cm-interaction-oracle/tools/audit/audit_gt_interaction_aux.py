#!/usr/bin/env python3
"""Frozen latent/value readout on common source-policy full MC episodes."""
from pathlib import Path
import sys
import argparse
import json
import hashlib
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[5]
sys.path[:0]=[str(ROOT),str(ROOT/'src/task/cm-interaction-oracle/src')]
from gt_interaction_aux import InteractionDecoder, rollout_targets
from critic_ranking import FrozenCritic
from src.task.CmResidual.paired_evaluation import fingerprint

ARMS=('plain','conditioned','shuffle','stopgrad')
SOURCE=ROOT/'outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ridge(features, target, train, test):
    x=features.double();y=target.double()
    mean=x[train].mean(0);scale=x[train].std(0).clamp_min(1e-4)
    x=(x-mean)/scale
    ymean=y[train].mean(0);yc=y-ymean
    gram=x[train].T@x[train]/train.sum()
    weights=torch.linalg.solve(gram+.01*torch.eye(x.shape[1],device=x.device,dtype=x.dtype),
                               x[train].T@yc[train]/train.sum())
    prediction=x@weights+ymean
    mse=(prediction[test]-y[test]).square().mean(0)
    r2=1-mse/((y[test]-y[test].mean(0)).square().mean(0)).clamp_min(1e-12)
    return dict(mse=mse.cpu().tolist(),r2=r2.cpu().tolist()),prediction.cpu()


class Latent(nn.Module):
    def __init__(self,ckpt):
        super().__init__()
        weights={k.replace('_orig_mod.',''):v for k,v in ckpt['model'].items()}
        layers=[]
        for idx in (0,2,4,6):
            key=f'a2c_network.actor_mlp.{idx}'
            w,b=weights[key+'.weight'],weights[key+'.bias']
            layer=nn.Linear(w.shape[1],w.shape[0]);layer.weight.data.copy_(w);layer.bias.data.copy_(b)
            layers.extend((layer,nn.ReLU()))
        self.net=nn.Sequential(*layers)
        rms=ckpt['running_mean_std'];self.register_buffer('mean',rms['running_mean'].float())
        self.register_buffer('variance',rms['running_var'].float())
        self.eval();self.requires_grad_(False)
    def forward(self,x):return self.net(((x-self.mean)/torch.sqrt(self.variance+1e-5)).clamp(-5,5))


def paired(a,b):
    delta=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    rng=np.random.default_rng(294)
    boots=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
    return dict(gain=float(delta.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),
                rescue=int(((np.asarray(a)>0)&(np.asarray(b)==0)).sum()),
                harm=int(((np.asarray(a)==0)&(np.asarray(b)>0)).sum()))


def training_contract(root, smoke=False):
    from_checkpoint=torch.load(SOURCE,map_location='cpu',weights_only=False)
    raw=[];reports={};checkpoints={}
    n=1 if smoke else 64
    for arm in ARMS:
        log=(root/arm/'run.log').read_text()
        def events(prefix):return [json.loads(l[len(prefix):]) for l in log.splitlines() if l.startswith(prefix)]
        init=events('REF2DEX_GT_INIT ')[0];physics=events('REF2DEX_GT_INITIAL_PHYSICS ')[0]
        grad=events('REF2DEX_GT_GRAD ')[0];epochs=events('REF2DEX_GT_EPOCH ')
        if len(epochs)!=n or epochs[-1]['epoch']!=260+n or any(v['minibatches']!=48 for v in epochs):
            raise ValueError('fixed PPO epoch/update count mismatch')
        if grad['auxiliary_critic_grad_norm']!=0 or not init['rng_unchanged']:
            raise ValueError('wrong auxiliary trunk or RNG')
        if (grad['applied_latent_grad_norm']>0)!=(arm in ('conditioned','shuffle')):
            raise ValueError('wrong applied encoder gradient')
        raw.append((init['source_model'],init['rms'],init['decoder'],physics))
        files=list((root/arm/'train').rglob('GRAB.pth'))
        if len(files)!=1:raise ValueError('checkpoint missing/ambiguous')
        ckpt=torch.load(files[0],map_location='cpu',weights_only=False);checkpoints[arm]=files[0]
        if ckpt['epoch']!=260+n or ckpt['frame']!=n*2048:raise ValueError('PPO frame/epoch mismatch')
        oldstep=[float(v['step']) for v in from_checkpoint['optimizer']['state'].values() if 'step' in v]
        newstep=[float(v['step']) for v in ckpt['optimizer']['state'].values() if 'step' in v]
        # Source PPO params retain Adam history; decoder groups have fresh steps.
        if set(newstep)-{n*48} != {v+n*48 for v in oldstep}:
            raise ValueError('source optimizer continuation update count mismatch')
        reports[arm]=dict(initial=init,initial_physics=physics,gradient=grad,epochs=epochs,
                         checkpoint_sha256=sha(files[0]),epoch=ckpt['epoch'],frame=ckpt['frame'])
    if any(v!=raw[0] for v in raw):raise ValueError('initial matched weights/RMS/decoder/physics mismatch')
    a=torch.load(checkpoints['plain'],map_location='cpu',weights_only=False)
    d=torch.load(checkpoints['stopgrad'],map_location='cpu',weights_only=False)
    # Whole native models and normalizers, not decoder states.
    equal=fingerprint(a['model'])==fingerprint(d['model']) and fingerprint(a['running_mean_std'])==fingerprint(d['running_mean_std'])
    return reports,checkpoints,equal


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();root=a.run_dir.resolve();torch.set_num_threads(2);start=time.monotonic()
    reports,checkpoints,ad_equal=training_contract(root,a.smoke)
    if a.smoke:
        result=dict(status='PASS',training=reports,plain_stopgrad_native_model_rms_exact=ad_equal,
                    elapsed_seconds=time.monotonic()-start)
        (root/'smoke_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='PASS',ad_exact=ad_equal)))
        return
    results={arm:json.loads((root/'evaluation'/arm/'results.json').read_text()) for arm in (*ARMS,'source')}
    hashes={v['initial_fingerprint'] for v in results.values()}
    if len(hashes)!=1:raise ValueError('evaluation cohort initial states differ')
    poolpath=root/'evaluation/source/pool.pt';pool=torch.load(poolpath,map_location='cpu',weights_only=False)
    t,n=pool['active'].shape
    if n!=96:raise ValueError('common source pool scope changed')
    mask=pool['active'].flatten().bool();envs=torch.arange(n)[None].expand(t,-1).flatten()[mask]
    raw=pool['observations'].reshape(-1,1442)[mask].to(a.device)
    chunks=pool['action_chunks'].reshape(-1,8,18)[mask].to(a.device)
    target=pool['gt_target'].reshape(-1,19)[mask].to(a.device)
    valid=pool['target_mask'].reshape(-1)[mask].bool().to(a.device)
    returns=pool['mc_return'].flatten()[mask].to(a.device)
    train=(envs%4!=0).to(a.device);test=(envs%4==0).to(a.device)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    vsource=FrozenCritic(source).to(a.device)
    source_values=torch.cat([vsource(x) for x in raw.split(512)])
    y=torch.stack((returns,returns-source_values),-1)
    readouts={};arrays={};future={};critic={}
    initial_decoder=None
    # Common held-out actual chunks, then same-time env-rotated chunks.
    full_shuffled=pool['action_chunks'].roll(1,dims=1).reshape(-1,8,18)[mask].to(a.device)
    for arm in ARMS:
        ckpt=torch.load(checkpoints[arm],map_location='cpu',weights_only=False)
        encoder=Latent(ckpt).to(a.device)
        features=torch.cat([encoder(x) for x in raw.split(512)])
        readouts[arm],prediction=ridge(features,y,train,test);arrays[arm+'_readout']=prediction.numpy()
        decoder=InteractionDecoder().to(a.device).eval();decoder.load_state_dict(ckpt['gt_interaction_decoder'])
        pred=torch.cat([decoder(z,c) for z,c in zip(features.split(512),chunks.split(512))])
        shuffled=torch.cat([decoder(z,c) for z,c in zip(features.split(512),full_shuffled.split(512))])
        select=test&valid
        loss=F.smooth_l1_loss(pred[select],target[select],reduction='none')
        sloss=F.smooth_l1_loss(shuffled[select],target[select],reduction='none')
        future[arm]=dict(heldout_smooth_l1=float(loss.mean()),shuffle_smooth_l1=float(sloss.mean()),
            components=[float(loss[:,:3].mean()),float(loss[:,3:6].mean()),float(loss[:,6:9].mean()),float(loss[:,9:14].mean()),float(loss[:,14:].mean())])
        model_v=FrozenCritic(ckpt).to(a.device)
        vv=torch.cat([model_v(x) for x in raw.split(512)])
        critic[arm]=dict(source_policy_mc_mse=float((vv[test]-returns[test]).square().mean()),
                         policy_mismatch=True)
        arrays[arm+'_critic']=vv.cpu().numpy()
        if arm=='plain':initial_decoder=ckpt['gt_interaction_decoder']
    # B's same z but initial source decoder is a separate before-learning diagnostic.
    source_encoder=Latent(source).to(a.device)
    z0=torch.cat([source_encoder(x) for x in raw.split(512)])
    init=InteractionDecoder().to(a.device).eval();init.load_state_dict(initial_decoder)
    p0=torch.cat([init(z,c) for z,c in zip(z0.split(512),chunks.split(512))])
    initial_loss=float(F.smooth_l1_loss(p0[test&valid],target[test&valid]))
    for arm in (*ARMS,'source'):
        path=root/'evaluation'/arm/'pool.pt';data=torch.load(path,map_location='cpu',weights_only=False)
        gt,chunk,m=rollout_targets(data['before'],data['after'],data['actions'],data['dones'])
        ticks=data['sample_ticks']
        if not torch.equal(gt[ticks],data['gt_target']) or not torch.equal(chunk[ticks],data['action_chunks']) or not torch.equal(m[ticks],data['target_mask']):
            raise ValueError('GT endpoint/action/done alignment audit failed')
    comparisons={arm:paired([v['sustained'] for v in results['conditioned']['per_episode']],
                            [v['sustained'] for v in results[arm]['per_episode']]) for arm in ('plain','shuffle','stopgrad')}
    positive=all(results['conditioned']['sustained_success']>=results[arm]['sustained_success']+3 and
                 results['conditioned']['stable45']>=results[arm]['stable45'] for arm in ('plain','shuffle','stopgrad'))
    positive=positive and all(readouts['conditioned']['mse'][0]<=.95*readouts[arm]['mse'][0] for arm in ('plain','shuffle'))
    learned=future['conditioned']['heldout_smooth_l1']<=.95*initial_loss
    result=dict(status='PROMISING' if positive else ('UNPROMISING' if learned else 'UNCLEAR'),
        training=reports,plain_stopgrad_native_model_rms_exact=ad_equal,
        evaluation={k:{kk:vv for kk,vv in v.items() if kk!='per_episode'} for k,v in results.items()},
        pair_comparisons= comparisons,common_source_pool=dict(samples=int(mask.sum()),train_envs=72,test_envs=24,
            train_samples=int(train.sum()),test_samples=int(test.sum()),gt_test_samples=int((test&valid).sum()),
            pool_sha256=sha(poolpath),initial_decoder_gt_loss=initial_loss),
        latent_readouts=readouts,gt_future=future,native_critic_mc=critic,
        learned=learned,elapsed_seconds=time.monotonic()-start,
        scope='Single-seed bounded PPO Probe; actor-only independent episodes. Latent readout targets source-policy MC, not current PPO advantage.')
    arrays.update(targets=y.cpu().numpy(),train=train.cpu().numpy(),test=test.cpu().numpy(),env_id=envs.numpy(),gt=target.cpu().numpy(),valid=valid.cpu().numpy())
    np.savez_compressed(root/'readouts.npz',**arrays)
    (root/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','learned','plain_stopgrad_native_model_rms_exact','latent_readouts','gt_future','elapsed_seconds')}),flush=True)


if __name__=='__main__':main()
