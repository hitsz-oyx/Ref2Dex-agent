"""GPU same-observation frozen learner/expert control errors, with replay check.

This is a bounded empirical DART-inspired candidate bank, not DART's fitted
Gaussian covariance or a causal failure-direction estimator.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within,K
from consequence_evaluator.value_outcomes import RAW_SCHEMA,PARAMETERS,task_trace
from consequence_evaluator.value_perturbations import BANK_SCHEMA,error_chunk


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--expert',type=Path,required=True);p.add_argument('--expert-sha',required=True)
    p.add_argument('--learner',type=Path,required=True);p.add_argument('--learner-sha',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--phase',choices=('hold','contact'),default='hold')
    a=p.parse_args();out=a.output.resolve();raw=a.source.resolve();started=time.monotonic()
    if out.exists() or not is_within(out,ROOT/'outputs/consequence-evaluator') or not is_within(raw,ROOT/'outputs/consequence-evaluator'):
        p.error('task-owned source and fresh bank file required')
    if (digest(a.expert)!=a.expert_sha or digest(a.learner)!=a.learner_sha):raise ValueError('actor identity drift')
    occupied=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('GPU occupied before same-H inference: '+occupied)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    import torch
    from torch import nn
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    if torch.__version__!='2.0.1+cu118':raise ValueError('matched complete archived Torch runtime required')
    m=json.loads((raw/'manifest.json').read_text())
    if (m.get('schema')!=RAW_SCHEMA or m.get('status')!='COMPLETED' or not m.get('clean_only')
            or m.get('split')!='train' or not 32<=len(m['episodes'])<=128):
        raise ValueError('completed nominal train source with at least32episodes required')
    frozen={str(raw/'manifest.json'):digest(raw/'manifest.json'),str(a.expert.resolve()):a.expert_sha,
        str(a.learner.resolve()):a.learner_sha,str(Path(__file__).resolve()):digest(__file__),
        str(Path(sys.modules[RunningMeanStd.__module__].__file__).resolve()):digest(sys.modules[RunningMeanStd.__module__].__file__)}
    actor_cfg=ROOT/'third_party/DExplore/dexplore/data/cfg/train/rlg/inspire.yaml'
    frozen[str(actor_cfg)]=digest(actor_cfg)
    cfg_text=actor_cfg.read_text()
    if 'units: [1024, 1024, 1024, 512]' not in cfg_text or 'activation: relu' not in cfg_text:
        raise ValueError('manual actor projection no longer matches the pinned native configuration')
    def actor(path):
        c=torch.load(path,map_location='cpu');s=c['model']
        if all(k.startswith('_orig_mod.') for k in s):s={k[len('_orig_mod.'):]:v for k,v in s.items()}
        layers=[];sizes=[1442,1024,1024,1024,512]
        for i,(left,right) in enumerate(zip(sizes[:-1],sizes[1:])):
            layer=nn.Linear(left,right)
            layer.load_state_dict({name:s[f'a2c_network.actor_mlp.{i*2}.{name}'] for name in ('weight','bias')},strict=True)
            layers.extend((layer,nn.ReLU()))
        mu=nn.Linear(512,18);mu.load_state_dict({name:s['a2c_network.mu.'+name] for name in ('weight','bias')},strict=True)
        layers.append(mu);net=nn.Sequential(*layers).cuda().eval()
        rms=RunningMeanStd((1442,)).cuda().eval();rms.load_state_dict(c['running_mean_std'],strict=True)
        return net,rms
    expert,expert_rms=actor(a.expert);learner,learner_rms=actor(a.learner)
    observations=[];expected=[];members=[]
    for r in m['episodes']:
        if len(members)==32:break
        if r['split']!='train' or r['assigned_phase']!='clean':raise ValueError('held-out/intervened source forbidden')
        files=[raw/r['path'],raw/r['diagnostics']]
        for path,key in zip(files,('sha256','diagnostics_sha256')):
            if not is_within(path.resolve(),raw) or digest(path)!=r[key]:raise ValueError('source identity mismatch')
            frozen[str(path)]=r[key]
        with np.load(files[0],allow_pickle=False) as f:packet={k:f[k] for k in f.files}
        with np.load(files[1],allow_pickle=False) as f:d={k:f[k] for k in f.files}
        trace=task_trace(packet,d)
        ticks=np.flatnonzero(trace['held_run']>=PARAMETERS['stable_frames'] if a.phase=='hold'
                             else trace['near']&~trace['held'])
        if not len(ticks):continue
        tick=int(ticks[0])
        if tick+K>=trace['place_start']:continue
        observations.append(packet['history'][tick:tick+K])
        expected.append(packet['action'][tick:tick+K]-d['actual_residual'][tick:tick+K])
        members.append(dict(episode=r['episode'],source_control_begin=tick,source_control_end=tick+K,phase=a.phase))
    if len(members)!=32:raise ValueError('need32complete measured train observation chunks at the requested phase')
    h=torch.as_tensor(np.concatenate(observations),device='cuda',dtype=torch.float32)
    with torch.inference_mode():
        teacher=expert(expert_rms(h)).clamp(-1,1)
        student=learner(learner_rms(h)).clamp(-1,1)
        errors=(student-teacher).reshape(32,K,18).cpu().numpy()
    replay_error=float(np.abs(teacher.cpu().numpy()-np.concatenate(expected)).max())
    if replay_error>1e-5:raise ValueError('native policy projection replay failed: '+str(replay_error))
    chunks=[];indices=[]
    for member,e in zip(members,errors):
        for gain in (.25,.5,1.):
            chunks.append(error_chunk(e,gain).tolist());indices.append(dict(member,gain=gain))
    for name in ('value_perturbations.py','value_outcomes.py','collection.py'):
        path=TASK/'src/consequence_evaluator'/name;frozen[str(path)]=digest(path)
    if any(digest(path)!=value for path,value in frozen.items()):raise ValueError('same-H source drift')
    if time.monotonic()-started>120:raise TimeoutError('fixed120s GPU inference budget')
    bank=dict(schema=BANK_SCHEMA,phase=a.phase,source_split='train',source_seed=m['seed'],
        source_actor_sha256=a.expert_sha,learner_sha256=a.learner_sha,sources=frozen,chunks=chunks,members=indices,
        semantics='same-observation learner minus expert normalized control; bounded stage-conditioned empirical replay candidates',
        covariance_fitted=False,causal_failure_direction_established=False,gains=[.25,.5,1.],
        limits=dict(residual=.2,wrist_translation=.05),
        sampling='uniform member with replacement, same train-derived bank for independent groups',
        replay_max_abs_error=replay_error,same_h_error_rms=float(np.sqrt(np.mean(errors**2))),
        physical_gpu=a.gpu,elapsed_s=time.monotonic()-started,
        peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(bank,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:bank[k] for k in ('replay_max_abs_error','same_h_error_rms','elapsed_s','peak_gpu_memory_mib')},indent=2))


if __name__=='__main__':main()
