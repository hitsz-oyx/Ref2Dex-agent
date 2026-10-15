"""Reconstruct same-rollout delta1+16, audit delta0 and frozen teacher replay.

Run in the pinned native Torch2.0.1 runtime; no simulator is created.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
TASK = ROOT/'src/task/ptv3-shared-wam-v2'
NATIVE = ROOT/'third_party/DExplore/dexplore'


def observation_functions():
    # Compile only original pure tensor functions, preserving their formulas.
    import importlib.util
    import torch
    import isaacgym.torch_utils as gym_math
    spec = importlib.util.spec_from_file_location('full_teacher_torch_utils', NATIVE/'utils/torch_utils.py')
    utils = importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
    file = NATIVE/'env/tasks/base_dexplore_task.py'
    tree = ast.parse(file.read_text())
    names = ['compute_humanoid_observations_max','compute_obj_observations']
    functions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(functions)!=2:raise ValueError('original native observation functions missing')
    for function in functions:function.decorator_list=[]
    scope = dict(torch=torch,torch_utils=utils,quat_rotate=gym_math.quat_rotate,quat_mul=gym_math.quat_mul)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(file),'exec'),scope)
    return scope,utils


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--membership',type=Path,required=True)
    p.add_argument('--limit-sequences',type=int,default=0,help='engineering-only subset')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam-v2' not in out.parents:raise ValueError('fresh task output required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('assigned GPU occupied')
    if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('20GiB disk reserve')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    sys.path[:0]=['/home2/wyy/isaac-gym/isaacgym/python',str(TASK/'src')]
    import isaacgym
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation
    torch.set_num_threads(2)
    if torch.__version__!='2.0.1+cu118':raise ValueError('pinned observation arithmetic runtime required')
    started=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('1200s packing deadline')));signal.alarm(1200)
    source=a.source.resolve();manifest=json.loads((source/'manifest.json').read_text());contract=manifest['observation_contract']
    if not manifest.get('training_allowed') or not contract.get('live_future_invariance_passed'):raise ValueError('audited current corpus required')
    from types import SimpleNamespace
    membership=json.loads(a.membership.read_text())
    if membership['source']!=str(source):raise ValueError('membership/source mismatch')
    datasets={}
    for split,description in membership['splits'].items():
        records=description['records'];ticks=[np.asarray(t,dtype=np.int64) for t in description['ticks']]
        ends=np.cumsum([0]+[len(t) for t in ticks])
        def sequence(seq,records=records):
            with np.load(source/records[seq]['path']/'trajectory.npz') as f:
                return {k:f[k].copy() for k in f.files},None
        datasets[split]=SimpleNamespace(records=records,ticks_by_sequence=ticks,
            pools=[np.arange(ends[i],ends[i+1]) for i in range(len(records))],sequence=sequence)
    scope,utils=observation_functions()
    official=Path('/home2/wyy/oyx_ws/_external/dexplore_official_v120/checkpoint/inspire.pth')
    teacher=torch.load(official,map_location='cpu',weights_only=False)
    weights={k.removeprefix('_orig_mod.') if hasattr(k,'removeprefix') else (k[10:] if k.startswith('_orig_mod.') else k):v for k,v in teacher['model'].items()}
    layers=[]
    for i in [0,2,4,6]:
        w=weights['a2c_network.actor_mlp.%d.weight'%i];layer=torch.nn.Linear(w.shape[1],w.shape[0]);layer.weight.data.copy_(w);layer.bias.data.copy_(weights['a2c_network.actor_mlp.%d.bias'%i]);layers.extend([layer,torch.nn.ReLU()])
    w=weights['a2c_network.mu.weight'];layer=torch.nn.Linear(w.shape[1],w.shape[0]);layer.weight.data.copy_(w);layer.bias.data.copy_(weights['a2c_network.mu.bias']);layers.append(layer)
    teacher_net=torch.nn.Sequential(*layers).cuda().eval()
    rms=teacher['running_mean_std'];mean=rms['running_mean'].float().cuda();var=rms['running_var'].float().cuda()
    key=torch.tensor(contract['key_body_ids'],device='cuda');contact=torch.tensor(contract['contact_body_ids'],device='cuda')
    out.mkdir(parents=True);reports=[];meta={};controller=manifest['native_controller']
    global_zero_error=global_actor_error=global_pd_xyz=0.
    for split,dataset in datasets.items():
        records=dataset.records[:a.limit_sequences] if a.limit_sequences else dataset.records
        valid_records=[(i,r) for i,r in enumerate(records) if len(dataset.pools[i])]
        length=sum(len(dataset.pools[i]) for i,_ in valid_records)
        arrays={name:np.lib.format.open_memmap(out/(split+'-'+name+'.npy'),mode='w+',dtype=dtype,shape=shape)
            for name,dtype,shape in [('full','float32',(length,1442)),('current','float32',(length,721)),
                ('q','float32',(length,18)),('pd','float32',(length,18)),('action','float32',(length,18)),
                ('tick','int64',(length,)),('sequence','int64',(length,))]}
        memberships=[];cursor=0;pools=[];phase=[[],[]]
        for seq,record in valid_records:
            data,_=dataset.sequence(seq);ticks=dataset.ticks_by_sequence[seq];n=len(data['q']);b=torch.as_tensor(data['body_state'],device='cuda')
            force=torch.as_tensor(data['body_contact_forces'],device='cuda');refs=torch.as_tensor(data['reference_current'],device='cuda')
            state=torch.empty((n,13),device='cuda');state[:,:3]=torch.as_tensor(data['obj'][:,:3,3],device='cuda')
            state[:,3:7]=torch.as_tensor(Rotation.from_matrix(data['obj'][:,:3,:3]).as_quat().astype(np.float32),device='cuda');state[:,7:]=torch.as_tensor(data['velocity'],device='cuda')
            saved=torch.as_tensor(data['native_current_obs'],device='cuda')
            def branch(delta):
                ref=refs[torch.clamp(torch.arange(n,device='cuda')+delta,max=n-1)]
                body=scope['compute_humanoid_observations_max'](b[:,:,:3],b[:,:,3:7],b[:,:,7:10],b[:,:,10:13],False,True,force,contact,ref,key)
                obj,_=scope['compute_obj_observations'](b[:,:,:3],b[:,:,3:7],state,torch.zeros((n,1,3),device='cuda'),ref)
                # This measured IG has no reference dependence. Preserve exact
                # same-execution geometry samples, not a re-sampled CAD proxy.
                ig=saved[:,685:703]
                ref_ig=ref[:,184:232].reshape(n,16,3)[:,[0,3,6,9,12,15]].reshape(n,18)
                return torch.cat((body,obj,ig,ref_ig-ig),-1)
            with torch.no_grad():
                zero=branch(0);zero_error=float((zero-saved).abs().max());full=torch.cat((branch(1),branch(16)),-1)
                ix=torch.as_tensor(ticks,device='cuda');obs=full[ix]
                predicted=teacher_net(((obs-mean)/torch.sqrt(var+1e-5)).clamp(-5,5)).clamp(-1,1)
                labels=torch.as_tensor(data['applied'][ticks],device='cuda')
                actor_error=float((predicted-labels).abs().max())
                scale=torch.tensor(controller['scale'][:3],device='cuda')
                pd_xyz=float(torch.linalg.vector_norm((predicted[:,:3]-labels[:,:3])*scale,dim=-1).max()*1000)
            global_zero_error=max(global_zero_error,zero_error);global_actor_error=max(global_actor_error,actor_error);global_pd_xyz=max(global_pd_xyz,pd_xyz)
            report=dict(split=split,sequence=record['sequence'],rows=len(ticks),delta0_max_abs=zero_error,teacher_action_max_abs=actor_error,teacher_pd_xyz_max_mm=pd_xyz);reports.append(report)
            if zero_error>1e-4 or actor_error>2e-4 or pd_xyz>.2:
                (out/'failed-gate.json').write_text(json.dumps(report,indent=2)+'\n');raise ValueError('reconstructed teacher input does not reproduce original labels: '+str(report))
            sl=slice(cursor,cursor+len(ticks));arrays['full'][sl]=obs.cpu().numpy();arrays['current'][sl]=data['native_current_obs'][ticks]
            for name,key_name in [('q','q'),('pd','pd_targets'),('action','applied')]:arrays[name][sl]=data[key_name][ticks]
            arrays['tick'][sl]=ticks;arrays['sequence'][sl]=seq
            pool=np.arange(cursor,cursor+len(ticks));pools.append(pool)
            for j,mask in enumerate([ticks<16,ticks>=16]):
                if mask.any():phase[j].append(pool[mask])
            memberships.extend([dict(sequence=record['sequence'],tick=int(t)) for t in ticks]);cursor+=len(ticks)
            if len(reports)%50==0:
                consumers=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).split()))
                if consumers-{os.getpid()}:raise RuntimeError('foreign GPU consumer')
                if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('20GiB disk reserve')
                print(json.dumps(dict(sequences=len(reports),elapsed_s=time.monotonic()-started,max_teacher_pd_mm=global_pd_xyz)),flush=True)
        for array in arrays.values():array.flush()
        rng=np.random.default_rng(162);ids=[]
        for j in range(512):
            candidates=phase[0 if j<256 else 1] or pools
            ids.append(int(rng.choice(candidates[int(rng.integers(len(candidates)))])))
        np.save(out/(split+'-panel.npy'),np.array(ids))
        np.savez(out/(split+'-pools.npz'),**{'p%d'%i:x for i,x in enumerate(pools)})
        meta[split]=dict(rows=length,records=records,phase_pools=[[x.tolist() for x in ph] for ph in phase],panel=[memberships[i] for i in ids])
    final=dict(status='COMPLETED',schema='ref2dex.full-teacher-input-cache.v1',source=str(source),controller=controller,
        metadata=meta,audits=reports,delta0_max_abs=global_zero_error,teacher_action_max_abs=global_actor_error,
        teacher_pd_xyz_max_mm=global_pd_xyz,teacher_used_only_for_audit=True,teacher_deterministic=True,
        future_reference_deltas=[1,16],tau_input=False,elapsed_s=time.monotonic()-started)
    (out/'manifest.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({k:v for k,v in final.items() if k not in ['metadata','audits','controller']}),flush=True)


if __name__=='__main__':main()
