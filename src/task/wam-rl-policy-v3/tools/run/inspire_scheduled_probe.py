"""Fit or serve WAMv4 + spatial scheduler + scratch Inspire native-command BC.

GPU model process stays in graspenv, native simulator in its pinned Torch2.0
runtime. One TRAIN sequence only; no full-corpus prediction cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'),str(TASK.parent/'trajectory-policy/src')]


def write(path, packet):
    Path(path).write_text(json.dumps(packet,indent=2)+'\n')


def geometry_inputs(source, output):
    object_name=source.name.split('_')[1]
    import numpy as np
    with np.load(source/'canonical.npz') as z:
        points,normals = z['points'].astype(np.float32),z['normals'].astype(np.float32)
    normals /= np.linalg.norm(normals,axis=-1,keepdims=True)
    query=points[np.linspace(0,len(points)-1,32,dtype=int)]
    path=output/'geometry.npz'; np.savez_compressed(path,surface=points,normals=normals,query=query)
    write(output/'manifest.json',dict(records=[dict(object_code='dexplore/'+object_name,scale=1,geometry=str(path))]))
    dense=output/'surface-manifest.json'
    write(dense,dict(schema='ref2dex.wam-v4-dense-surface.v2',status='COMPLETED',objects=[dict(
        key='dexplore/'+object_name+'@1',path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())]))
    return dense


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['fit','serve'])
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wam',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/step80000-model.pt')
    p.add_argument('--bc',type=Path)
    p.add_argument('--socket',type=Path)
    p.add_argument('--gpu',type=int,default=6)
    p.add_argument('--seed',type=int,default=246)
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--seconds',type=int,default=900)
    a=p.parse_args(); a.output=a.output.resolve();a.source=a.source.resolve();a.wam=a.wam.resolve()
    if a.output.exists() or ROOT/'outputs/wam-rl-policy-v3' not in a.output.parents:
        raise ValueError('fresh Task output required')
    if not 1<=a.updates<=5000 or not 30<=a.seconds<=1200: raise ValueError('bounded probe required')
    if a.mode=='serve' and (not a.bc or not a.socket):raise ValueError('serve needs BC and project socket')
    a.output.mkdir(parents=True); scratch=ROOT/'tmp'/a.output.parent.name;scratch.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),OMP_NUM_THREADS='2',
        TRITON_CACHE_DIR=str(scratch/'triton'),PYTHONDONTWRITEBYTECODE='1')
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Inspire model deadline')));signal.alarm(a.seconds)
    import numpy as np
    import torch
    from wam_rl_policy_v3.inspire_contract import ACTIVE,CONTRACT,InspireConditioner,inspire_tau,executor_input,expand_action
    from wam_rl_policy_v3.inspire_executor import InspireExecutor,command
    from wam_rl_policy_v3.wam_v4_runtime import V4Runtime,collate_v4
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
    started=time.monotonic(); dense=geometry_inputs(a.source,a.output)
    object_name=a.source.name.split('_')[1]
    runtime=V4Runtime(a.wam,[dict(object_code='dexplore/'+object_name,scale=1)],a.output,
        dense_manifest=dense,allow_geometry_replacement=True)
    with np.load(a.source/'trajectory.npz') as values:
        data={k:values[k].copy() for k in values.files}
    controller=InspireConditioner(runtime,data['reference_obj'])
    identity=dict(wam=runtime.conditioning_identity,geometry=runtime.geometry_bindings,
        scheduler=controller.scheduler.config.__dict__,goal_type='se3_pose',conditioning=CONTRACT)
    meta=dict(schema='ref2dex.inspire-wam-probe.v1',status='RUNNING',mode=a.mode,source=str(a.source),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,torch=torch.__version__,gpu=a.gpu,seed=a.seed,identity=identity,
        role='single TRAIN sequence engineering; not generalization or PPO outcome')
    write(a.output/'run.json',meta)
    if a.mode=='fit':
        # Admission is explicit: a fresh collector batch is not a whole-corpus
        # approved dataset. This consumes only one TRAIN engineering episode.
        corpus=json.loads((ROOT/'outputs/ptv3-shared-wam-v2/dexplore-current-rollout-20261015-r1/manifest.json').read_text())
        membership={r['sequence']:r['split'] for r in corpus['records']}
        native=json.loads((a.source.parents[1]/'manifest.json').read_text())
        if (membership.get(a.source.name)!='train' or native['status']!='COMPLETED'
                or not native['observation_contract'].get('live_future_invariance_passed')
                or any(not np.isfinite(v).all() for v in data.values())):
            raise ValueError('TRAIN same-execution and live current-observation audit required')
        write(a.output/'training-admission.json',dict(training_allowed=True,engineering_only=True,
            source=str(a.source),inherited_split='train',scope='single TRAIN episode scratch BC',
            native_capture_completed=True,current_future_invariance_passed=True))
        count=len(data['action'])
        if count>600 or data['native_current_obs'].shape!=(count+1,721):raise ValueError('bounded same-execution data required')
        x=np.zeros((count,986),np.float32); samples=[];indices=[]
        goals=[]
        for i in range(count):
            sample,goal=controller.prepare(i,data['obj'][i],data['hand'][i]);goals.append(goal['goal_pose'])
            x[i]=executor_input(data['native_current_obs'][i],np.zeros((8,11,3)),False)
            if sample is not None:samples.append(sample);indices.append(i)
        taus=np.zeros((count,8,11,3),np.float32)
        for start in range(0,len(samples),32):
            batch={k:v.to(runtime.device) for k,v in collate_v4(samples[start:start+32]).items()}
            prediction=runtime.predict_hand(batch)['trajectory'].cpu().numpy()
            for j,trajectory in enumerate(prediction):
                i=indices[start+j];taus[i]=inspire_tau(trajectory,data['obj'][i]);x[i]=executor_input(data['native_current_obs'][i],taus[i],True)
        # Labels are commands captured before the native in-place PD conversion.
        if not np.array_equal(data['action'],data['applied']):raise ValueError('canonical action capture mismatch')
        y=data['action'][:,ACTIVE].astype(np.float32)
        xt,yt=torch.tensor(x,device='cuda'),torch.tensor(y,device='cuda')
        packet=dict(schema='ref2dex.inspire-scheduled-bc.v1',identity=identity,
            input_mean=xt.mean(0),input_std=xt.std(0,unbiased=False).clamp_min(.01),
            output_mean=yt.mean(0),output_std=yt.std(0,unbiased=False).clamp_min(.001))
        xn=((xt-packet['input_mean'])/packet['input_std']).clamp(-10,10)
        yn=(yt-packet['output_mean'])/packet['output_std']
        model=InspireExecutor().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        for step in range(a.updates):
            optimizer.zero_grad(set_to_none=True);loss=(model(xn)-yn).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            if step%500==0:print(json.dumps(dict(update=step,loss=float(loss),elapsed_s=time.monotonic()-started)),flush=True)
        with torch.no_grad():error=(command(model,xt,packet)-yt).abs().mean(0).cpu().numpy()
        packet['model']={k:v.cpu() for k,v in model.state_dict().items()}
        for k in ('input_mean','input_std','output_mean','output_std'):packet[k]=packet[k].cpu()
        packet.update(source=str(a.source),updates=a.updates,seed=a.seed)
        torch.save(packet,a.output/'bc.pt')
        np.savez_compressed(a.output/'training-conditions.npz',observation=x,action=y,tau=taus,goal_pose=np.array(goals))
        meta.update(status='COMPLETED',frames=count,updates=a.updates,normalized_mse=float(loss),
            train_command_mae=error.tolist(),elapsed_s=time.monotonic()-started)
    else:
        packet=torch.load(a.bc,map_location='cuda',weights_only=False)
        if packet.get('schema')!='ref2dex.inspire-scheduled-bc.v1' or packet['identity']!=identity:
            raise ValueError('BC/WAM/geometry/scheduler contract mismatch')
        model=InspireExecutor().cuda().eval();model.load_state_dict(packet['model'],strict=True)
        a.socket=a.socket.resolve()
        if ROOT/'tmp' not in a.socket.parents or a.socket.exists() or len(os.fsencode(a.socket))>=108:
            raise ValueError('fresh project-local short socket required')
        a.socket.parent.mkdir(exist_ok=True);os.chmod(a.socket.parent,0o700)
        listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);listener.settimeout(180);listener.bind(str(a.socket));listener.listen(1)
        print(json.dumps(dict(status='READY',pid=os.getpid())),flush=True)
        observations=[];actions=[];poses=[];goals=[];valid=[];times=[]
        try:
            connection,_=listener.accept();connection.settimeout(90);meta['native_peer_pid']=peer_pid(connection)
            with connection,torch.inference_mode():
                while True:
                    request=receive(connection)
                    if request.get('schema')!='ref2dex.inspire-v4-ipc.v1':raise ValueError('wrong native schema')
                    tick=request['tick']
                    if request['final']:
                        if tick!=len(actions):raise ValueError('missing executed step')
                        send(connection,dict(tick=tick,complete=True));break
                    step_start=time.monotonic();pose=np.asarray(request['pose']);hand=np.asarray(request['hand'])
                    sample,goal=controller.prepare(tick,pose,hand)
                    tau=np.zeros((8,11,3),np.float32)
                    if sample is not None:
                        batch={k:v.to(runtime.device) for k,v in collate_v4([sample]).items()}
                        tau=inspire_tau(runtime.predict_hand(batch)['trajectory'][0].cpu().numpy(),pose)
                    obs=executor_input(np.asarray(request['current']),tau,sample is not None)
                    active=command(model,torch.tensor(obs,device='cuda')[None],packet)[0].cpu().numpy()
                    action=expand_action(active)
                    send(connection,dict(tick=tick,action=action.tolist()))
                    observations.append(obs);actions.append(action);poses.append(pose);goals.append(goal['goal_pose']);valid.append(sample is not None);times.append(time.monotonic()-step_start)
            np.savez_compressed(a.output/'inference.npz',observation=observations,action=actions,pose=poses,goal_pose=goals,valid=valid,latency_s=times)
            meta.update(status='COMPLETED',controls=len(actions),valid_tau_steps=sum(valid),mean_latency_s=float(np.mean(times)),p95_latency_s=float(np.quantile(times,.95)),elapsed_s=time.monotonic()-started)
        finally:
            listener.close();a.socket.unlink(missing_ok=True)
    write(a.output/'run.json',meta);print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
