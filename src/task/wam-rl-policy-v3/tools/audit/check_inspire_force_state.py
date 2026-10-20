"""Bounded native two-env check of measured force timing and indexed reset."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src')]
from trajectory_policy.wam_ipc import send,receive,peer_pid
from wam_rl_policy_v3.inspire_state_codec import unpack_state
from wam_rl_policy_v3.robot_adapter import InspireAdapter
from wam_rl_policy_v3.task_annotations import task_from_teacher_archive
from wam_rl_policy_v3.task_runtime import TaskController
from wam_rl_policy_v3.executor_contract import ExecutorLayout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--sequence',default='s10_cup_drink_1');a=p.parse_args()
    if a.output.exists():raise ValueError('fresh smoke output required')
    a.output.mkdir(parents=True)
    meta=json.loads((a.cache/'manifest.json').read_text());row=next(r for r in meta['sequences'] if r['sequence']==a.sequence and r['split']=='train')
    batch=json.loads((Path(row['source']).parents[1]/'manifest.json').read_text())
    with np.load(Path(row['source'])/'trajectory.npz') as z:
        actions=z['action'][:12].copy();spec=task_from_teacher_archive(z,batch['observation_contract']['contact_body_ids'])
    g=Path(meta['records'][row['geometry']]['geometry']);g=g if g.is_absolute() else a.cache/g
    with np.load(g) as z:points=z['query'].copy();surface=z['surface'].copy()
    controllers=[TaskController(spec,points,interaction_points=surface) for _ in range(2)]
    private=ROOT/'tmp'/('force-state-'+str(os.getpid()));private.mkdir(mode=0o700)
    sock=private/'s';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.settimeout(120);server.bind(str(sock));server.listen(1)
    native='/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'
    command=[native,'-u',str(TASK/'tools/run/inspire_physics_host.py'),'--cache',str(a.cache.resolve()),
        '--output',str((a.output/'physics').resolve()),'--socket',str(sock),'--server-pid',str(os.getpid()),
        '--gpu',str(a.gpu),'--envs','2','--seed','246','--seconds','400','--state-transport','packed',
        '--sequence',a.sequence,'--extended-state']
    report=dict(status='RUNNING',engineering_only=True,gpu=a.gpu,envs=2,seconds_limit=400,
        source=str(row['source']),command=command,python=sys.executable,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        shape_feature='zero64 format fixture only; no learned policy or learned shape inference',optimizer_steps=0)
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    log=(a.output/'physics.log').open('x');proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (a.output/'process.json').write_text(json.dumps(dict(pid=os.getpid(),native_pid=proc.pid,command=command),indent=2)+'\n')
    connection=None;started=time.monotonic();save()
    def checked(packet,expected_valid):
        state=unpack_state(packet['state'])
        for field,shape in [('dof_force',(2,18)),('contact_forces',(2,5,3))]:
            value=np.asarray(state[field]);assert value.shape==shape and np.isfinite(value).all(),field
            assert not value[np.logical_not(expected_valid)].any(),field+' invalid reset force'
        for field in ('dof_force_valid','contact_force_valid'):assert state[field]==expected_valid,field
        objects=np.asarray(state['objects']);pose=np.repeat(np.eye(4)[None],2,0)
        pose[:,:3,3]=objects[:,:3];pose[:,:3,:3]=Rotation.from_quat(objects[:,3:7]).as_matrix()
        tasks=[]
        for i in range(2):
            tick=int(state['tick'][i]);result=controllers[i].observe(pose[i],InspireAdapter.canonical_hand(state['hand'][i]),
                objects[i,7:13],tick/30,contact_forces=state['contact_forces'][i],force_valid=expected_valid[i])
            tasks.append(result['task'])
        robot=InspireAdapter.state(state,np.broadcast_to(spec.object_path[-1,:3,3],(2,3)),np.zeros((2,12)),shape_feature=np.zeros((2,64)))
        observation=ExecutorLayout(InspireAdapter.robot_id,411,12,30).pack(robot.values,np.asarray(tasks))
        assert observation.shape==(2,686) and np.isfinite(observation).all()
        return state
    try:
        connection,_=server.accept();connection.settimeout(120)
        assert peer_pid(connection)==proc.pid
        state=checked(receive(connection),[False,False]);maximum_force=0.
        for action in actions:
            send(connection,dict(op='step',action=np.broadcast_to(action,(2,18)).tolist(),episodes=state['episodes']))
            state=checked(receive(connection),[True,True])
            maximum_force=max(maximum_force,float(np.abs(state['dof_force']).max()))
        before={k:np.asarray(state[k]).copy() for k in ('q','dq','hand','objects','tick','dof_force','contact_forces')}
        episodes=list(state['episodes']);send(connection,dict(op='reset',ids=[0]));packet=receive(connection)
        # Only the reset environment gets a new controller/clock. The untouched
        # environment is not observed twice at the same timestamp.
        state=unpack_state(packet['state']);assert state['episodes']==[episodes[0]+1,episodes[1]]
        controllers[0].reset()
        for field in ('dof_force_valid','contact_force_valid'):assert state[field]==[False,True]
        assert not np.asarray(state['dof_force'])[0].any() and not np.asarray(state['contact_forces'])[0].any()
        assert state['tick'][0]==0
        for field,old in before.items():assert np.array_equal(np.asarray(state[field])[1],old[1]),'reset changed neighbor '+field
        send(connection,dict(op='step',action=np.broadcast_to(actions[0],(2,18)).tolist(),episodes=state['episodes']))
        state=checked(receive(connection),[True,True])
        send(connection,dict(op='close'));receive(connection);proc.wait(timeout=20)
        if proc.returncode:raise RuntimeError('native exit '+str(proc.returncode))
        manifest=json.loads((a.output/'physics/manifest.json').read_text())
        assert manifest['maximum_pd_error']==0 and manifest['episode_action_mismatches']==0
        report.update(status='COMPLETED',elapsed_s=time.monotonic()-started,steps=13,indexed_reset_passed=True,
            initial_invalid_passed=True,post_step_valid_passed=True,neighbor_unchanged_passed=True,
            finite_force_passed=True,robot_dim=411,scheduled_dim=686,maximum_abs_dof_force=maximum_force,
            maximum_pd_error=manifest['maximum_pd_error'],episode_action_mismatches=manifest['episode_action_mismatches'])
        save();print(json.dumps(report),flush=True)
    except BaseException as error:report.update(status='FAILED',error=repr(error));save();raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if connection:connection.close()
        server.close();sock.unlink(missing_ok=True);log.close()


if __name__=='__main__':main()
