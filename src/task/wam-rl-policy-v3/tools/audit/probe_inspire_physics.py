"""Bounded native-only reset/step reproducer: no WAM, actor, or PPO update."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src')]
from trajectory_policy.wam_ipc import send,receive
from wam_rl_policy_v3.inspire_state_codec import unpack_state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',type=Path,default=TASK/'tools/run/inspire_physics_host.py')
    p.add_argument('--sequence',default='s1_stamp_lift')
    p.add_argument('--object-manifest',type=Path)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--resets',type=int,default=2)
    p.add_argument('--seed',type=int,default=246)
    p.add_argument('--sync',action='store_true')
    p.add_argument('--extended',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents or not 2<=a.envs<=512 or not 1<=a.steps<=128 or not 0<=a.resets<=3:
        raise ValueError('fresh bounded native-only diagnostic required')
    out.mkdir(parents=True);scratch=ROOT/'tmp'/out.name;scratch.mkdir()
    address=scratch/'s';listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.settimeout(180);listener.bind(str(address));listener.listen(1)
    cmd=[Path('/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'),'-u',a.host.resolve(),
         '--cache',ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache',
         '--output',out/'native','--socket',address,'--server-pid',os.getpid(),
         '--gpu',a.gpu,'--envs',a.envs,'--seed',a.seed,'--seconds',240,'--state-transport','packed',
         '--admission-manifest',ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json']
    if a.object_manifest:cmd+=['--object-manifest',a.object_manifest.resolve()]
    else:cmd+=['--sequence',a.sequence]
    if a.sync:cmd+=['--synchronize-setters']
    if a.extended:cmd+=['--extended-state']
    info=dict(status='RUNNING',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              argv=list(map(str,cmd)),steps=0,resets=a.resets,canonical_action='zero18',wam=False,ppo=False)
    def save():(out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    save();connection=None;child=None;started=time.monotonic()
    try:
        with (out/'native.log').open('x') as log:
            child=subprocess.Popen(info['argv'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info['native_pid']=child.pid;save()
            connection,_=listener.accept();connection.settimeout(60)
            hello=receive(connection);state=unpack_state(hello['state']);info['native_audit']=hello['audit']
            for _ in range(a.resets):
                send(connection,dict(op='reset',ids=list(range(a.envs))))
                state=unpack_state(receive(connection)['state'])
            for tick in range(a.steps):
                old=list(state['episodes'])
                send(connection,dict(op='step',episodes=old,action=[[0.]*18 for _ in range(a.envs)]))
                state=unpack_state(receive(connection)['state'])
                if old!=state['episodes'] or not (state['tick']==tick+1).all():
                    raise ValueError('native action/clock mismatch')
                info['steps']=tick+1;save()
            send(connection,dict(op='close'));receive(connection)
            child.wait(timeout=30)
            if child.returncode:raise RuntimeError('native exit '+str(child.returncode))
            info['status']='PASS'
    except BaseException as error:
        info.update(status='FAIL',error=repr(error))
    finally:
        if connection:connection.close()
        listener.close();address.unlink(missing_ok=True)
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        info.update(native_exit=child.returncode if child else None,elapsed_s=time.monotonic()-started);save()
    print(json.dumps(info),flush=True)
    if info['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
