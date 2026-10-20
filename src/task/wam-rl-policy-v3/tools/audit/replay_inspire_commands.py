"""Replay stored canonical commands in the current pure host; no BC/WAM/PPO."""
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
from wam_rl_policy_v3.inspire_state_codec import unpack_state
from trajectory_policy.wam_ipc import send,receive,peer_pid


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--sequence',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--match-collection-batch',action='store_true')
    p.add_argument('--audit-manifest',type=Path,help='same-execution measured corpus; cache supplies only native geometry')
    p.add_argument('--controls',type=int,default=100,help='bounded replay prefix, at most 1200 controls')
    a=p.parse_args()
    if not 1<=a.controls<=1200:raise ValueError('bounded replay controls required')
    if a.output.exists():raise ValueError('fresh replay output required')
    a.output.mkdir(parents=True);meta=json.loads((a.cache/'manifest.json').read_text())
    rows=json.loads(a.audit_manifest.read_text())['records'] if a.audit_manifest else meta['sequences']
    row=next(r for r in rows if r['sequence']==a.sequence and r['split']=='train')
    with np.load(Path(row['source'])/'trajectory.npz') as z:gt={k:z[k].copy() for k in ['q','dq','hand','obj','velocity','native_current_obs','action','applied','pd_targets']}
    actions=None;target=None;envs=2;seed=246
    if a.match_collection_batch:
        batch_file=Path(row['source']).parents[1]/'manifest.json';batch=json.loads(batch_file.read_text())
        envs=len(batch['sequences']);seed=batch['seed'];target=batch['sequences'].index(a.sequence)
        by_name={r['sequence']:r for r in rows};controls=[]
        for name in batch['sequences']:
            with np.load(Path(by_name[name]['source'])/'trajectory.npz') as z:controls.append(z['action'].copy())
        steps=min(a.controls,*[len(x) for x in controls]);actions=np.stack([x[:steps] for x in controls],1)
    else:gt['action']=gt['action'][:a.controls]
    if not np.array_equal(gt['action'],gt['applied']):raise ValueError('source canonical/applied differs')
    private=ROOT/'tmp'/('ireplay-'+str(os.getpid()));private.mkdir(mode=0o700);sock=private/'s'
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(120);server.bind(str(sock));server.listen(1)
    host='/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python'
    cmd=[host,'-u',str(TASK/'tools/run/inspire_physics_host.py'),'--cache',str(a.cache.resolve()),'--output',str((a.output/'physics').resolve()),
         '--socket',str(sock),'--server-pid',str(os.getpid()),'--gpu',str(a.gpu),'--envs',str(envs),'--seed',str(seed),'--seconds','600',
         '--state-transport','packed']
    cmd+=['--fixed-motions-json',str(batch_file)] if a.match_collection_batch else ['--sequence',a.sequence]
    report=dict(status='RUNNING',sequence=a.sequence,source=str(row['source']),gpu=a.gpu,canonical_replay=True,
        matched_collection_batch=a.match_collection_batch,envs=envs,optimizer_steps=0,frames=[],
        audit_manifest=str(a.audit_manifest) if a.audit_manifest else None,controls_requested=a.controls,
        git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    log=(a.output/'physics.log').open('x');proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (a.output/'process.json').write_text(json.dumps(dict(pid=os.getpid(),native_pid=proc.pid,command=cmd),indent=2)+'\n')
    connection=None;save()
    try:
        connection,_=server.accept();connection.settimeout(120)
        if peer_pid(connection)!=proc.pid:raise ValueError('unexpected replay peer')
        hello=receive(connection);state=unpack_state(hello['state'])
        def error(t,state):
            obj=np.asarray(state['objects']);hand=np.asarray(state['hand']);q=np.asarray(state['q'])
            dq=np.asarray(state['dq']);physical=np.asarray(state['physical'])
            if target is not None:
                obj=obj[target:target+1];hand=hand[target:target+1];q=q[target:target+1]
                dq=dq[target:target+1];physical=physical[target:target+1]
            rotation=Rotation.from_matrix(gt['obj'][t,:3,:3].T @ Rotation.from_quat(obj[:,3:7]).as_matrix()).magnitude()
            return dict(tick=t,q_max=float(np.abs(q-gt['q'][t]).max()),q_rmse=float(np.sqrt(np.square(q-gt['q'][t]).mean())),
                dq_max=float(np.abs(dq-gt['dq'][t]).max()),physical248_max=float(np.abs(physical-gt['native_current_obs'][t,9:257]).max()),
                object_velocity_max=float(np.abs(obj[:,7:13]-gt['velocity'][t]).max()),
                object_position_max_m=float(np.linalg.norm(obj[:,:3]-gt['obj'][t,:3,3],axis=-1).max()),
                object_rotation_max_rad=float(rotation.max()),hand_mean_m=float(np.linalg.norm(hand-gt['hand'][t],axis=-1).mean()))
        report['initial']=error(0,state);save()
        for t,action in enumerate(actions if actions is not None else gt['action']):
            commands=action if actions is not None else np.broadcast_to(action,(2,18))
            send(connection,dict(op='step',action=commands.tolist(),episodes=state['episodes'],return_pd=True))
            packet=receive(connection);state=unpack_state(packet['state']);e=error(t+1,state)
            pd=np.asarray(packet['pd_targets']);pd=pd[target:target+1] if target is not None else pd
            e['pd_target_max']=float(np.abs(pd-gt['pd_targets'][t]).max());report['frames'].append(e)
            if (t+1)%30==0:save()
        send(connection,dict(op='close'));receive(connection);proc.wait(timeout=20)
        if proc.returncode:raise RuntimeError('native replay exited '+str(proc.returncode))
        initial=report['initial'];prefix=report['frames'][:8]
        report.update(status='COMPLETED',first_pd_error=report['frames'][0]['pd_target_max'],
            prefix8_object_max_m=max(r['object_position_max_m'] for r in prefix),prefix8_hand_max_m=max(r['hand_mean_m'] for r in prefix),
            full_object_max_m=max(r['object_position_max_m'] for r in report['frames']),full_hand_max_m=max(r['hand_mean_m'] for r in report['frames']))
        report['initial_and_prefix_gate']=bool(initial['q_max']<=1e-5 and initial['object_position_max_m']<=1e-5 and initial['hand_mean_m']<=1e-5
            and report['first_pd_error']<=1e-5 and report['prefix8_object_max_m']<=.005 and report['prefix8_hand_max_m']<=.005)
        save();print(json.dumps({k:v for k,v in report.items() if k!='frames'}),flush=True)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));save();raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        if connection:connection.close()
        server.close();sock.unlink(missing_ok=True);log.close()


if __name__=='__main__':main()
