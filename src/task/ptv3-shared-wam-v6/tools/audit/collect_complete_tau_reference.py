"""Bounded source-PD continuation through raw RELEASE endpoint + settle + lookahead.

No optimizer, policy checkpoint or WAM inference. All writes stay in this Task.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
NAME = 's1_cylindersmall_pass_1'
CALIBRATION = ROOT/'outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--gpu', type=int, choices=[4, 5, 6], default=4)
    a = p.parse_args()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(TASK/'src'), str(ROOT/'src/task/wam-rl-policy-v3/src'),
                   str(ROOT/'src/task/trajectory-policy/src')]
    from ptv3_shared_wam_v6.tau_contract import gt_reference_tau, reference_coverage
    meta = json.loads((CALIBRATION/'manifest.json').read_text())
    source = Path(meta['source_pd_reference'])
    side = ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/NAME
    task_packet = json.loads((side/'raw-task.json').read_text())
    if task_packet['terminal_mode'] != 'RELEASE' or task_packet['source_fps'] != 30:
        raise ValueError('original full RELEASE at 30Hz required')
    endpoint = int(task_packet['source_frames'][-1])
    controls = endpoint + 30 + 8
    if controls > 600:
        raise ValueError('bounded source replay exceeds 600 controls')
    with np.load(source/'trajectory.npz') as z:
        targets = z['pd_targets'].copy()
    if len(targets) != endpoint:
        raise ValueError('complete original PD target stream required')
    with np.load(source/'canonical.npz') as z:
        points = z['points'].copy()
    with np.load(side/'admission.npz') as z:
        bounds = z['table_bounds'].copy()
    out = a.output.resolve()
    if TASK/'outputs' not in out.parents:
        raise ValueError('output must stay within Task/outputs')
    report = dict(status='DRY_RUN_PASS', source_sequence=NAME, source=str(source),
        task=task_packet, predicted_coverage=reference_coverage(controls+1, endpoint),
        release_source_tick=next(e['source_frame'] for e in reversed(task_packet['events']) if e['kind']=='RELEASE'),
        reference_end_policy='hold last recorded PD target after original endpoint; measured physical continuation, no state padding',
        new_assumptions=['Stationary last-PD-target continuation after the original 166 commands; not an observed teacher action suffix.',
                         'Fresh uninterrupted replay of the same sequence/targets; not bitwise continuation of the historical 81-state execution.',
                         'Original raw RELEASE is retained as declared intent, not independently certified intended release.'],
        optimizer_updates=0, maximum_environment_controls=controls, environment_controls=0,
        python=sys.executable, python_version=sys.version.split()[0], numpy=np.__version__,
        conda_env=os.environ.get('CONDA_DEFAULT_ENV'), gpu=a.gpu,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    if a.dry_run:
        print(json.dumps({k:v for k,v in report.items() if k != 'task'}, indent=2));return
    if shutil.disk_usage(TASK).free < 20*2**30:
        raise ValueError('20GiB free storage boundary')
    gpu = subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
        '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    _, memory, utilization = map(int, gpu.strip().split(','))
    if memory > 128 or utilization > 5:
        raise ValueError('selected GPU occupied: '+gpu)
    out.mkdir(parents=True, exist_ok=False)
    private = TASK/'tmp'/out.name
    private.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='', TMPDIR=str(private), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    from scipy.spatial.transform import Rotation
    from trajectory_policy.wam_ipc import send, receive, peer_pid
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.inspire_contract import ACTIVE, NATIVE_TO_WAM
    from wam_rl_policy_v3.physical_safety import physical_safety
    from wam_rl_policy_v3.task_runtime import TaskController
    from wam_rl_policy_v3.task_spec import TaskSpec
    from wam_rl_policy_v3.measured_execution import online_support
    from ptv3_shared_wam_v6.execution_contract import encoder_domain_available
    fixed = private/'motions.json';write(fixed, dict(sequences=[NAME]))
    sock = private/'rpc.sock'
    # Linux AF_UNIX is limited to 108 bytes. A task-local relative path is safe
    # because client and native host have the same cwd.
    sock_arg = str(sock.relative_to(TASK))
    os.chdir(TASK)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_arg)
    server.settimeout(120);server.listen(1)
    command = ['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python', '-u',
        str(TASK/'tools/audit/complete_reference_host.py'), '--cache',
        str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),
        '--output', str(out/'native'), '--socket', sock_arg, '--server-pid', str(os.getpid()),
        '--envs', '1', '--gpu', str(a.gpu), '--seed', '195', '--seconds', '180',
        '--fixed-motions-json', str(fixed), '--state-transport', 'packed', '--extended-state', '--synchronize-setters']
    report.update(status='STARTING', gpu_before=gpu, native_command=command)
    write(out/'manifest.json', report)
    log = (out/'native.log').open('x')
    child = subprocess.Popen(command, cwd=TASK, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    connection = None;trace = [];actions = [];clipped = 0;started = time.monotonic();task_trace = []
    try:
        connection, _ = server.accept();connection.settimeout(60)
        if peer_pid(connection) != child.pid:
            raise ValueError('native peer mismatch')
        hello = receive(connection);state = unpack_state(hello['state'])
        report['native_initial_audit'] = hello['audit']
        if state['sequences'] != [NAME]:raise ValueError('wrong sequence')
        # Match calibration's explicit reset before collecting. Never splice two runs.
        send(connection, dict(op='reset', ids=[0]));state = unpack_state(receive(connection)['state'])
        controller = TaskController(TaskSpec.from_packet(task_packet), points)
        offset = np.asarray(hello['audit']['controller']['offset'], np.float32)
        scale = np.asarray(hello['audit']['controller']['scale'], np.float32)
        terminal = None;max_pd_error = 0.;abort = None
        for t in range(controls+1):
            if time.monotonic()-started > 180 or shutil.disk_usage(TASK).free < 20*2**30:
                raise TimeoutError('wall/storage bound')
            safe = physical_safety(state, 0, points, bounds, hello['audit']['physical_joint_limits'])
            numeric = {k: np.asarray(state[k]).copy() for k in
                ('q', 'dq', 'hand', 'objects', 'physical', 'tick', 'episodes', 'table_pose',
                 'contact_forces', 'dof_force', 'contact_force_valid', 'dof_force_valid', 'object_contact_force', 'palm_pose')}
            trace.append(numeric)
            obj = numeric['objects'][0];pose = np.eye(4)
            pose[:3,:3] = Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3] = obj[:3]
            if terminal is None:
                d = controller.observe(pose, numeric['hand'][0,NATIVE_TO_WAM], obj[7:13], t/30,
                    contact_forces=numeric['contact_forces'][0], force_valid=bool(numeric['contact_force_valid'][0]),
                    supported=online_support(state,0,pose,points,bounds), workspace_ok=not safe['abort'])['decision']
                task_trace.append(dict(tick=t, phase=d.phase.name, task_success=bool(d.task_complete),
                    path_complete=bool(d.path_complete), failure_reason=d.failure_reason))
                if d.terminated:terminal=task_trace[-1]
            if safe['abort']:
                abort=dict(tick=t, **safe);break
            if t == controls:break
            target = targets[min(t, len(targets)-1)][None].copy();q = numeric['q']
            target[:,3:6] += 2*np.pi*np.round((q[:,3:6]-target[:,3:6])/(2*np.pi))
            value=(target-offset)/scale
            value[:,:6]=(target[:,:6]-q[:,:6]-offset[:6])/scale[:6];value[:,6:]=2*value[:,6:]-1
            full=np.zeros_like(value);full[:,ACTIVE]=value[:,ACTIVE]
            clipped += int((np.abs(full)>1.000001).any())
            action=np.clip(full,-1,1).astype(np.float32)
            send(connection,dict(op='step',action=action.tolist(),episodes=list(state['episodes']),return_pd=True))
            reply=receive(connection);state=unpack_state(reply['state']);actions.append(action[0,ACTIVE])
            max_pd_error=max(max_pd_error,float(np.abs(np.asarray(reply['pd_targets'])-target).max()))
            if state['episodes'] != numeric['episodes'].tolist() or not np.array_equal(np.asarray(state['tick']),numeric['tick']+1):
                raise ValueError('episode/tick discontinuity')
            report['environment_controls']+=1
        data={k:np.stack([s[k] for s in trace]) for k in trace[0]}
        hand=data['hand'][:,0]
        tau,valid=zip(*(gt_reference_tau(hand,t) for t in range(len(hand))))
        domain=[]
        for t,obj in enumerate(data['objects'][:,0]):
            pose=np.eye(4);pose[:3,:3]=Rotation.from_quat(obj[3:7]).as_matrix();pose[:3,3]=obj[:3]
            domain.append(encoder_domain_available(points,hand[t,NATIVE_TO_WAM],pose))
        valid=np.asarray(valid)&domain
        tau=np.asarray(tau);tau[~valid]=0
        np.savez_compressed(out/'reference.npz',**data, action=np.asarray(actions),tau=tau,tau_valid=valid)
        with np.load(CALIBRATION/'source-pd/trajectory.npz') as old:
            n=min(len(old['hand']),len(hand))
            prefix_error=float(np.abs(old['hand'][:n,0]-hand[:n]).max())
        coverage=reference_coverage(len(hand),endpoint) if len(hand)>endpoint else None
        report.update(status='COMPLETED' if coverage and coverage['endpoint_plus_1s_tau_covered'] and not abort else 'INCOMPLETE',
            coverage=coverage, actual_valid_tau_controls=int(valid[:len(actions)].sum()),
            physical_safety_abort=abort, clipped_controls=clipped, maximum_pd_target_error=max_pd_error,
            original81_prefix_max_hand_error_m=prefix_error,
            original81_bitwise_prefix_preserved=bool(prefix_error==0),
            full_release_success=bool(terminal and terminal['task_success']), task_terminal=terminal,
            task_trace=task_trace, elapsed_s=time.monotonic()-started)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error));raise
    finally:
        if connection:
            try:send(connection,dict(op='close'));receive(connection)
            except Exception:pass
            connection.close()
        server.close()
        if child.poll() is None:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=10)
        log.close()
        report['native_returncode']=child.returncode
        write(out/'manifest.json',report)
    print(json.dumps({k:report.get(k) for k in ('status','coverage','actual_valid_tau_controls',
        'full_release_success','task_terminal','original81_prefix_max_hand_error_m','clipped_controls','native_returncode')},indent=2))
    if report['status']!='COMPLETED' or child.returncode:raise SystemExit(1)


if __name__=='__main__':main()
