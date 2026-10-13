"""Own both pinned runtimes, one GPU and a bounded simulated execution group."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'tools/audit'),str(TASK/'src')]
from audit_wam_ref2_shadow import sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--expert-recovery',action='store_true')
    p.add_argument('--controls',type=int,default=542)
    p.add_argument('--takeover-tick',type=int,default=16)
    p.add_argument('--experiment-id',default=None)
    p.add_argument('--head-dir',type=Path,default=ROOT/'outputs/trajectory-policy/wam-ref2-head-calibration-20261012-r1')
    p.add_argument('--r-dir',type=Path,default=None)
    p.add_argument('--chunk-dir',type=Path)
    p.add_argument('--effect-checkpoint',type=Path)
    p.add_argument('--deadline-seconds',type=float,default=600.)
    a=p.parse_args()
    a.experiment_id=a.experiment_id or ('P-20261012-goal-point-wam-ref2-recovery' if a.expert_recovery else 'P-20261012-goal-point-wam-ref2-execution')
    artifact_limit = {'P-20261012-goal-point-wam-ref2-startup-anchor':120_000_000,
        'P-20261012-goal-point-wam-ref22-gt-anchored':850_000_000,
        'P-20261012-goal-point-wam-ref22-multitakeover':850_000_000,
        'P-20261012-goal-point-wam-ref23-multitakeover-fit':850_000_000,
        'P-20261012-goal-point-wam-ref2-joint-retargeter':620_000_000}.get(a.experiment_id,100_000_000)
    if not 1<=a.controls<=542 or (a.expert_recovery and not 0<=a.takeover_tick<a.controls):
        raise ValueError('bounded controls and in-horizon takeover required')
    if not 1<=a.deadline_seconds<=600:
        raise ValueError('owned group deadline must be <=600s')
    a.head_dir=a.head_dir.resolve()
    from trajectory_policy.wam_retargeter_admission import retargeter_admission
    head_manifest=json.loads((a.head_dir/'manifest.json').read_text())
    if a.chunk_dir:
        from trajectory_policy.wam_effect_admission import chunk_admission
        r_path,r_identity,r_hashes=chunk_admission(a.chunk_dir)
    else:r_path,r_identity,r_hashes=retargeter_admission(a.r_dir,head_manifest)
    r_args=['--r-dir',str(a.r_dir.resolve())] if a.r_dir is not None else []
    if a.chunk_dir:r_args+=['--chunk-dir',str(a.chunk_dir.resolve())]
    if a.effect_checkpoint:r_args+=['--effect-checkpoint',str(a.effect_checkpoint.resolve())]
    out=a.output.resolve()
    if out.exists() or ROOT/'outputs/trajectory-policy' not in out.parents or not out.name.startswith('wam-ref2-execution-'):
        raise ValueError('fresh task-owned execution group required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
                               '--format=csv,noheader,nounits'],text=True).strip():
        raise RuntimeError('GPU occupied before execution group')
    scratch=ROOT/'tmp'/out.name
    if scratch.exists():
        raise FileExistsError(scratch)
    scratch.mkdir(mode=0o700,parents=True)
    address=scratch/'w.sock'
    if len(os.fsencode(str(address)))>=108:
        raise ValueError('Unix socket path too long')
    out.mkdir(parents=True)
    started=time.monotonic()
    status=dict(status='STARTING',pid=os.getpid(),physical_gpu=a.gpu,seed=292,
        run_id=out.name,experiment_id=a.experiment_id,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        started_at=time.time(),deadline=time.time()+a.deadline_seconds,scope='four simulated training-time recovery roles; no autonomous recovery claim' if a.expert_recovery else 'three simulated nominal rows; no real robot',head_directory=str(a.head_dir),controls=a.controls,takeover_tick=a.takeover_tick if a.expert_recovery else None,partial_episode=a.controls<542)
    status['retargeter_identity']=r_identity
    status['retargeter_input_sha256']=r_hashes
    status['artifact_limit_bytes']=artifact_limit
    if a.effect_checkpoint:status['artifact_limit_bytes']=1_000_000_000
    files=[Path(__file__),TASK/'tools/run/wam_native_server.py',TASK/'tools/run/execute_wam_native_bridge.py',
           TASK/'src/trajectory_policy/wam_ipc.py',TASK/'src/trajectory_policy/wam_scene_patch.py',
           TASK/'src/trajectory_policy/wam_retargeter_admission.py',TASK/'src/trajectory_policy/wam_joint.py']
    if a.expert_recovery:
        files.append(TASK/'tools/run/wam_expert_probe.py')
    if a.chunk_dir:files += [TASK/'src/trajectory_policy/wam_effect_admission.py',TASK/'src/trajectory_policy/chunk_retargeter.py']
    if a.effect_checkpoint:files.append(a.effect_checkpoint.resolve())
    status['source_sha256']={str(path):sha(path) for path in files}
    write(out/'group_status.json',status)
    processes=[];streams=[]

    def stop_owned():
        for process,script in processes:
            if process.poll() is not None:
                continue
            command=Path('/proc')/str(process.pid)/'cmdline'
            if command.exists() and os.fsencode(str(script)) in command.read_bytes().split(b'\0'):
                os.killpg(process.pid,signal.SIGTERM)

    def guard():
        if time.monotonic()-started>a.deadline_seconds:
            raise TimeoutError('whole execution group600s deadline')
        pids=set(map(int,subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True).split()))
        if pids-{process.pid for process,_ in processes}:
            raise RuntimeError('foreign GPU PID during owned group')
        paths=[path for path in (ROOT/'outputs/trajectory-policy').glob('wam-ref2-*') if path.is_dir()]
        paths += [path for path in (ROOT/'tmp').glob('wam-ref2-*') if path.is_dir()]
        paths += [ROOT/'tmp/trajectory-policy/wam-robot-head']
        if a.experiment_id in ('P-20261012-goal-point-wam-ref2-joint-retargeter','P-20261012-goal-point-wam-ref22-gt-anchored','P-20261012-goal-point-wam-ref22-multitakeover','P-20261012-goal-point-wam-ref23-multitakeover-fit'):
            paths += [p for p in (ROOT/'tmp/trajectory-policy').glob('wam*') if p.name!='wam-robot-head']
        size=sum(int(subprocess.check_output(['du','-sb',str(path)],text=True).split()[0]) for path in paths if path.exists())
        if a.effect_checkpoint:
            size=sum(int(subprocess.check_output(['du','-sb',str(path)],text=True).split()[0]) for path in (out,scratch))
            total=sum(int(line.split()[0]) for line in subprocess.check_output(['du','-sb',str(ROOT/'outputs'),str(ROOT/'tmp')],text=True).splitlines())
            if total>=400_000_000_000:raise RuntimeError('global400GB artifact cap')
        if size>=status['artifact_limit_bytes']:
            raise RuntimeError('combined ref2 artifacts>=%dMB' % (artifact_limit//1_000_000))
        status['artifact_bytes']=size
        if any(sha(path)!=digest for path,digest in status['source_sha256'].items()):
            raise ValueError('owned launcher/server/native IPC source drift')
        if any(sha(path)!=digest for path,digest in r_hashes.items()):
            raise ValueError('owned execution R evidence drift')

    try:
        server_script=TASK/'tools/run/wam_native_server.py'
        server_log=(out/'server.log').open('w');streams.append(server_log)
        server=subprocess.Popen(['/home2/wyy/miniconda3/envs/graspenv/bin/python',str(server_script),
            '--gpu',str(a.gpu),'--controls',str(a.controls),'--experiment-id',a.experiment_id,'--head-dir',str(a.head_dir),'--socket',str(address),'--output',str(out/'server'),*r_args],cwd=ROOT,
            stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((server,server_script));status['server_pid']=server.pid
        write(out/'group_status.json',status)
        for _ in range(240):
            if server.poll() is not None:
                raise RuntimeError('inference server exited before READY')
            readiness=out/'server/status.json'
            if readiness.exists() and json.loads(readiness.read_text())['status']=='READY':
                break
            guard();time.sleep(.5)
        else:
            raise TimeoutError('inference server not READY within120s')
        native_script=TASK/'tools/run/execute_wam_native_bridge.py'
        native_log=(out/'native.log').open('w');streams.append(native_log)
        native=subprocess.Popen(['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python',str(native_script),
            '--gpu',str(a.gpu),'--controls',str(a.controls),'--takeover-tick',str(a.takeover_tick),'--experiment-id',a.experiment_id,'--head-dir',str(a.head_dir),'--socket',str(address),'--server-pid',str(server.pid),'--output',str(out/'native'),*r_args,*(['--expert-recovery'] if a.expert_recovery else [])],
            cwd=ROOT,stdout=native_log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((native,native_script));status.update(status='RUNNING',native_pid=native.pid)
        write(out/'group_status.json',status)
        while native.poll() is None:
            guard();status['elapsed_s']=time.monotonic()-started
            write(out/'group_status.json',status)
            if server.poll() not in (None,0):
                raise RuntimeError('inference server failed while native active')
            time.sleep(2)
        if native.returncode!=0:
            raise RuntimeError('native execution failed with exit '+str(native.returncode))
        server.wait(timeout=30)
        if server.returncode!=0:
            raise RuntimeError('server terminal exit failure')
        guard()
        status.update(status='COMPLETED',elapsed_s=time.monotonic()-started,
            native_exit_code=native.returncode,server_exit_code=server.returncode)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started)
        stop_owned()
        raise
    finally:
        stop_owned()
        for process,_ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                script=dict((p.pid,s) for p,s in processes)[process.pid]
                command=Path('/proc')/str(process.pid)/'cmdline'
                if command.exists() and os.fsencode(str(script)) in command.read_bytes().split(b'\0'):
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=10)
        for stream in streams:
            stream.close()
        write(out/'group_status.json',status)
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
