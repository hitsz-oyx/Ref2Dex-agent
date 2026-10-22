"""Production-reset source-PD positive control; complete source plus real suffix.

No executor/WAM learning. Original full declarations, meshes and PD targets are
read-only. Infeasible inverse commands are explicit failures, never clipped.
"""
import argparse,json,os,shutil,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/wam-rl-policy-v3'
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src')]


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--declarations',type=Path,required=True)
    p.add_argument('--panel',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/teacher-upper-bound-validation-20261022-r1/manifest.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,choices=[4,5,6],default=4)
    p.add_argument('--suffix-controls',type=int,default=120);p.add_argument('--seconds',type=int,default=600)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents or not 60<=a.suffix_controls<=180 or not 60<=a.seconds<=900:raise ValueError('fresh bounded task-local output required')
    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if int(gpu.split(',')[1])>64:raise RuntimeError('GPU occupied: '+gpu)
    out.mkdir(parents=True);private=ROOT/'tmp'/out.name;private.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES='',TMPDIR=str(private),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.source_pd import source_pd_action
    from wam_rl_policy_v3.inspire_contract import expand_action
    from wam_rl_policy_v3.physical_safety import physical_safety,PHYSICAL_SAFETY_POLICY
    from wam_rl_policy_v3.continuous_wrist import RESET_ANGLE_POLICY
    panel=json.loads(a.panel.read_text());names=[r['sequence'] for r in panel['records']]
    manifest=json.loads(a.declarations.read_text());by_name={r['sequence']:r for r in manifest['records']}
    records=[by_name[name] for name in names];sources=[];points=[];bounds=[]
    for row in records:
        source=Path(row['source']);side=a.declarations.resolve().parent/row['sidecar']
        with np.load(source/'trajectory.npz') as z:sources.append({k:z[k].copy() for k in ('q','dq','hand','obj','pd_targets')})
        with np.load(source/'canonical.npz') as z:points.append(z['points'].copy())
        with np.load(side/'admission.npz') as z:bounds.append(z['table_bounds'].copy())
    lengths=[len(s['obj'])-1 for s in sources]
    if any(len(s['pd_targets'])!=length for s,length in zip(sources,lengths)):raise ValueError('one recorded PD target per complete-source transition required')
    n=len(names);controls=max(lengths)+a.suffix_controls
    if controls*n>6000:raise ValueError('registered control budget exceeded')
    frozen=out/'runtime-code';shutil.copytree(TASK,frozen/'src/task/wam-rl-policy-v3',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    (frozen/'src/task/trajectory-policy').symlink_to(ROOT/'src/task/trajectory-policy',target_is_directory=True)
    (out/'script.py').write_bytes(Path(__file__).read_bytes())
    (out/'before-task-scheduler.py').write_bytes(subprocess.check_output(['git','show','8fe1a29f:src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/task_scheduler.py'],cwd=ROOT))
    (out/'source.diff').write_bytes(subprocess.check_output(['git','diff','HEAD','--',str(TASK)],cwd=ROOT))
    fixed=private/'motions.json';write(fixed,dict(sequences=names));sock=private/'rpc.sock'
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(2);server.bind(str(sock));server.listen(1)
    command=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(frozen/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'),
        '--cache',str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),'--output',str(out/'physics'),'--socket',str(sock),
        '--server-pid',str(os.getpid()),'--envs',str(n),'--gpu',str(a.gpu),'--seed','195','--seconds',str(a.seconds),
        '--fixed-motions-json',str(fixed),'--state-transport','packed','--extended-state','--synchronize-setters']
    environment=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT),PYTHONPATH=str(frozen/'src/task/wam-rl-policy-v3/src')+':'+str(ROOT/'src/task/trajectory-policy/src'))
    report=dict(status='STARTING',run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_manifest=str(a.declarations.resolve()),panel=str(a.panel.resolve()),sequence_order=names,gpu_before=gpu,native_command=command,
        source_controls=lengths,suffix_controls=a.suffix_controls,registered_max_envcontrols=6000,controls_limit=controls,
        policy_updates=0,environment_controls=0,source_pd_saturation_rejections=0,clipped_envcontrols=0,maximum_roundoff_excess=0.,maximum_pd_error=0.,
        physical_safety_policy=PHYSICAL_SAFETY_POLICY,reset_angle_policy=RESET_ANGLE_POLICY,
        suffix_policy='Live integration with final recorded absolute PD target held; not a recorded teacher continuation.',
        execution_chain='source-PD -> inverse canonical action -> production native PD; no GT-tau or WAM-tau executor claim')
    write(out/'manifest.json',report);log=(out/'native.log').open('x');child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(out/'process.json',dict(client_pid=os.getpid(),native_pid=child.pid));started=time.monotonic();connection=None;trace=[];applied=[];aborts=[None]*n
    try:
        while connection is None:
            if child.poll() is not None:raise RuntimeError('native exited before handshake: '+str(child.returncode))
            if time.monotonic()-started>180:raise TimeoutError('native startup')
            try:connection,_=server.accept()
            except socket.timeout:continue
        connection.settimeout(120)
        if peer_pid(connection)!=child.pid:raise ValueError('native peer mismatch')
        hello=receive(connection);state=unpack_state(hello['state']);report['native_initial_audit']=hello['audit']
        if state['sequences']!=names or hello['audit']['reset_angle_policy']!=RESET_ANGLE_POLICY:raise ValueError('production reset/membership changed')
        offset=np.asarray(hello['audit']['controller']['offset']);scale=np.asarray(hello['audit']['controller']['scale'])
        initial={}
        for key in ('q','dq','hand'):
            expected=np.stack([s[key][0] for s in sources]);actual=np.asarray(state[key]);delta=actual-expected
            if key=='q':delta[:,3:6]=(delta[:,3:6]+np.pi)%(2*np.pi)-np.pi
            initial[key]=dict(maximum_error=float(np.abs(delta).max()),continuous_q_compared_modulo_2pi=key=='q')
        report['initial_source_comparison']=initial
        for tick in range(controls+1):
            if time.monotonic()-started>a.seconds-10 or shutil.disk_usage(ROOT).free<20*2**30:raise TimeoutError('wall/storage budget')
            keys=('q','dq','hand','objects','physical','tick','episodes','table_pose','contact_forces','dof_force','contact_force_valid','dof_force_valid','object_contact_force','palm_pose')
            numeric={k:np.asarray(state[k]).copy() for k in keys};trace.append(numeric)
            if tick==controls:break
            actions=np.zeros((n,12),np.float32);targets=np.zeros((n,18));active=np.array([abort is None for abort in aborts])
            for i in range(n):
                if not active[i]:continue
                safety=physical_safety(state,i,points[i],bounds[i],hello['audit']['physical_joint_limits'])
                if safety['abort']:aborts[i]=dict(tick=tick,reason=safety['reason'],evidence=safety);active[i]=False;continue
                raw=sources[i]['pd_targets'][min(tick,lengths[i]-1)]
                try:action,target,roundoff=source_pd_action(raw[None],numeric['q'][i:i+1],offset,scale)
                except ValueError as error:
                    aborts[i]=dict(tick=tick,reason='unrepresentable_source_pd',error=str(error));active[i]=False;report['source_pd_saturation_rejections']+=int('saturation' in str(error));continue
                actions[i]=action[0];targets[i]=target[0];report['maximum_roundoff_excess']=max(report['maximum_roundoff_excess'],roundoff)
            send(connection,dict(op='step',action=expand_action(actions).tolist(),episodes=list(state['episodes']),return_pd=True))
            reply=receive(connection);state=unpack_state(reply['state']);actual_pd=np.asarray(reply['pd_targets'])
            if active.any():report['maximum_pd_error']=max(report['maximum_pd_error'],float(np.abs(actual_pd[active]-targets[active]).max()))
            if state['episodes']!=numeric['episodes'].tolist() or not np.array_equal(np.asarray(state['tick']),numeric['tick']+1):raise ValueError('episode/tick ownership changed')
            applied.append(dict(action=actions,target=targets,actual_pd=actual_pd,active=active));report['environment_controls']+=n
            if tick%90==0:write(out/'progress.json',dict(report,status='RUNNING',tick=tick,aborts=aborts))
        send(connection,dict(op='close'));receive(connection);child.wait(timeout=20)
        if child.returncode:raise RuntimeError('native returncode '+str(child.returncode))
        report.update(status='COMPLETED',elapsed_s=time.monotonic()-started,aborts=aborts,native_returncode=child.returncode)
    except BaseException as error:
        report.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-started,aborts=aborts);raise
    finally:
        if trace:np.savez_compressed(out/'trajectory.npz',**{k:np.stack([row[k] for row in trace]) for k in trace[0]})
        if applied:np.savez_compressed(out/'controls.npz',**{k:np.stack([row[k] for row in applied]) for k in applied[0]})
        write(out/'manifest.json',report)
        if connection:
            try:connection.close()
            except Exception:pass
        server.close();log.close()
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    print(json.dumps({k:report[k] for k in ('status','environment_controls','source_pd_saturation_rejections','clipped_envcontrols','maximum_pd_error','aborts')}))


if __name__=='__main__':main()
