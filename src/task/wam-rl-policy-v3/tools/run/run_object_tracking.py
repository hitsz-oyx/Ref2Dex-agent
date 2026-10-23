"""Bounded object-first tracking probe; no tau, models, schema or gain changes."""
import argparse, json, os, runpy, shutil, socket, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
TASK=ROOT/'src/task/wam-rl-policy-v3'


def short_sockets():
    for operation in ('bind','connect'):
        original=getattr(socket.socket,operation)
        def short(self,address,*args,_original=original,**kwargs):
            if self.family==socket.AF_UNIX and isinstance(address,str) and len(os.fsencode(address))>=108:
                path=Path(address);fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
                try:return _original(self,'/proc/self/fd/{}/{}'.format(fd,path.name),*args,**kwargs)
                finally:os.close(fd)
            return _original(self,address,*args,**kwargs)
        setattr(socket.socket,operation,short)


def write(path,data):
    partial=path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');partial.replace(path)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--native-entry':
        short_sockets();script=sys.argv[2];sys.argv=sys.argv[2:];runpy.run_path(script,run_name='__main__');return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,choices=[4,5,6,7],default=4)
    p.add_argument('--seconds',type=int,default=7200)
    p.add_argument('--sequence',default='s1_cylindersmall_pass_1')
    p.add_argument('--declarations',type=Path,default=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json')
    p.add_argument('--matrix',choices=['sentinel','full'],default='full')
    p.add_argument('--reference-npz',type=Path,help='Generic world-frame object SE3: arrays times (seconds) and poses (N,4,4); overrides shape matrix')
    p.add_argument('--stages',choices=['A','B','both'],default='both')
    p.add_argument('--reference-start',type=Path,help='Reuse frozen world object start for a paired stage-only run')
    a=p.parse_args();out=a.output.resolve()
    if ROOT/'outputs/wam-rl-policy-v3' not in out.parents or not 60<=a.seconds<=7200:raise ValueError('bounded task-local output required')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise ValueError('do not overwrite an existing run')
    disk=shutil.disk_usage(ROOT)
    if disk.free<22*2**30:raise RuntimeError('storage guard: leave 20GiB plus registered 2GiB output')
    gpu=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(gpu.strip())>64:raise RuntimeError('assigned GPU occupied')
    scratch=out/'scratch';scratch.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES='',TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/trajectory-policy/src')]
    import numpy as np
    from scipy.spatial.transform import Rotation
    from trajectory_policy.wam_ipc import send,receive
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.inspire_contract import expand_action
    from wam_rl_policy_v3.source_pd import source_pd_action
    from wam_rl_policy_v3.pose_consumer import pose7
    from wam_rl_policy_v3.physical_safety import physical_safety
    from wam_rl_policy_v3.teacher_quality import table_support
    from wam_rl_policy_v3.object_tracking import ObjectPoseTracker,ObjectTrajectory,make_trajectory,hand_setpoint,errors,joint_margin,SHAPES
    external=None
    if a.reference_npz:
        with np.load(a.reference_npz) as z:external=ObjectTrajectory(z['times'],z['poses'])
        if external.times[0]!=0 or external.times[-1]>90:raise ValueError('external reference requires t0=0 and <=90 seconds')
    rows=json.loads(a.declarations.read_text())['records'];row=next(r for r in rows if r['sequence']==a.sequence)
    with np.load(Path(row['source'])/'trajectory.npz') as z:source=z['pd_targets'].copy()
    with np.load(Path(row['source'])/'canonical.npz') as z:points=z['points'].copy()
    with np.load(a.declarations.parent/row['sidecar']/'admission.npz') as z:bounds=z['table_bounds'].copy()
    fixed=out/'motions.json';write(fixed,dict(sequences=[a.sequence]))
    # A fixed source snapshot makes a parallel session's later edits irrelevant.
    runtime=out/'runtime-code';shutil.copytree(TASK,runtime/'src/task/wam-rl-policy-v3',symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','outputs'))
    (runtime/'src/task/trajectory-policy').symlink_to(ROOT/'src/task/trajectory-policy',target_is_directory=True)
    frozen=runtime/'src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py'
    short_sockets();sock=out/'rpc.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(2);server.bind(str(sock));server.listen(1)
    cmd=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(Path(__file__).resolve()),'--native-entry',str(frozen),
         '--cache',str(ROOT/'outputs/wam-rl-policy-v2/inspire-shadow-parity-20261019-r1/cache'),'--output',str(out/'physics'),
         '--socket',str(sock),'--server-pid',str(os.getpid()),'--envs','1','--gpu',str(a.gpu),'--seed','195','--seconds',str(a.seconds),
         '--fixed-motions-json',str(fixed),'--state-transport','packed','--extended-state','--synchronize-setters','--physics-backend','gpu-cpu','--object-tracking-probe']
    manifest=dict(status='STARTING',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),native_command=cmd,
                  sequence=a.sequence,policy_updates=0,model_calls=0,tau_used=False,maximum_controls=300000,maximum_new_bytes=2*2**30,
                  disk_free_before=disk.free,gpu=a.gpu,client_python=sys.executable,controls=0,results=[],unmeasured=[],rigid_A_scope='kinematic grasp constraint at control samples; not load-bearing welded dynamics')
    write(out/'manifest.json',manifest)
    log=(out/'native.log').open('x');environment=dict(os.environ,REF2DEX_RUN_ROOT=str(ROOT))
    child=subprocess.Popen(cmd,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
    write(out/'process.json',dict(client_pid=os.getpid(),native_pid=child.pid));start=time.monotonic();conn=None
    try:
        while conn is None:
            if child.poll() is not None:raise RuntimeError('native startup exit '+str(child.returncode))
            if time.monotonic()-start>180:raise TimeoutError('startup timeout')
            try:conn,_=server.accept()
            except socket.timeout:pass
        conn.settimeout(120);hello=receive(conn);state=unpack_state(hello['state']);audit=hello['audit'];manifest['native_audit']=audit
        offset=np.asarray(audit['controller']['offset']);scale=np.asarray(audit['controller']['scale']);limits=audit['physical_joint_limits']
        roles=audit['tracking_body_roles'];names=audit['tracking_body_names']
        def rpc(command):
            nonlocal state
            send(conn,command);reply=receive(conn);state=unpack_state(reply['state']);return reply
        def step(action):
            if time.monotonic()-start>a.seconds-20 or manifest['controls']>=300000 or shutil.disk_usage(ROOT).free<20*2**30:raise TimeoutError('wall/control/storage budget')
            previous_tick=int(state['tick'][0]);previous_episode=state['episodes'][:]
            reply=rpc(dict(op='step',action=expand_action(action[None]).tolist(),episodes=previous_episode,return_pd=True))
            if state['episodes']!=previous_episode or int(state['tick'][0])!=previous_tick+1:raise ValueError('reset/tick ownership corruption')
            manifest['controls']+=1
            return np.asarray(reply['pd_targets'])[0]
        def collisions():
            bad=[]
            for c in state['tracking_contacts']:
                i,j=c['body0'],c['body1'];ri=roles[i] if i>=0 else 'ground';rj=roles[j] if j>=0 else 'ground'
                relevant=('table' in (ri,rj) and ('hand' in (ri,rj) or 'object' in (ri,rj))) or ri==rj=='hand'
                if relevant and c['force']>.1:
                    bad.append(dict(c,roles=[ri,rj],names=[names[i] if i>=0 else 'ground',names[j] if j>=0 else 'ground']))
            return bad
        initial_reference=None if a.reference_start is None else np.load(a.reference_start)
        cells=[(s,r,v) for s in SHAPES for r in (.02,.05,.10) for v in (.03,.10,.25)]
        if a.matrix=='sentinel':cells=cells[:1]
        if external is not None:cells=[('external',0.,0.)]
        stages=('A','B') if a.stages=='both' else (a.stages,)
        plan=[dict(shape=s,scale_m=r,requested_mean_speed_m_s=v,stage=stage,feedforward=ff) for s,r,v in cells for stage in stages for ff in (False,True)]
        write(out/'registered-matrix.json',plan)
        stop_p0=False
        for case_index,case in enumerate(plan):
            if stop_p0:
                manifest['unmeasured'].append(dict(case,reason='NOT_RUN_P0'));continue
            folder=out/('case%03d'%case_index);folder.mkdir();rpc(dict(op='reset',ids=[0]));np.savez_compressed(folder/'initial.npz',q=state['q'],dq=state['dq'],objects=state['objects'],hand=state['hand'],tick=state['tick'])
            held_since=None;acquisition=[];acquired=False;abort=None;initial_z=float(state['objects'][0][2])
            for tick in range(len(source)+120):
                safety=physical_safety(state,0,points,bounds,limits)
                if safety['abort']:abort=dict(kind='ACQUISITION_FAILURE',safety=safety,tick=tick);break
                obj=np.asarray(state['objects'])[0];pose=pose7(obj)
                supported=bool(table_support(pose[None],points,np.asarray(state['table_pose'])[0],bounds,np.asarray(state['object_contact_force']))[0][0])
                force=np.linalg.norm(np.asarray(state['contact_forces'])[0],axis=-1)
                held=bool(state['contact_force_valid'][0] and (force>=.1).sum()>=2 and not supported and obj[2]-initial_z>=.05)
                held_since=tick if held and held_since is None else held_since if held else None
                acquisition.append(dict(tick=tick,q=np.asarray(state['q'])[0].tolist(),object=obj.tolist(),forces=force.tolist(),supported=supported,held=held))
                if held_since is not None and (tick-held_since)/30>=.3-1e-10:acquired=True;break
                action,_,_=source_pd_action(source[min(tick,len(source)-1)][None],np.asarray(state['q']),offset,scale);step(action[0])
            write(folder/'acquisition.json',acquisition)
            if not acquired:
                result=dict(case,index=case_index,status='ACQUISITION_FAILURE',abort=abort,acquisition_controls=len(acquisition),tracked_controls=0)
                manifest['results'].append(result);write(folder/'result.json',result)
                stop_p0=True;manifest['stop_reason']='No qualified initial grasp; do not expand trajectory shapes';write(out/'manifest.json',manifest);continue
            obj=pose7(np.asarray(state['objects'])[0]);palm=pose7(np.asarray(state['palm_pose'])[0]);relative=np.linalg.inv(palm)@obj
            if initial_reference is None:
                initial_reference=obj.copy() if external is None else external.poses[0].copy()
                np.save(out/'reference-start.npy',initial_reference)
            if external is None:curve,meta=make_trajectory(case['shape'],case['scale_m'],case['requested_mean_speed_m_s'],initial_reference)
            else:
                curve=external;end_error=errors(curve.poses[-1],curve.poses[0])
                meta=dict(duration_s=float(curve.times[-1]),closed=end_error[0]<1e-8 and end_error[1]<1e-8,reference_input=str(a.reference_npz.resolve()))
            if meta['duration_s']>90:raise ValueError('trajectory physical-time cap')
            np.savez_compressed(folder/'reference.npz',times=curve.times,poses=curve.poses,hand_object=relative)
            tracker=ObjectPoseTracker(np.asarray(state['q'])[0],palm,offset,scale)
            if case['stage']=='A':
                rpc(dict(op='tracking_attachment',enabled=True,hand_object=np.r_[relative[:3,3],Rotation.from_matrix(relative[:3,:3]).as_quat()].tolist()))
            captures=[];metrics=[];details=[];raw_actions=[];pds=[];N=int(round(meta['duration_s']*30));last_lost=None
            for tick in range(N+1):
                now=tick/30;desired,v,w=curve.evaluate(now);actual=pose7(np.asarray(state['objects'])[0]);palm=pose7(np.asarray(state['palm_pose'])[0]);err,ang=errors(actual,desired);drift,rot_drift=errors(np.linalg.inv(palm)@actual,relative)
                margins=joint_margin(np.asarray(state['q'])[0],limits);j=int(np.argmin(margins))
                force=np.linalg.norm(np.asarray(state['contact_forces'])[0],axis=-1);contact=bool(state['contact_force_valid'][0] and (force>=.1).sum()>=2)
                last_lost=now if not contact and last_lost is None else last_lost if not contact else None
                safety=physical_safety(state,0,points,bounds,limits);bad=collisions()
                metrics.append(dict(tick=tick,time_s=now,position_error_m=err,orientation_error_rad=ang,drift_m=drift,drift_rad=rot_drift,
                                    margin_min=float(margins[j]),margin_joint=j,contact=contact,collision_pairs=bad,safety=safety,
                                    target_speed_m_s=float(np.linalg.norm(v)),target_omega_rad_s=float(np.linalg.norm(w)),
                                    joint_margins=[None if not np.isfinite(x) else float(x) for x in margins]))
                captures.append({k:np.asarray(state[k]).copy() for k in ('q','dq','objects','palm_pose','hand','tick','contact_forces','object_contact_force','table_pose')})
                if safety['abort']:
                    abort=dict(kind='limit' if safety['reason']=='physical_joint_limit' else 'collision',tick=tick,safety=safety)
                    if case['scale_m']==.02 and case['requested_mean_speed_m_s']==.03 and safety['reason']=='physical_joint_limit':
                        stop_p0=True;manifest['stop_reason']='Lowest-parameter joint safety abort: stop shape expansion per preregistration'
                    break
                if bad:abort=dict(kind='collision',tick=tick,pairs=bad);break
                if case['stage']=='B' and (drift>.05 or last_lost is not None and now-last_lost>=.2-1e-10):abort=dict(kind='slip',tick=tick,drift_m=drift);break
                if tick==N:break
                target,hv,hw=hand_setpoint(desired,v,w,relative)
                # Rigid fixture A fixes the grasp. Real-contact B retains the
                # existing MeasuredGraspConsumer's force-based finger holding.
                raw,info=tracker.step(np.asarray(state['q'])[0],np.asarray(state['dq'])[0],palm,target,hv,hw,np.asarray(state['contact_forces'])[0],case['stage']=='B' and bool(state['contact_force_valid'][0]),case['feedforward'])
                raw_actions.append(raw);details.append(info);pds.append(step(np.clip(raw,-1,1)))
            if captures:np.savez_compressed(folder/'trajectory.npz',**{k:np.stack([s[k] for s in captures]) for k in captures[0]})
            if pds:np.savez_compressed(folder/'controls.npz',raw_action=np.asarray(raw_actions),pd_target=np.asarray(pds))
            write(folder/'ticks.json',metrics);write(folder/'commands.json',details)
            e=np.asarray([m['position_error_m'] for m in metrics]);angles=np.asarray([m['orientation_error_rad'] for m in metrics]);min_margin=min(metrics,key=lambda m:m['margin_min'])
            close_pos,close_rot=errors(pose7(np.asarray(state['objects'])[0]),initial_reference) if meta['closed'] else (None,None)
            result=dict(case,**{k:v for k,v in meta.items() if k not in case},index=case_index,status='ABORT' if abort else 'COMPLETED',abort=abort,
                        acquisition_controls=len(acquisition)-1,tracked_controls=len(pds),knots_file=str(folder/'reference.npz'),
                        position_quantiles_m=np.quantile(e,[0,.5,.9,.95,1]).tolist(),orientation_quantiles_deg=np.rad2deg(np.quantile(angles,[0,.5,.9,.95,1])).tolist(),
                        closure_error_m=close_pos,closure_error_deg=None if close_rot is None else float(np.rad2deg(close_rot)),
                        drift_max_m=max(m['drift_m'] for m in metrics),drift_max_deg=float(np.rad2deg(max(m['drift_rad'] for m in metrics))),
                        minimum_joint_margin=min_margin['margin_min'],minimum_margin_joint=min_margin['margin_joint'],
                        clipped_controls=sum(x['clipped'] for x in details),collision_ticks=sum(bool(m['collision_pairs']) for m in metrics),
                        pass_tracking=bool(not abort and np.quantile(e,.95)<=.015 and np.quantile(angles,.95)<=np.deg2rad(5) and max(m['drift_m'] for m in metrics)<=.01 and max(m['drift_rad'] for m in metrics)<=np.deg2rad(10) and all(all(v is None or v>=.02 for v in m['joint_margins']) for m in metrics) and (not meta['closed'] or close_pos<=.015 and close_rot<=np.deg2rad(5))))
            manifest['results'].append(result);write(folder/'result.json',result);write(out/'manifest.json',manifest)
            print(json.dumps(result),flush=True)
        send(conn,dict(op='close'));receive(conn);child.wait(timeout=30)
        if child.returncode:raise RuntimeError('native exit '+str(child.returncode))
        manifest.update(status='STOPPED_P0' if stop_p0 else 'COMPLETED',elapsed_s=time.monotonic()-start,native_returncode=child.returncode)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-start);raise
    finally:
        write(out/'manifest.json',manifest)
        if conn:conn.close()
        server.close();log.close()
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    print(json.dumps(dict(status=manifest['status'],controls=manifest['controls'],completed_cases=len(manifest['results']),unmeasured=len(manifest['unmeasured']))))

if __name__=='__main__':main()
