"""Actual A/B/C boundary smoke with one deliberate encoder-only outage; no RL."""
import argparse,importlib.util,json,os,shutil,socket,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);private=ROOT/'tmp'/out.name;private.mkdir(parents=True,exist_ok=False)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    if any(int(r.split(',')[1])>128 for r in gpu.splitlines() if int(r.split(',')[0]) in (4,5)):raise ValueError('GPU4/5 occupied')
    frozen=out/'runtime-code'
    for task in ('ptv3-shared-wam-v6','wam-rl-policy-v3'):
        for part in ('src','tools/run'):
            destination=frozen/'src/task'/task/part;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(ROOT/'src/task'/task/part,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for task in (ROOT/'src/task').iterdir():
        destination=frozen/'src/task'/task.name
        if task.is_dir() and not destination.exists():destination.symlink_to(task.resolve(),target_is_directory=True)
    (frozen/'third_party').symlink_to((ROOT/'third_party').resolve(),target_is_directory=True)
    shutil.copyfile(__file__,out/'smoke-script.py')
    (out/'snapshot.json').write_text(json.dumps(dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        copied=['ptv3-shared-wam-v6/src','ptv3-shared-wam-v6/tools/run','wam-rl-policy-v3/src','wam-rl-policy-v3/tools/run'],
        unchanged_dependencies='other task and third_party sources are read-only links to the current baseline'),indent=2)+'\n')
    os.environ.update(CUDA_VISIBLE_DEVICES='5',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(private),OMP_NUM_THREADS='2',
        REF2DEX_RUN_ROOT=str(ROOT),REF2DEX_SAMPLE_CODE_ROOT=str(frozen),REF2DEX_V6_V5_CODE_ROOT=str(ROOT),REF2DEX_V5_BASE_CODE_ROOT=str(ROOT))
    sys.path[:0]=[str(frozen/'src/task/ptv3-shared-wam-v6/src'),str(frozen/'src/task/wam-rl-policy-v3/src'),str(ROOT/'src/task/trajectory-policy/src')]
    import torch
    from trajectory_policy.wam_ipc import send,receive,peer_pid
    from wam_rl_policy_v3.inspire_state_codec import unpack_state
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from ptv3_shared_wam_v6.online_runtime import OnlineV6Runtime
    from ptv3_shared_wam_v6.execution_contract import TASK_CONTRACT,ENCODER_POLICY,FALLBACK_POLICY
    spec=importlib.util.spec_from_file_location('samples_smoke_environment',frozen/'src/task/ptv3-shared-wam-v6/tools/run/samples_efficiency_env.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2);torch.manual_seed(196)
    model=MeasuredInspirePolicy(with_tau=True).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(ROOT/'outputs/wam-rl-policy-v3/inspire-scheduled686-bc-20261020-r4/latest.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    geometry=ROOT/'outputs/ptv3-shared-wam-v6/executor-qualification-20261022-r1/geometry'
    manifest=ROOT/'outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/manifest.json'
    forecast=OnlineV6Runtime(ROOT/'outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt',ROOT/'outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/stats.json',geometry,forecast_only=True)
    fixed=private/'motions.json';fixed.write_text(json.dumps(dict(sequences=['s10_cubemedium_pass_1','s10_cylindermedium_pass_1'])))
    sock=private/'rpc.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.settimeout(180);server.bind(str(sock));server.listen(1)
    command=['/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python','-u',str(frozen/'src/task/ptv3-shared-wam-v6/tools/run/samples_efficiency_physics.py'),
        '--cache',str(geometry),'--admission-manifest',str(manifest),'--output',str(out/'physics'),'--socket',str(sock),'--server-pid',str(os.getpid()),
        '--envs','2','--gpu','4','--seed','195','--seconds','600','--fixed-motions-json',str(fixed),'--state-transport','packed','--extended-state','--synchronize-setters']
    report=dict(status='STARTING',scope='2env x10controls x3arms engineering only; no method score, no optimizer',gpu_before=gpu,native_command=command,
        task_contract=TASK_CONTRACT,encoder_policy=ENCODER_POLICY,fallback_policy=FALLBACK_POLICY,source_full_task_evaluation=False,arms=[],environment_controls=0)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    log=(out/'native.log').open('x');child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);connection=None
    try:
        server.settimeout(2);deadline=time.monotonic()+180
        while connection is None:
            if child.poll() is not None:raise RuntimeError('native exited before handshake: '+str(child.returncode))
            if time.monotonic()>deadline:raise TimeoutError('native handshake deadline')
            try:connection,_=server.accept()
            except socket.timeout:continue
        connection.settimeout(120)
        if peer_pid(connection)!=child.pid:raise ValueError('native peer')
        hello=receive(connection);report['native_audit']=hello['audit'];initial=None
        def rpc(op,**kwargs):
            send(connection,dict(op=op,**kwargs));return receive(connection)
        for arm in ('A','B','C'):
            snapshot=rpc('reset',ids=[0,1]);state=unpack_state(snapshot['state'])
            fields={key:np.asarray(state[key]).copy() for key in ('q','dq','objects','hand','physical','table_pose','tick','contact_force_valid','dof_force_valid')}
            if initial is None:initial=fields
            if any(not np.array_equal(initial[key],value) for key,value in fields.items()):raise ValueError('arm initial states differ')
            env=module.SampleEfficiencyEnv(connection,dict(state=state,audit=hello['audit']),manifest,type('Runtime',(),dict(device=torch.device('cuda:0')))(),model,arm=arm,forecast=forecast if arm=='C' else None,geometry_cache=geometry)
            original=module.encoder_domain_available;injected=[];trace=[];invocations=0
            def gate(*args,**kwargs):
                nonlocal invocations
                slot=invocations%2;invocations+=1
                if env.steps==3 and slot==0 and not injected:injected.append(True);return False
                return original(*args,**kwargs)
            module.encoder_domain_available=gate
            try:
                for control in range(10):
                    obs=env.observations.clone()
                    with torch.inference_mode():action=model.actor(obs[:,:686]).clamp(-1,1)
                    next_obs,reward,done,info=env.step(action);report['environment_controls']+=2
                    trace.append(dict(control=control+1,tick=np.asarray(env.native['tick']).tolist(),done=done.tolist(),input_tau_valid=obs[:,685].tolist(),
                        encoder_available=info['encoder_available'].tolist(),physical_safety_abort=info['physical_safety_abort'].tolist()))
                    if done.any():raise ValueError('engineering episode unexpectedly terminated')
                    if control==3 and (next_obs[0,421:686].any() or info['encoder_available'][0]):raise ValueError('outage did not mask tau')
                if not injected or env.encoder_unavailable_steps!=7:raise ValueError('encoder control coverage mismatch')
                row=dict(arm=arm,encoder_unavailable_controls=env.encoder_unavailable_steps,decision_controls=env.decision_controls,
                    wam_covered_controls=env.wam_covered_controls,physical_safety_aborts=env.physical_safety_aborts,completed_episodes=env.completed_episodes,trace=trace)
                report['arms'].append(row)
            finally:module.encoder_domain_available=original
        report['status']='PASS'
    except BaseException as error:report.update(status='FAIL',error=repr(error));raise
    finally:
        if connection:
            try:send(connection,dict(op='close'));receive(connection)
            except Exception:pass
            connection.close()
        server.close();log.close()
        if child.poll() is None:
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=20)
        report.update(native_pid=child.pid,native_returncode=child.returncode)
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'source.diff').write_bytes(subprocess.check_output(['git','diff','HEAD','--','src/task/ptv3-shared-wam-v6','src/task/wam-rl-policy-v3']))
    print(json.dumps(dict(status=report['status'],environment_controls=report['environment_controls'],arms=[{k:v for k,v in row.items() if k!='trace'} for row in report['arms']])))


if __name__=='__main__':main()
