"""Frozen robot WAM head inference in graspenv, serving one local native peer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/goal-point-wam/src'),
             str(ROOT/'src/task/cm-pointflow-effect-pretrain/src'),str(TASK/'tools/audit')]
from audit_wam_ref2_shadow import sha,write
from trajectory_policy.wam_ipc import SCHEMA,send,receive,peer_pid


def scene_patch_admission(packet, manifest, result):
    """Authenticate optional dec0 provenance before application in either runtime."""
    from trajectory_policy.wam_joint import JOINT_SCHEMA, policy_patch_admission
    if packet.get('schema')==JOINT_SCHEMA or 'policy_patch' in packet or 'policy_patch_schema' in packet:
        return policy_patch_admission(packet,manifest,result)
    if any(k.startswith('scene_patch') and k not in {'scene_patch','scene_patch_schema'} for k in packet):
        raise ValueError('unknown scene patch packet field')
    if set(packet.get('head', {})) != {'trajectory', 'hand_query'}:
        raise ValueError('only original trajectory/hand-query head modules admitted')
    if 'scene_patch' not in packet:
        if ('scene_patch_schema' in packet or 'scene_patch_schema' in manifest
                or 'scene_patch_prefix' in manifest):
            raise ValueError('scene patch marker without patch')
        return dict(applied=False)
    from trajectory_policy.wam_scene_patch import SCENE_PATCH_SCHEMA, SCENE_PATCH_PREFIX
    patch = packet['scene_patch']
    if (packet.get('scene_patch_schema') != SCENE_PATCH_SCHEMA
            or manifest.get('scene_patch_schema') != SCENE_PATCH_SCHEMA
            or manifest.get('scene_patch_prefix') != SCENE_PATCH_PREFIX
            or packet.get('schema') != 'ref2dex.wam-visited-pd-dec0.v1'
            or manifest.get('schema') != packet['schema']
            or not isinstance(patch, dict) or not patch
            or not all(isinstance(k,str) and k.startswith(SCENE_PATCH_PREFIX) for k in patch)):
        raise ValueError('unknown scene patch schema or module prefix')
    scene_hash = result.get('scene_final_hash', '')
    if (not isinstance(scene_hash, str) or len(scene_hash) != 64
            or any(c not in '0123456789abcdef' for c in scene_hash)):
        raise ValueError('completed scene hash required')
    parent = Path(manifest.get('warm_head_dir', '')).resolve() / 'head.pt'
    digest = packet.get('parent_head_sha256')
    if (ROOT/'outputs/trajectory-policy' not in parent.parents or not parent.is_file()
            or digest != manifest.get('warm_head_sha256')
            or digest != manifest.get('input_sha256', {}).get(str(parent))
            or digest != sha(parent)):
        raise ValueError('scene patch parent identity differs')
    return dict(applied=True, schema=SCENE_PATCH_SCHEMA, prefix=SCENE_PATCH_PREFIX,
        scene_sha256=scene_hash, parent_head_sha256=digest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',type=Path)
    p.add_argument('--bench',action='store_true',
                   help='time serial and batched forwards, then exit without a socket')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--controls',type=int,default=542)
    p.add_argument('--experiment-id',default='P-20261012-goal-point-wam-ref2-execution')
    p.add_argument('--head-dir',type=Path,default=ROOT/'outputs/trajectory-policy/wam-ref2-head-calibration-20261012-r1')
    p.add_argument('--r-dir',type=Path,default=None)
    p.add_argument('--chunk-dir',type=Path)
    p.add_argument('--allow-repaired-chunk',action='store_true',help='explicit qualified R replacement; actor training identity unchanged')
    p.add_argument('--effect-checkpoint',type=Path)
    p.add_argument('--control-checkpoint',type=Path)
    p.add_argument('--reference-packet',type=Path)
    p.add_argument('--open-queries',action='store_true',
                   help='accept a declared number of live queries; ticks need not be 0..N-1')
    a=p.parse_args()
    if a.open_queries:
        if not 1<=a.controls<=200000:
            raise ValueError('open query budget required')
    elif not 1<=a.controls<=542:
        raise ValueError('bounded controls required')
    a.output=a.output.resolve()
    if a.bench:
        if a.socket is not None:
            raise ValueError('benchmark does not open a socket')
        cache_root=a.output.parent
    else:
        if a.socket is None:
            raise ValueError('socket required')
        a.socket=a.socket.resolve()
        cache_root=a.socket.parent
        if (ROOT/'tmp' not in a.socket.parents or a.socket.exists()
                or len(os.fsencode(str(a.socket)))>=108):
            raise ValueError('fresh task-owned execution/server output and short project socket required')
    if (a.output.exists() or ROOT/'outputs/trajectory-policy' not in a.output.parents
            or 'wam-ref2-execution-' not in a.output.parent.name):
        raise ValueError('fresh task-owned execution/server output and short project socket required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(cache_root),
                      TRITON_CACHE_DIR=str(cache_root/'triton'))
    import torch
    from trajectory_policy.wam_shadow import active_hand_tau
    from oakink_wm.pointworld_performance import install_fused_hilbert
    torch.set_num_threads(2)
    torch.manual_seed(292)
    torch.backends.cuda.matmul.allow_tf32=True
    head_dir=a.head_dir.resolve()
    if ROOT/'outputs/trajectory-policy' not in head_dir.parents:
        raise ValueError('task-owned authenticated head required')
    head_result=json.loads((head_dir/'result.json').read_text())
    head_manifest=json.loads((head_dir/'manifest.json').read_text())
    if head_manifest['status']!='COMPLETED' or sha(head_dir/'head.pt')!=head_result['head_sha256']:
        raise ValueError('authenticated final robot head required')
    reference_mode=a.reference_packet is not None
    # Reference mode loads the three-domain base, not this head. The later
    # horizon argument defaults to 24 and does not change the competent command.
    if not reference_mode and not a.open_queries:
        for path,digest in head_manifest['input_sha256'].items():
            if sha(path)!=digest:
                raise ValueError('head training evidence drift: '+path)
    from trajectory_policy.student_pd import competent_r_identity
    competent=ROOT/'outputs/trajectory-policy/competent-pd-fit-20261011-r1/tau.pt'
    original=ROOT/'outputs/trajectory-policy/wam-ref2-shadow-20261012-r2/code/model.py'
    base=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
    if reference_mode and (a.chunk_dir or a.r_dir or a.effect_checkpoint or a.control_checkpoint):
        raise ValueError('reference packet replaces chunk R, effect and the 27-D control adapter')
    if reference_mode:
        r_identity,r_hashes=competent_r_identity(competent,sha)
    else:
        from trajectory_policy.wam_retargeter_admission import retargeter_admission
        if a.chunk_dir:
            from trajectory_policy.wam_effect_admission import chunk_admission
            _,r_identity,r_hashes=chunk_admission(a.chunk_dir)
        else:_,r_identity,r_hashes=retargeter_admission(a.r_dir,head_manifest)
    stats=ROOT/'outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json'
    state=torch.load(base,map_location='cpu',weights_only=False)
    if sha(original)!=state['identity']['model_sha256'] or sha(stats)!=state['identity']['stats_sha256']:
        raise ValueError('frozen original WAM implementation/stats drift')
    spec=importlib.util.spec_from_file_location('frozen_wam_native_bridge',original)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_fused_hilbert()
    model=module.GoalPointWAM(json.loads(stats.read_text()),state['config']).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    conditioned_actor=None
    if reference_mode:
        from trajectory_policy.student_pd import student_control_identity
        from goal_point_wam.reference_adapter import ReferenceAdapter
        packet,control_identity=student_control_identity(a.reference_packet,base,competent,sha)
        model.trajectory.load_state_dict(packet['trajectory'],strict=True)
        adapter=ReferenceAdapter().cuda().eval().requires_grad_(False)
        adapter.load_state_dict(packet['adapter'],strict=True)
        patch_identity=dict(applied=False)
        effect_identity=dict(applied=False)
        head=None
        del packet
    else:
        head=torch.load(head_dir/'head.pt',map_location='cpu',weights_only=False)
        if (head['base_checkpoint_sha256']!=sha(base)
                or head_manifest.get('base_checkpoint_sha256')!=head['base_checkpoint_sha256']
                or set(head['head'])!={'trajectory','hand_query'}):
            raise ValueError('head belongs to another base or module set')
        patch_identity=scene_patch_admission(head,head_manifest,head_result)
        for name,value in head['head'].items():
            getattr(model,name).load_state_dict(value,strict=True)
        if patch_identity['applied'] and 'policy_patch' in head:
            from trajectory_policy.wam_joint import load_policy_patch
            load_policy_patch(model,head,patch_identity)
        elif patch_identity['applied']:
            from trajectory_policy.wam_scene_patch import load_scene_patch
            from calibrate_wam_robot_startup import network_hash
            load_scene_patch(model.scene,head['scene_patch'])
            if network_hash(model.scene)!=patch_identity['scene_sha256']:
                raise ValueError('applied scene differs from authenticated final scene')
        from trajectory_policy.wam_effect_admission import effect_admission
        effect_identity,effect_values=effect_admission(a.effect_checkpoint,head_dir,a.chunk_dir,a.allow_repaired_chunk)
        if effect_values is not None:
            if Path(effect_identity['base_checkpoint']).resolve()!=base:raise ValueError('C base checkpoint differs')
            model.trajectory.load_state_dict(effect_values,strict=True)
        del effect_values
        from trajectory_policy.wam_control_admission import control_admission
        control_identity,control_packet=control_admission(a.control_checkpoint,effect_identity,r_identity)
        adapter=None
        if control_packet is not None:
            from goal_point_wam.control_context import RobotControlAdapter
            from goal_point_wam.control_context import conditioned_actor as conditioned_actor
            adapter=RobotControlAdapter().cuda().eval().requires_grad_(False)
            adapter.load_state_dict(control_packet['adapter'],strict=True)
            model.trajectory.load_state_dict(control_packet['trajectory'],strict=True)
            del control_packet
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if not torch.isfinite(parameter).all():
            raise ValueError('nonfinite frozen weight')
    del head,state
    canonical_path=ROOT/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz'
    with np.load(canonical_path,allow_pickle=False) as stream:
        canonical={k:stream[k].copy() for k in stream.files}
    chunk_net=None
    if a.chunk_dir and not reference_mode:
        from trajectory_policy.chunk_retargeter import load_chunk
        chunk_packet=torch.load(r_identity['checkpoint'],map_location='cuda',weights_only=False)
        chunk_net=load_chunk(chunk_packet,'cuda')
        for parameter in chunk_net.parameters():
            parameter.requires_grad_(False)
        del chunk_packet
    support=ROOT/'src/task/cm-pointflow-effect-pretrain/src/oakink_wm'
    vendor=ROOT/'third_party/PointWorld'
    implementation=[support/name for name in ('model.py','data.py','pointworld.py','pointworld_performance.py')]
    implementation += list((vendor/'ptv3').rglob('*.py'))
    implementation += [vendor/'ptv3/ptv3_arch.yaml',ROOT/'src/task/goal-point-wam/src/goal_point_wam/data.py']
    implementation += [TASK/'src/trajectory_policy/wam_scene_patch.py',
                       TASK/'src/trajectory_policy/scene_workspace.py']
    implementation += [TASK/'src/trajectory_policy/wam_retargeter_admission.py']
    implementation += [TASK/'src/trajectory_policy/wam_joint.py',
                       TASK/'src/trajectory_policy/wam_batch.py']
    if chunk_net is not None:
        implementation += [TASK/'src/trajectory_policy/chunk_retargeter.py',
                           TASK/'src/trajectory_policy/learned_retargeter.py',
                           TASK/'src/trajectory_policy/retargeter_data.py']
    if patch_identity['applied']:
        implementation += [TASK/'tools/run/calibrate_wam_robot_startup.py']
    hashes={str(path):sha(path) for path in (Path(__file__),TASK/'src/trajectory_policy/wam_ipc.py',
        TASK/'src/trajectory_policy/wam_shadow.py',original,base,stats,head_dir/'head.pt',
        head_dir/'manifest.json',head_dir/'result.json',canonical_path,*implementation)}
    hashes.update(r_hashes)
    if a.effect_checkpoint:
        hashes[str(a.effect_checkpoint.resolve())]=effect_identity['checkpoint_sha256']
        hashes[str(TASK/'src/trajectory_policy/wam_effect_admission.py')]=sha(TASK/'src/trajectory_policy/wam_effect_admission.py')
    if a.control_checkpoint:
        hashes[str(a.control_checkpoint.resolve())]=control_identity['checkpoint_sha256']
        for path in (TASK/'src/trajectory_policy/wam_control_admission.py',
                     ROOT/'src/task/goal-point-wam/src/goal_point_wam/control_context.py'):
            hashes[str(path)]=sha(path)
    if reference_mode:
        hashes[str(Path(a.reference_packet).resolve())]=control_identity['checkpoint_sha256']
        for path in (TASK/'src/trajectory_policy/student_pd.py',
                     ROOT/'src/task/goal-point-wam/src/goal_point_wam/reference_adapter.py'):
            hashes[str(path)]=sha(path)
    if a.bench:
        from trajectory_policy.wam_batch import run_forward_benchmark
        report=run_forward_benchmark(model,adapter,canonical,reference_mode,conditioned_actor,chunk_net)
        a.output.mkdir(parents=True,exist_ok=False)
        write(a.output/'bench.json',report)
        print(json.dumps(report),flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=False)
    if a.socket.exists() or ROOT/'tmp' not in a.socket.resolve().parents:
        raise ValueError('fresh project-owned socket required')
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(a.socket))
    os.chmod(a.socket,0o600)
    server.listen(1);server.settimeout(180 if a.open_queries else 120)
    status=dict(schema=SCHEMA,status='READY',pid=os.getpid(),physical_gpu=a.gpu,seed=292,
        input_sha256=hashes,base_sha256=sha(base),head_sha256=sha(head_dir/'head.pt'),
        torch_version=torch.__version__,training_allowed=False,queries=0,
        scene_patch_identity=patch_identity,controls=a.controls,experiment_id=a.experiment_id,
        retargeter_identity=r_identity,effect_identity=effect_identity,control_identity=control_identity)
    write(a.output/'status.json',status)
    began=time.monotonic()
    request=None
    try:
        connection,_=server.accept()
        connection.settimeout(180 if a.open_queries else 30)
        with connection:
            status['client_pid']=peer_pid(connection)
            send(connection,status)
            from trajectory_policy.scene_workspace import mask_scene_outliers
            from trajectory_policy.wam_batch import MAX_BATCH, chunk_base, collate_robot_batch, predict_trajectory, request_rows
            goal_key='reference_goal' if reference_mode else 'robot_goal'
            # Open queries visit phase resets, which can put a few more
            # points outside the grid than a frame-0 episode. The limit is
            # per row, so a batch does not sum four environments together.
            outlier_limit=64 if a.open_queries else 8
            for query in range(a.controls):
                request=receive(connection)
                rows,batched=request_rows(request,goal_key,adapter is not None,a.open_queries,query)
                if time.monotonic()-began>(14400 if a.open_queries else 600):
                    raise TimeoutError('bounded WAM inference server deadline')
                before=time.monotonic()
                batch=collate_robot_batch(rows,canonical,goal_key,query*MAX_BATCH,adapter is not None)
                dropped=mask_scene_outliers(batch, model.scene.voxel_origin_m, float(model.scene.voxel_size_m),
                                            limit=outlier_limit)
                dropped_rows=list(batch.get('_dropped_per_row',[0]*len(rows)))
                if dropped:
                    status['workspace_points_dropped']=status.get('workspace_points_dropped',0)+dropped
                if query==0:
                    rng_cpu=torch.get_rng_state();rng_gpu=torch.cuda.get_rng_state()
                prediction=predict_trajectory(model,adapter,batch,reference_mode,conditioned_actor)
                tau=active_hand_tau(prediction.cpu().numpy(),batch['start_hand_valid'].cpu().numpy())
                if query==0:
                    # The actor-vs-full-model check stays on one row. A second
                    # full-batch forward would double the first-step activation.
                    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_gpu)
                    one=collate_robot_batch(rows[:1],canonical,goal_key,query*MAX_BATCH,adapter is not None)
                    mask_scene_outliers(one, model.scene.voxel_origin_m, float(model.scene.voxel_size_m),
                                        limit=outlier_limit)
                    one_prediction=predict_trajectory(model,adapter,one,reference_mode,conditioned_actor)
                    one_tau=active_hand_tau(one_prediction.cpu().numpy(),one['start_hand_valid'].cpu().numpy())[0]
                    hook=None
                    if adapter is not None:
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            raw_one=model.observed_hand_condition(one)
                            residual=adapter(one['proprio'],one[goal_key],one['robot_mask']).to(raw_one.dtype)
                        hook=model.trajectory.register_forward_pre_hook(
                            lambda _,args: (*args[:4],args[4]+residual[:,None,None]))
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        direct=model(one)['trajectory'].float()
                    if hook is not None:hook.remove()
                    direct_tau=active_hand_tau(direct.cpu().numpy(),one['start_hand_valid'].cpu().numpy())[0]
                    error=float(np.max(np.abs(one_tau-direct_tau)))
                    if error>1e-5:
                        raise ValueError('batch1 original-model vs server actor-only equivalence failed')
                    status['first_query_direct_max_error_m']=error
                    write(a.output/'status.json',status)
                base=None
                if chunk_net is not None and 'retarget_state' in rows[0]:
                    base=chunk_base(chunk_net,rows,tau)
                latency=time.monotonic()-before
                if batched:
                    responses=[]
                    for index,row in enumerate(rows):
                        item=dict(env_id=row['env_id'],tick=row['tick'],tau=tau[index].tolist(),
                                  anchor_object_pose=row['object_history'][-1],
                                  workspace_points_dropped=dropped_rows[index])
                        if base is not None:
                            item['base_action']=base[index].tolist()
                        responses.append(item)
                    send(connection,dict(schema=SCHEMA,batch=True,elapsed_s=latency,responses=responses))
                else:
                    reply_tick=rows[0]['tick'] if a.open_queries else query
                    reply=dict(schema=SCHEMA,tick=reply_tick,tau=tau[0].tolist(),elapsed_s=latency,
                               anchor_object_pose=rows[0]['object_history'][-1],
                               workspace_points_dropped=dropped_rows[0])
                    if base is not None:
                        reply['base_action']=base[0].tolist()
                    send(connection,reply)
                with (a.output/'latency.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(tick=query,elapsed_s=latency,batch=len(rows)))+'\n')
                status['queries']=query+1
            if any(sha(path)!=digest for path,digest in hashes.items()):
                raise ValueError('frozen server source/input drift')
            status.update(status='COMPLETED',elapsed_s=time.monotonic()-began,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-began)
        try:
            if request is not None:
                payload=request['batch'][0] if isinstance(request.get('batch'),list) and request['batch'] else request
                write(a.output/'failed_request.json',request)
                failed={key:np.asarray(payload[key],np.float32) for key in
                        ('object_history','hand_history','planned_object_goal') if key in payload}
                np.savez_compressed(a.output/'failed_request.npz',**failed)
                status.update(failed_query=payload.get('tick'),failed_request_sha256=sha(a.output/'failed_request.npz'))
                status['failed_request_json_sha256']=sha(a.output/'failed_request.json')
                if 'batch' in locals():
                    valid_xyz=batch['xyz'][batch['point_valid']]
                    status.update(failed_scene_xyz_min_m=valid_xyz.amin(0).cpu().tolist(),
                                  failed_scene_xyz_max_m=valid_xyz.amax(0).cpu().tolist(),
                                  workspace_origin_m=model.scene.voxel_origin_m.detach().cpu().tolist(),
                                  voxel_m=model.scene.voxel_size_m)
        except Exception as recording_error:
            status['failure_recording_error']=repr(recording_error)
        raise
    finally:
        write(a.output/'status.json',status)
        server.close()
        a.socket.unlink(missing_ok=True)


if __name__=='__main__':
    main()
