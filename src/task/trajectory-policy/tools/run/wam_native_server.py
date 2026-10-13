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
    p.add_argument('--socket',type=Path,required=True)
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
    a=p.parse_args()
    if not 1<=a.controls<=542:
        raise ValueError('bounded controls required')
    a.output=a.output.resolve();a.socket=a.socket.resolve()
    if (a.output.exists() or ROOT/'outputs/trajectory-policy' not in a.output.parents
            or 'wam-ref2-execution-' not in a.output.parent.name
            or ROOT/'tmp' not in a.socket.parents or a.socket.exists()
            or len(os.fsencode(str(a.socket)))>=108):
        raise ValueError('fresh task-owned execution/server output and short project socket required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(a.socket.parent),
                      TRITON_CACHE_DIR=str(a.socket.parent/'triton'))
    import torch
    from trajectory_policy.wam_shadow import robot_goal_sample,active_hand_tau
    from goal_point_wam.data import goal_collate
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
    for path,digest in head_manifest['input_sha256'].items():
        if sha(path)!=digest:
            raise ValueError('head training evidence drift: '+path)
    from trajectory_policy.wam_retargeter_admission import retargeter_admission
    if a.chunk_dir:
        from trajectory_policy.wam_effect_admission import chunk_admission
        _,r_identity,r_hashes=chunk_admission(a.chunk_dir)
    else:_,r_identity,r_hashes=retargeter_admission(a.r_dir,head_manifest)
    original=ROOT/'outputs/trajectory-policy/wam-ref2-shadow-20261012-r2/code/model.py'
    base=ROOT/'outputs/goal-point-wam/goal-memory-ddp-20261012-r3/train/final.pt'
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
        from goal_point_wam.control_context import RobotControlAdapter,conditioned_actor
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
    support=ROOT/'src/task/cm-pointflow-effect-pretrain/src/oakink_wm'
    vendor=ROOT/'third_party/PointWorld'
    implementation=[support/name for name in ('model.py','data.py','pointworld.py','pointworld_performance.py')]
    implementation += list((vendor/'ptv3').rglob('*.py'))
    implementation += [vendor/'ptv3/ptv3_arch.yaml',ROOT/'src/task/goal-point-wam/src/goal_point_wam/data.py']
    implementation += [TASK/'src/trajectory_policy/wam_scene_patch.py']
    implementation += [TASK/'src/trajectory_policy/wam_retargeter_admission.py']
    implementation += [TASK/'src/trajectory_policy/wam_joint.py']
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
    a.output.mkdir(parents=True,exist_ok=False)
    if a.socket.exists() or ROOT/'tmp' not in a.socket.resolve().parents:
        raise ValueError('fresh project-owned socket required')
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(a.socket))
    os.chmod(a.socket,0o600)
    server.listen(1);server.settimeout(120)
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
        connection.settimeout(30)
        with connection:
            status['client_pid']=peer_pid(connection)
            send(connection,status)
            for query in range(a.controls):
                request=receive(connection)
                fields={'tick','object_history','hand_history','planned_object_goal'}
                if adapter is not None:fields|={'proprio','robot_goal'}
                if set(request)!=fields or request['tick']!=query:
                    raise ValueError('live H4/object-goal-only sequential request required')
                if time.monotonic()-began>600:
                    raise TimeoutError('bounded WAM inference server deadline')
                before=time.monotonic()
                sample=robot_goal_sample(np.asarray(request['object_history'],np.float32),
                    np.asarray(request['hand_history'],np.float32),
                    np.asarray(request['planned_object_goal'],np.float32),canonical,sample_id=query)
                batch={k:v.cuda() for k,v in goal_collate([sample]).items()}
                if adapter is not None:
                    for key in ('proprio','robot_goal'):
                        value=np.asarray(request[key],np.float32)
                        if not np.isfinite(value).all():raise ValueError('nonfinite robot context')
                        batch[key]=torch.tensor(value[None],device='cuda')
                    batch['robot_mask']=torch.ones(1,device='cuda',dtype=torch.bool)
                if query==0:
                    rng_cpu=torch.get_rng_state();rng_gpu=torch.cuda.get_rng_state()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    scene=model.scene(batch)
                    tokens,mask=model.goal(batch['start_object_points'],batch['goal_object_points'],batch['object_goal_valid'])
                    raw=model.observed_hand_condition(batch)
                    context=(scene,tokens,mask,raw)
                    prediction=(conditioned_actor(model,context,adapter,batch) if adapter is not None
                        else model.trajectory(scene['scene_tokens'],scene['scene_mask'],tokens,mask,raw)).float()
                tau=active_hand_tau(prediction.cpu().numpy(),batch['start_hand_valid'].cpu().numpy())[0]
                if query==0:
                    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_gpu)
                    hook=None
                    if adapter is not None:
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            residual=adapter(batch['proprio'],batch['robot_goal'],batch['robot_mask']).to(raw.dtype)
                        # Independent full-WAM forward with the same additive
                        # robot query condition; World still receives no goal.
                        hook=model.trajectory.register_forward_pre_hook(
                            lambda _,args: (*args[:4],args[4]+residual[:,None,None]))
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        direct=model(batch)['trajectory'].float()
                    if hook is not None:hook.remove()
                    direct_tau=active_hand_tau(direct.cpu().numpy(),batch['start_hand_valid'].cpu().numpy())[0]
                    error=float(np.max(np.abs(tau-direct_tau)))
                    if error>1e-5:
                        raise ValueError('batch1 original-model vs server actor-only equivalence failed')
                    status['first_query_direct_max_error_m']=error
                    write(a.output/'status.json',status)
                latency=time.monotonic()-before
                send(connection,dict(schema=SCHEMA,tick=query,tau=tau.tolist(),elapsed_s=latency,
                                     anchor_object_pose=request['object_history'][-1]))
                with (a.output/'latency.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(tick=query,elapsed_s=latency))+'\n')
                status['queries']=query+1
            if any(sha(path)!=digest for path,digest in hashes.items()):
                raise ValueError('frozen server source/input drift')
            status.update(status='COMPLETED',elapsed_s=time.monotonic()-began,
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        status.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-began)
        try:
            if request is not None:
                write(a.output/'failed_request.json',request)
                failed={key:np.asarray(request[key],np.float32) for key in
                        ('object_history','hand_history','planned_object_goal') if key in request}
                np.savez_compressed(a.output/'failed_request.npz',**failed)
                status.update(failed_query=request.get('tick'),failed_request_sha256=sha(a.output/'failed_request.npz'))
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
