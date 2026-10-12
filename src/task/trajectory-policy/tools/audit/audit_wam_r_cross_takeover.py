"""Forward-only R transfer diagnostic: true t16 expert recovery versus nominal.

Offline-failed checkpoints are authenticated for diagnosis, never admitted to
physics. All future labels are measured on their own entirely expert-controlled
row. This script grants neither actor training nor any physical override.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src')]
OWNER = ROOT/'outputs/trajectory-policy'
BASE_SHA = '414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f'
SOURCE_SHA = 'b0edec52551bc9e7f528c80022f7aad946793d02e450fc234bac8eabde8c32ee'
ROLES = ['teacher','gt_tau_R','wam_R','wam_command_replay_then_expert_tick16']
BLOCKS = [(16,32),(32,48),(48,64),(64,80),(80,96)]


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):
            digest.update(block)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() or out.parent!=OWNER or not out.name.startswith('wam-ref2-') or args.gpu!=2:
        raise ValueError('fresh task-owned wam-ref2-* output and GPU2 required')
    started=time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    import torch
    from trajectory_policy.pd_retargeter import load_model,features,encode_target,decode_target,command,pd_from_action
    from trajectory_policy.wam_retargeter_admission import validate_r_packet
    torch.set_num_threads(2)
    evidence={}
    def bind(path,expected=None):
        path=Path(path)
        if not path.is_absolute():
            path=ROOT/path
        path=path.resolve(); digest=sha(path)
        if expected is not None and digest!=expected:
            raise ValueError('input identity mismatch: '+str(path))
        if str(path) in evidence and evidence[str(path)]!=digest:
            raise ValueError('conflicting input identity')
        evidence[str(path)]=digest
        return digest
    def read(path):
        bind(path)
        return json.loads(Path(path).read_text())
    base_path=OWNER/'competent-pd-fit-20261011-r1/tau.pt'
    bind(base_path,BASE_SHA)
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    bm=read(base_path.parent/'manifest.json'); br=read(base_path.parent/'result.json')
    if bm.get('status')!='COMPLETED' or br.get('checkpoint_sha256',{}).get('tau')!=BASE_SHA:
        raise ValueError('completed authenticated base R required')
    # The base's fixed identity authenticates architecture and original norms.
    base_contract=dict(base,base_R_sha256=BASE_SHA)
    validate_r_packet(base_contract,base)
    candidates={}; identities={}
    for name,dirname in [('fit1000','wam-ref2-recovery-retargeter-20261012-r1'),
                         ('alpha_half','wam-ref2-recovery-retargeter-half-20261012-r1')]:
        directory=OWNER/dirname
        manifest=read(directory/'manifest.json'); result=read(directory/'result.json')
        checkpoint=directory/'tau.pt'; checkpoint_sha=bind(checkpoint)
        if (manifest.get('schema')!='ref2dex.wam-recovery-R-fit.v1'
                or manifest.get('status')!='COMPLETED' or result.get('status')!='COMPLETED'
                or manifest.get('base_R_sha256')!=BASE_SHA or result.get('base_R_sha256')!=BASE_SHA
                or checkpoint_sha!=manifest.get('checkpoint_sha256')
                or checkpoint_sha!=result.get('checkpoint_sha256')
                or sha(directory/'result.json')!=manifest.get('result_sha256')):
            raise ValueError('completed R checkpoint/manifest/result identity differs')
        for path,digest in manifest['input_sha256'].items():
            bind(path,digest)
        admission_path=Path(manifest['admission_path']).resolve()
        if directory not in admission_path.parents or str(admission_path)!=result.get('admission_path'):
            raise ValueError('task-owned admission path mismatch')
        admission_sha=bind(admission_path,manifest['admission_sha256'])
        admission=read(admission_path)
        if (admission_sha!=result.get('admission_sha256')
                or admission.get('schema')!='ref2dex.unmixed-recovery-R-admission.v1'
                or admission.get('admitted') is not True or admission.get('R_training_allowed') is not True
                or admission.get('actor_future_hand_training_allowed') is not False
                or admission.get('mixed_policy_future_hand_training_allowed') is not False
                or tuple(admission.get(k) for k in ('role','first_tick','last_tick','horizon','windows'))!=(3,32,168,24,137)):
            raise ValueError('R-only training admission identity differs')
        for kind in ('native_source','native_manifest','root_audit','native_result','nominal_audit','nominal_trajectory'):
            path=Path(admission[kind+'_path']).resolve(); digest=admission[kind+'_sha256']
            if OWNER not in path.parents or manifest['input_sha256'].get(str(path))!=digest:
                raise ValueError('admission not bound to consumption identity: '+kind)
            bind(path,digest)
        packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
        validate_r_packet(packet,base)
        if (packet.get('input_sha256')!=manifest['input_sha256']
                or packet.get('scoped_admission_sha256')!=admission_sha
                or ('admission_sha256' in packet and packet['admission_sha256']!=admission_sha)
                or ('admission_path' in packet and packet['admission_path']!=str(admission_path))):
            raise ValueError('R packet provenance mismatch')
        if name=='fit1000' and manifest.get('updates')!=1000:
            raise ValueError('fixed1000 fit required')
        if name=='alpha_half' and (manifest.get('net_interpolation_alpha')!=.5
                or manifest.get('optimization_updates')!=0 or result.get('net_interpolation_alpha')!=.5):
            raise ValueError('fixed half interpolation required')
        candidates[name]=packet
        identities[name]=dict(checkpoint_path=str(checkpoint),checkpoint_sha256=checkpoint_sha,
            offline_screen_pass=result.get('offline_screen_pass'),recorded_gates=result.get('gates'),
            admission_sha256=admission_sha,physics_override_allowed=False)
    half=candidates['alpha_half']; fit=candidates['fit1000']
    hm=json.loads((OWNER/'wam-ref2-recovery-retargeter-half-20261012-r1/manifest.json').read_text())
    if hm.get('parent_checkpoint_sha256')!=identities['fit1000']['checkpoint_sha256']:
        raise ValueError('interpolation parent checkpoint differs')
    for name,value in half['model'].items():
        expected=base['model'][name]+.5*(fit['model'][name]-base['model'][name]) if name.startswith('net.') else base['model'][name]
        if not torch.equal(value,expected):
            raise ValueError('half checkpoint is not exact fixed interpolation: '+name)
    native=OWNER/'wam-ref2-execution-recovery-20261012-r2/native'
    nm=read(native/'manifest.json'); audit=read(native/'root_recovery_audit.json'); nr=read(native/'result.json')
    bind(native/'trajectory.npz',SOURCE_SHA)
    if (nm.get('status')!='COMPLETED' or nm.get('controls')!=542 or nm.get('takeover_tick')!=16
            or nm.get('roles')!=ROLES or nm.get('trajectory_sha256')!=SOURCE_SHA
            or nm.get('expert_PD_training_allowed') is not True
            or nm.get('mixed_policy_future_hand_training_allowed') is not False
            or audit.get('pass') is not True or audit.get('checks_max')!=0
            or audit.get('check_count')!=9768 or audit.get('native_data_sha256')!=SOURCE_SHA
            or audit.get('controls')!=542 or audit.get('measured_states')!=543
            or ('takeover_tick' in audit and audit['takeover_tick']!=16)
            or nr.get('status')!='COMPLETED'):
        raise ValueError('bound old542/t16/root9768 source required')
    keys=('q','dq','hand','obj','velocity','action','applied','pd_targets','expert_action','expert_PD','expert_applied')
    with np.load(native/'trajectory.npz',allow_pickle=False) as stream:
        live={k:stream[k].copy() for k in keys}
    if (any(not np.isfinite(v).all() for v in live.values())
            or live['q'].shape!=(543,4,18) or live['applied'].shape!=(542,4,18)
            or live['expert_applied'].shape!=(542,2) or live['expert_applied'].dtype!=bool
            or not np.array_equal(live['action'],live['applied'])
            or not np.array_equal(live['expert_applied'],np.c_[np.zeros(542,bool),np.arange(542)>=16])
            or not np.array_equal(live['applied'][16:,3],live['expert_action'][16:,1])
            or not np.array_equal(live['pd_targets'][16:,3],live['expert_PD'][16:,1])):
        raise ValueError('actually executed own-row expert action/PD chain differs')
    offset,scale=[torch.as_tensor(nm['native_controller'][k]) for k in ('offset','scale')]
    reconstructed=pd_from_action(torch.from_numpy(live['expert_action'][16:,1]),
        torch.from_numpy(live['q'][16:542,3]),offset,scale).numpy()
    PD_error=float(np.abs(reconstructed-live['expert_PD'][16:,1]).max())
    if PD_error>1e-6:
        raise ValueError('expert current-q PD reconstruction differs')
    source=OWNER/'competent-pd-teacher-20261011-r2'
    sm=read(source/'manifest.json'); sr=read(source/'result.json'); sa=read(source/'audit.json')
    if (sm.get('status')!='COMPLETED' or not sm.get('training_allowed') or not sr.get('teacher_readiness')
            or not sa.get('pass') or sm['native_controller']!=nm['native_controller']):
        raise ValueError('qualified compatible nominal teacher required')
    for name in ('trajectory.npz','manifest.json','result.json','audit.json'):
        bind(source/name,base['input_sha256'][str(source/name)])
    ticks=np.arange(16,96)
    arrays={}; raw={}
    # Compare teacher and recovery rows from the SAME native PhysX trace.
    # Original nominal source above authenticates the base, not this comparator.
    for name,data,role in [('recovery_t16',live,3),('nominal_role0',live,0)]:
        if name=='recovery_t16' and any(not live['expert_applied'][t:t+24,1].all() for t in ticks):
            raise ValueError('every future causal control must actually be expert')
        future=data['hand'][ticks[:,None]+np.arange(1,25),role]
        state,tau=features(*(data[k][ticks,role] for k in ('q','dq','hand','obj','velocity')),future)
        label=encode_target(data['pd_targets'][ticks,role],data['q'][ticks,role])
        if any(not np.isfinite(v).all() for v in (state,tau,label)):
            raise ValueError('nonfinite diagnostic arrays')
        arrays[name]=(state,tau,label)
        raw.update({name+'_state87':state,name+'_tau24x33':tau,name+'_encoded_true_PD':label,
                    name+'_row_ids':np.c_[ticks,np.full(len(ticks),role)]})
    for path in (Path(__file__),TASK/'src/trajectory_policy/pd_retargeter.py',
                 TASK/'src/trajectory_policy/retargeter_data.py',TASK/'src/trajectory_policy/wam_retargeter_admission.py'):
        bind(path)
    def verify_inputs():
        if any(sha(path)!=digest for path,digest in evidence.items()):
            raise ValueError('actual diagnostic input drift')
    def guard():
        if time.monotonic()-started>=60:
            raise TimeoutError('60s diagnostic deadline')
        pids=set(map(int,subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True).split()))
        if pids-{os.getpid()}:
            raise RuntimeError('foreign GPU2 process')
        paths=[p for p in OWNER.glob('wam-ref2-*') if p.is_dir()]
        paths += [p for p in (ROOT/'tmp').rglob('wam-*') if p.is_dir()
                  and not any(q.name.startswith('wam-') for q in p.parents if q!=ROOT/'tmp')]
        total=sum(int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]) for p in paths)
        if total>=100_000_000:
            raise RuntimeError('all ref2 artifacts/scratch100MB limit')
        return total
    verify_inputs(); guard()
    old_handler=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame):
        raise TimeoutError('60s diagnostic deadline')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(max(1,int(60-(time.monotonic()-started))))
    out.mkdir()
    manifest=dict(schema='ref2dex.wam-R-cross-takeover.v1',status='RUNNING',task='goal-point-wam',
        experiment_id='P-20261012-goal-point-wam-ref2-recovery-retargeter',run_id=out.name,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        physical_gpu=2,input_sha256=evidence,source_trajectory_sha256=SOURCE_SHA,
        source_takeover_tick=16,ticks=[16,96],horizon=24,base_R_sha256=BASE_SHA,
        comparator='teacher role0 from same native trace and identical ticks',
        training_allowed=False,actor_future_hand_training_allowed=False,physics_override_allowed=False,
        purpose='test t32 recovery R coverage transfer to an existing t16 expert trace',
        limitation='same motion/reset; overlapping windows, cross-takeover diagnostic only')
    write(out/'manifest.json',manifest)
    try:
        reports={}
        offset,scale=offset.cuda(),scale.cuda()
        packets=dict(base=base,**candidates)
        for model_name,packet in packets.items():
            model=load_model(packet,'cuda').eval()
            for p in model.parameters():
                p.requires_grad_(False)
            reports[model_name]={}
            for split,(state,tau,label) in arrays.items():
                st,tt,lt=[torch.from_numpy(v).cuda() for v in (state,tau,label)]
                with torch.no_grad():
                    pd=decode_target(model(st,tt),st[:,:18]); truth=decode_target(lt,st[:,:18])
                    _,action,clipped=command(pd,st[:,:18],offset,scale)
                    delta=(pd-truth).cpu().numpy(); clip=clipped.cpu().numpy()
                def stats(mask):
                    part=delta[mask]; xyz=np.linalg.norm(part[:,:3],axis=-1)*1000
                    return dict(count=len(part),xyz_p50_mm=float(np.percentile(xyz,50)),
                        xyz_p95_mm=float(np.percentile(xyz,95)),
                        rotation_p95_rad=float(np.percentile(np.abs(part[:,3:6]).max(-1),95)),
                        finger_p95_rad=float(np.percentile(np.abs(part[:,[6,8,10,12,14,15]]).max(-1),95)),
                        clipping_rate=float(clip[mask].mean()))
                reports[model_name][split]=dict(all=stats(np.ones(len(ticks),bool)),
                    by_block={f'{lo}:{hi}':stats((ticks>=lo)&(ticks<hi)) for lo,hi in BLOCKS})
                raw.update({model_name+'_'+split+'_PD':pd.cpu().numpy(),
                    model_name+'_'+split+'_true_PD':truth.cpu().numpy(),
                    model_name+'_'+split+'_action':action.cpu().numpy(),model_name+'_'+split+'_clipped':clip})
            del model
            guard()
        verify_inputs()
        result=dict(status='COMPLETED',reports=reports,model_identities=dict(base=dict(checkpoint_sha256=BASE_SHA),**identities),
            current_q_PD_reconstruction_max=PD_error,source_expert_windows=80,
            comparator_same_native_trace=True,comparator_role=0,
            source_future_controls_all_expert=True,training_allowed=False,
            actor_training_allowed=False,physics_override_allowed=False,
            conclusion='UNCLEAR',scope='forward-only transfer diagnostic; failed offline gates remain failed',
            elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        np.savez_compressed(out/'predictions.npz',**raw)
        write(out/'result.json',result)
        manifest.update(status='COMPLETED',result_sha256=sha(out/'result.json'),predictions_sha256=sha(out/'predictions.npz'))
        write(out/'manifest.json',manifest)
        if int(subprocess.check_output(['du','-sb',str(out)],text=True).split()[0])>=1_000_000:
            raise RuntimeError('new diagnostic output exceeds1MB')
        guard()
        print(json.dumps(result),flush=True)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error)); write(out/'manifest.json',manifest)
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)


if __name__=='__main__':
    main()
