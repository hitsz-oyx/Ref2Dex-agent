"""Bounded R-only adaptation to audited, unmixed expert recovery futures.

The source actor-future prohibition remains in force. Scoped admission permits
only role3's actually executed expert controls/futures as retargeter examples.
Held windows overlap nearby training trajectories: they are local diagnostics.
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
sys.path[:0] = [str(TASK / 'src')]
BASE_SHA = '414411a0c16d92a9e43c3b7a40615f8995ec4ea3b66bcaca3dc1a0fcae10181f'
ROLES = ['teacher', 'gt_tau_R', 'wam_R', 'wam_command_replay_then_expert_tick32']
PANEL = [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 448, 518]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def recovery_split():
    ticks = np.arange(32, 169)
    held = ((ticks >= 40) & (ticks < 48)) | ((ticks >= 96) & (ticks < 112))
    return ticks[~held], ticks[held]


def admit(manifest, audit, result, data, trajectory_sha):
    """CPU checks on the exact source; no privilege granted to actor labels."""
    if (manifest.get('status') != 'COMPLETED' or manifest.get('controls') != 192
            or manifest.get('takeover_tick') != 32 or manifest.get('roles') != ROLES
            or manifest.get('partial_episode') is not True
            or manifest.get('expert_PD_training_allowed') is not True
            or manifest.get('actor_future_hand_training_allowed') is not False
            or manifest.get('mixed_policy_future_hand_training_allowed') is not False
            or manifest.get('trajectory_sha256') != trajectory_sha):
        raise ValueError('exact 192/t32 source and original actor-future prohibition required')
    if (not audit.get('pass') or audit.get('check_count') != 3468
            or audit.get('checks_max') != 0 or not audit.get('all_arrays_finite')
            or audit.get('native_data_sha256') != trajectory_sha
            or audit.get('controls') != 192 or audit.get('measured_states') != 193
            or audit.get('takeover_tick') != 32 or not audit.get('partial_episode')):
        raise ValueError('bound root recovery audit3468 required')
    outcomes = result.get('outcomes', [])
    if (result.get('status') != 'COMPLETED' or not result.get('partial_episode')
            or result.get('local_grasp_calibration') is not True
            or len(outcomes) != 4 or [v.get('role') for v in outcomes] != ROLES
            or any(v['maximum_held_frames'] < 45 for v in (outcomes[0], outcomes[1], outcomes[3]))
            or outcomes[3]['first_stable_tick'] is None
            or outcomes[3]['first_stable_tick'] > 160 or outcomes[3]['clipping_rate'] >= .01):
        raise ValueError('local teacher/GT/expert recovery support required')
    shapes = dict(q=(193, 4, 18), dq=(193, 4, 18), hand=(193, 4, 11, 3),
        obj=(193, 4, 4, 4), velocity=(193, 4, 6), action=(192, 4, 18),
        applied=(192, 4, 18), pd_targets=(192, 4, 18), expert_action=(192, 2, 18),
        expert_PD=(192, 2, 18), expert_applied=(192, 2), expert_clipped=(192, 2))
    if any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in shapes.items()):
        raise ValueError('source shapes/nonfinite values')
    expected = np.zeros((192, 2), bool)
    expected[32:, 1] = True
    if (data['expert_applied'].dtype != bool or data['expert_clipped'].dtype != bool
            or not np.array_equal(data['expert_applied'], expected)
            or data['expert_clipped'][32:, 1].any()
            or not np.array_equal(data['action'], data['applied'])
            or not np.array_equal(data['applied'][32:, 3], data['expert_action'][32:, 1])
            or not np.array_equal(data['pd_targets'][32:, 3], data['expert_PD'][32:, 1])):
        raise ValueError('recovery controls must be actually applied unclipped expert actions/PD')
    train, held = recovery_split()
    for tick in np.r_[train, held]:
        if not data['expert_applied'][tick:tick+24, 1].all():
            raise ValueError('mixed/forked future cannot enter R')
    return dict(schema='ref2dex.unmixed-recovery-R-admission.v1', admitted=True,
        source_trajectory_sha256=trajectory_sha, root_audit_checks=3468,
        role=3, first_tick=32, last_tick=168, horizon=24,
        windows=137, train_ticks=train.tolist(), held_ticks=held.tolist(),
        actor_future_hand_training_allowed=False, mixed_policy_future_hand_training_allowed=False,
        R_training_allowed=True, scope='R only: measured own role3 futures caused entirely by actual expert controls',
        limitation='same episode, overlapping futures; local coverage diagnostic, no independent generalization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--engineering-smoke', action='store_true')
    args = parser.parse_args()
    out, native = args.output.resolve(), args.native.resolve()
    owner = ROOT / 'outputs/trajectory-policy'
    if out.exists() or out.parent != owner or not out.name.startswith('wam-ref2-'):
        raise ValueError('fresh outputs/trajectory-policy/wam-ref2-* output required')
    expected_native = owner / 'wam-ref2-execution-aggcollect-20261012-r1/native'
    if native != expected_native:
        raise ValueError('this bounded probe admits only the pinned aggregation source')
    if args.gpu != 2:
        raise ValueError('this bounded probe is assigned GPU2')
    started = time.monotonic()
    scratch = ROOT / 'tmp/trajectory-policy/wam-recovery-retargeter'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), TMPDIR=str(scratch))
    import torch
    import torch.nn.functional as F
    from trajectory_policy.pd_retargeter import SCHEMA, load_model, windows, decode_target, command, pd_from_action
    torch.set_num_threads(2)
    torch.manual_seed(290)
    rng = np.random.default_rng(291)
    qm, qa, qr = [json.loads((native / k).read_text()) for k in
                   ('manifest.json', 'root_recovery_audit.json', 'result.json')]
    keys = ('q', 'dq', 'hand', 'obj', 'velocity', 'action', 'applied', 'pd_targets',
            'expert_action', 'expert_PD', 'expert_applied', 'expert_clipped')
    with np.load(native / 'trajectory.npz', allow_pickle=False) as stream:
        live = {k:stream[k].copy() for k in keys}
    admission = admit(qm, qa, qr, live, sha(native / 'trajectory.npz'))
    offset, scale = [torch.as_tensor(qm['native_controller'][k]) for k in ('offset', 'scale')]
    reconstructed = pd_from_action(torch.from_numpy(live['expert_action'][32:, 1]),
        torch.from_numpy(live['q'][32:192, 3]), offset, scale).numpy()
    error = float(np.abs(reconstructed-live['expert_PD'][32:, 1]).max())
    if error > 1e-6:
        raise ValueError('expert PD does not bind applied action to current measured q')
    admission['current_q_PD_reconstruction_max'] = error
    source = owner / 'competent-pd-teacher-20261011-r2'
    sm, sr, sa = [json.loads((source / k).read_text()) for k in ('manifest.json', 'result.json', 'audit.json')]
    if (sm.get('status') != 'COMPLETED' or not sm.get('training_allowed')
            or not sr.get('teacher_readiness') or not sa.get('pass')
            or sha(source / 'trajectory.npz') != sm['trajectory_sha256']):
        raise ValueError('qualified nominal teacher required')
    with np.load(source / 'trajectory.npz', allow_pickle=False) as stream:
        nominal = {k:stream[k][:, :28].copy() for k in ('q', 'dq', 'hand', 'obj', 'velocity', 'pd_targets')}
    base = owner / 'competent-pd-fit-20261011-r1/tau.pt'
    if sha(base) != BASE_SHA:
        raise ValueError('pinned competent R required')
    packet = torch.load(base,map_location='cpu',weights_only=False)
    # The pinned base packet binds its original audit and nominal data together.
    for name in ('trajectory.npz','manifest.json','result.json','audit.json'):
        path=source/name
        if packet['input_sha256'].get(str(path))!=sha(path):
            raise ValueError('original nominal data/audit differs from authenticated base: '+name)
    if sm['native_controller']!=qm['native_controller']:
        raise ValueError('nominal/recovery native controller differs')
    evidence = {}
    for path in [Path(__file__), TASK/'src/trajectory_policy/pd_retargeter.py',
                 TASK/'src/trajectory_policy/retargeter_data.py', base] + [native/k for k in
                 ('trajectory.npz', 'manifest.json', 'root_recovery_audit.json', 'result.json')] + [source/k for k in
                 ('trajectory.npz', 'manifest.json', 'result.json', 'audit.json')]:
        digest = sha(path)
        if str(path) in evidence and evidence[str(path)] != digest:
            raise ValueError('source dependency drift: '+str(path))
        evidence[str(path)] = digest
    def verify_inputs():
        if any(sha(path) != digest for path, digest in evidence.items()):
            raise ValueError('input identity drift')
    verify_inputs()
    admission.update(native_source_path=str(native/'trajectory.npz'),
        native_source_sha256=sha(native/'trajectory.npz'),
        native_manifest_path=str(native/'manifest.json'),native_manifest_sha256=sha(native/'manifest.json'),
        root_audit_path=str(native/'root_recovery_audit.json'),root_audit_sha256=sha(native/'root_recovery_audit.json'),
        native_result_path=str(native/'result.json'),native_result_sha256=sha(native/'result.json'),
        nominal_audit_path=str(source/'audit.json'),nominal_audit_sha256=sha(source/'audit.json'),
        nominal_trajectory_path=str(source/'trajectory.npz'),nominal_trajectory_sha256=sha(source/'trajectory.npz'),
        nominal_audit_bound_by_base_R_sha256=BASE_SHA)
    train_ticks, held_ticks = recovery_split()
    all_live = windows(live, [3])
    nominal_all = windows(nominal, list(range(16)))
    if len(nominal_all[0]) != 519*16 or not np.isfinite(nominal_all[0]).all():
        raise ValueError('exact finite nominal519x16 windows required')
    report_all = windows(nominal, list(range(24, 28)))
    arrays = dict(nominal_train=nominal_all[:3],
        nominal_report=tuple(v[np.isin(report_all[3], PANEL)] for v in report_all[:3]),
        nominal_startup=tuple(v[report_all[3] == 0] for v in report_all[:3]),
        recovery_train=tuple(v[np.isin(all_live[3], train_ticks)] for v in all_live[:3]),
        recovery_held=tuple(v[np.isin(all_live[3], held_ticks)] for v in all_live[:3]),
        recovery_earlyheld=tuple(v[(all_live[3] >= 40) & (all_live[3] < 48)] for v in all_live[:3]))
    if any(not all(np.isfinite(v).all() for v in values) for values in arrays.values()):
        raise ValueError('nonfinite fit arrays')
    rows = dict(nominal_train=np.asarray([(t,e) for t in range(519) for e in range(16)]),
        nominal_report=np.asarray([(t,e) for t in PANEL for e in range(24,28)]),
        nominal_startup=np.asarray([(0,e) for e in range(24,28)]),
        recovery_train=np.c_[train_ticks,np.full(len(train_ticks),3)],
        recovery_held=np.c_[held_ticks,np.full(len(held_ticks),3)],
        recovery_earlyheld=np.c_[np.arange(40,48),np.full(8,3)])
    if any(len(arrays[k][0]) != len(v) for k,v in rows.items()):
        raise ValueError('sample rows do not bind arrays')
    def guard(phase, step, total):
        elapsed = time.monotonic()-started
        if elapsed >= 600:
            raise TimeoutError('600s budget exhausted')
        pids = set(map(int, subprocess.check_output(['nvidia-smi','-i','2',
            '--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).split()))
        if pids - {os.getpid()}:
            raise RuntimeError('foreign GPU2 process')
        paths = [p for p in owner.glob('wam-ref2-*') if p.is_dir()]
        paths += [p for p in (ROOT/'tmp').rglob('wam-*') if p.is_dir() and not any(q.name.startswith('wam-') for q in p.parents if q != ROOT/'tmp')]
        size = sum(int(subprocess.check_output(['du','-sb',str(p)], text=True).split()[0]) for p in paths)
        if size >= 100_000_000:
            raise RuntimeError('all ref2 outputs/scratch exceed100MB')
        print(json.dumps(dict(phase=phase,step=step,total=total,elapsed_s=elapsed,
            eta_s=elapsed/max(step,1)*(total-step),artifact_bytes=size)), flush=True)
    guard('admitted',0,1000)
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout(signum, frame):
        raise TimeoutError('600s wall budget exhausted')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(max(1,int(600-(time.monotonic()-started))))
    out.mkdir()
    write(out/'scoped_admission.json', admission)
    manifest = dict(schema='ref2dex.wam-recovery-R-fit.v1', status='RUNNING',
        task='goal-point-wam', experiment_id='P-20261012-goal-point-wam-ref2-recovery-retargeter',
        run_id=out.name,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        seed=290,draw_seed=291,physical_gpu=2,input_sha256=evidence,base_R_sha256=BASE_SHA,
        scoped_admission_sha256=sha(out/'scoped_admission.json'),
        admission_path=str(out/'scoped_admission.json'),admission_sha256=sha(out/'scoped_admission.json'),
        checkpoint_sha256=None,historical_producer_input_sha256=dict(native=qm['input_sha256'],
            native_server=qm['server_identity']['input_sha256'],nominal=sm['input_sha256']),
        updates=1 if args.engineering_smoke else 1000,learning_rate=1e-4,weight_decay=.01,
        batch='nominal64 (startup16/later48) + recovered64; replacement',
        normalization='original buffers unchanged; equal normalized Huber PD losses',
        fixed_final_only=True,training_allowed=False,actor_future_hand_training_allowed=False,
        split='113 recovery train /24 held; overlapping same-episode diagnostics, not independent generalization')
    write(out/'manifest.json',manifest)
    try:
        model = load_model(packet,'cuda')
        if not model.use_tau or any(not torch.equal(v.cpu(),packet['model'][k]) for k,v in model.state_dict().items()):
            raise ValueError('initial R identity/use_tau differs from base')
        def buffer_hash():
            d=hashlib.sha256()
            for name,value in model.named_buffers():
                d.update(name.encode()); d.update(value.detach().cpu().numpy().tobytes())
            return d.hexdigest()
        frozen_buffer_hash=buffer_hash()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.net.parameters():
            p.requires_grad_(True)
        tensors={k:tuple(torch.from_numpy(v).cuda() for v in values) for k,values in arrays.items()}
        offset,scale=offset.cuda(),scale.cuda()
        raw={}
        def report(split, phase):
            state,tau,target=tensors[split]
            with torch.no_grad():
                encoded=model(state,tau)
                current_q=state[:,:18]
                pd=decode_target(encoded,current_q); truth=decode_target(target,current_q)
                _,action,clipped=command(pd,current_q,offset,scale)
                delta=(pd-truth).cpu().numpy()
            xyz=np.linalg.norm(delta[:,:3],axis=-1)*1000
            stats=dict(count=len(xyz),xyz_p50_mm=float(np.percentile(xyz,50)),xyz_p95_mm=float(np.percentile(xyz,95)),
                rotation_p95_rad=float(np.percentile(np.abs(delta[:,3:6]).max(-1),95)),
                finger_p95_rad=float(np.percentile(np.abs(delta[:,[6,8,10,12,14,15]]).max(-1),95)),
                clipping_rate=float(clipped.float().mean()))
            raw.update({phase+'_'+split+'_pd':pd.cpu().numpy(),phase+'_'+split+'_true_PD':truth.cpu().numpy(),
                phase+'_'+split+'_action':action.cpu().numpy(),phase+'_'+split+'_row_ids':rows[split]})
            return stats
        report_splits=('nominal_report','nominal_startup','recovery_train','recovery_held','recovery_earlyheld')
        initial={k:report(k,'initial') for k in report_splits}
        optimizer=torch.optim.AdamW(model.net.parameters(),lr=1e-4,weight_decay=.01)
        startup=np.flatnonzero(nominal_all[3]<16); later=np.flatnonzero(nominal_all[3]>=16)
        curve=[]; gradient=None
        for step in range(1,manifest['updates']+1):
            ni=np.r_[rng.choice(startup,16),rng.choice(later,48)]
            ri=rng.choice(len(train_ticks),64)
            ns,nt,npd=(v[ni] for v in tensors['nominal_train'])
            rs,rt,rpd=(v[ri] for v in tensors['recovery_train'])
            optimizer.zero_grad(set_to_none=True)
            nl=F.huber_loss(model(ns,nt)/model.target_scale,npd/model.target_scale)
            rl=F.huber_loss(model(rs,rt)/model.target_scale,rpd/model.target_scale)
            loss=nl+rl
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite R loss')
            loss.backward()
            if step==1:
                gradient=float(sum(p.grad.abs().sum() for p in model.net.parameters() if p.grad is not None))
                if not np.isfinite(gradient) or gradient<=0:
                    raise ValueError('R.net gradient missing')
            norm=torch.nn.utils.clip_grad_norm_(model.net.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite R gradient')
            optimizer.step()
            if step==1 or step%25==0:
                curve.append(dict(step=step,nominal_loss=float(nl),recovery_loss=float(rl),gradient_norm=float(norm)))
                guard('train',step,manifest['updates'])
        final={k:report(k,'final') for k in report_splits}
        if buffer_hash()!=frozen_buffer_hash:
            raise ValueError('original normalization buffers changed')
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise FloatingPointError('nonfinite final R')
        verify_inputs()
        gates=dict(recovery_held_XYZ20mm=final['recovery_held']['xyz_p95_mm']<=20,
            nominal_XYZ20mm=final['nominal_report']['xyz_p95_mm']<=20,
            startup_XYZ10mm=final['nominal_startup']['xyz_p95_mm']<=10,
            rotation_finger_retained=all(final[k][m]<=1.5*initial[k][m] for k in
                ('recovery_held','recovery_earlyheld','nominal_report','nominal_startup')
                for m in ('rotation_p95_rad','finger_p95_rad')))
        result=dict(status='COMPLETED',initial=initial,final=final,gates=gates,curve=curve,
            offline_screen_pass=all(gates.values()) and not args.engineering_smoke,
            local_GT_physical_calibration_allowed=all(gates.values()) and not args.engineering_smoke,
            conclusion='engineering PASS' if args.engineering_smoke else ('PROMISING' if all(gates.values()) else 'UNCLEAR'),
            scope='R coverage only; permits local GT physical calibration, not actor/placement success',
            base_R_sha256=BASE_SHA,initial_net_identical_base=True,normalization_buffer_sha256=frozen_buffer_hash,
            final_normalization_buffer_sha256=buffer_hash(),net_gradient_sum=gradient,
            elapsed_s=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            scoped_admission_sha256=manifest['scoped_admission_sha256'],
            admission_path=manifest['admission_path'],admission_sha256=manifest['admission_sha256'],
            checkpoint_sha256=None)
        if not args.engineering_smoke:
            torch.save(dict(schema=SCHEMA,width=model.width,use_tau=model.use_tau,
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                base_R_sha256=BASE_SHA,input_sha256=evidence,scoped_admission_sha256=manifest['scoped_admission_sha256'],
                selected_update=1000),out/'tau.pt')
            if (out/'tau.pt').stat().st_size>=2_000_000:
                raise ValueError('R checkpoint exceeds2MB')
            result['checkpoint_sha256']=sha(out/'tau.pt')
            manifest['checkpoint_sha256']=result['checkpoint_sha256']
        np.savez_compressed(out/'predictions.npz',**raw)
        write(out/'result.json',result)
        guard('completed',1,1)
        manifest.update(status='COMPLETED',result_sha256=sha(out/'result.json'),predictions_sha256=sha(out/'predictions.npz'))
        write(out/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='FAILED',error=repr(error))
        write(out/'manifest.json',manifest)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,previous_handler)


if __name__=='__main__':
    main()
