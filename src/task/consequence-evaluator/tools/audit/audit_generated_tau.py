"""Frozen generated-tau score transfer and bounded native-geometry audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]; ROOT = TASK.parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK/'tools/run')]
from consequence_evaluator.contracts import is_within
from consequence_evaluator.data import sha
from consequence_evaluator.old_utility import panel_metrics
from consequence_evaluator.proposal_history import condition
from consequence_evaluator.proposal_runtime import TauProposal, retrieve_rows, choose_generated
from consequence_evaluator.tau_projection import project_tau
from consequence_evaluator.tau_tracking import FINGERS, FINGER_LIMITS
from consequence_evaluator.trajectory_utility import TrajectoryUtility
from probe_measured_history_tau import collect
from probe_reference_tracking import gpu_state

NAMES = ['persistence', 'displacement']+['retrieval%d'%i for i in range(8)]+['observed_comparison']


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--evaluator', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--seconds', type=int, default=840)
    a = p.parse_args(); begin = time.monotonic(); output = a.output.resolve()
    if output.exists() or not is_within(output, ROOT/'outputs/consequence-evaluator') or not 60 <= a.seconds <= 840:
        raise ValueError('fresh bounded task-owned audit required')
    before = gpu_state(a.gpu)
    if before['used_mib'] > 512 or before['utilization'] > 10 or before['total_mib']-before['used_mib'] < 20480:
        raise RuntimeError('GPU not idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    pm = json.loads((a.proposal/'manifest.json').read_text()); em = json.loads((a.evaluator/'manifest.json').read_text())
    panel_m = json.loads((a.panel/'manifest.json').read_text())
    if any(m.get('status') != 'COMPLETED' for m in (pm, em, panel_m)):
        raise ValueError('completed frozen artifacts required')
    hashes = {}
    for folder, files in [(a.proposal, ['manifest.json','displacement-best.pt','test-candidates.npz']),
                          (a.evaluator, ['manifest.json','T.pt','panel-predictions.npz']),
                          (a.panel, ['manifest.json','windows.npz'])]:
        for f in files:
            path=(folder/f).resolve(); hashes[str(path)] = sha(path)
    if sha(a.proposal/'displacement-best.pt') != pm['checkpoint_sha256']['displacement']:
        raise ValueError('proposal hash mismatch')
    if sha(a.evaluator/'T.pt') != em['checkpoint_sha256']['T']:
        raise ValueError('evaluator hash mismatch')
    if em['input_sha256'].get(str((a.panel/'windows.npz').resolve())) != sha(a.panel/'windows.npz'):
        raise ValueError('evaluator panel identity mismatch')
    source = {}
    for split in ('train', 'test'):
        paths=[Path(x).parent for x in pm['input_sha256'] if x.endswith('/manifest.json') and ('bank-'+split+'-') in x]
        if len(paths) != 1:
            raise ValueError('unique original source required')
        source[split]=paths[0]
        for f in ('manifest.json','trajectory.npz'):
            path=paths[0]/f; digest=sha(path)
            if pm['input_sha256'].get(str(path)) != digest:
                raise ValueError('source identity mismatch')
            hashes[str(path)]=digest
    bank, _, _ = collect(source['train'])
    with np.load(source['test']/'trajectory.npz', allow_pickle=False) as s:
        live={k:s[k] for k in ('hand_keypoints','object_pose','dof_position','dof_velocity','object_velocity')}
    with np.load(a.panel/'windows.npz', allow_pickle=False) as s:
        data={k:s[k] for k in ('panel','candidate','split','tick','source_env','episode','label',
                              'pw_hand_future','pw_hand_history')}
    with np.load(a.evaluator/'panel-predictions.npz', allow_pickle=False) as s:
        original={k:s[k] for k in ('panel_rows','T','donor')}
    rows=original['panel_rows']; flat=rows.reshape(-1)
    if rows.shape != (18,7) or not np.all(data['split'][flat]=='test') or not np.all(data['candidate'][rows]==np.arange(7)):
        raise ValueError('frozen eighteen seven-candidate test panels required')
    if len({tuple(row) for row in rows}) != 18:
        raise ValueError('duplicate panel identity')
    features=[]; current=[]; poses=[]; q=[]; world_current=[]
    for row in flat:
        tick=int(data['tick'][row]); env=int(data['source_env'][row]); sl=slice(tick-3,tick+1)
        h, hand=condition(live['object_pose'][sl,env],live['hand_keypoints'][sl,env],
            live['dof_position'][sl,env],live['dof_velocity'][sl,env],live['object_velocity'][sl,env])
        if np.max(np.abs(hand-data['pw_hand_history'][row,-1])) > 3e-6:
            raise ValueError('current hand/frame mismatch')
        pose=live['object_pose'][tick,env]
        comparison=np.einsum('ji,tpj->tpi',pose[:3,:3],
            live['hand_keypoints'][tick+1:tick+25,env]-pose[None,None,:3,3])
        if comparison.shape != (24,11,3) or np.max(np.abs(comparison-data['pw_hand_future'][row])) > 3e-6:
            raise ValueError('recorded future comparator/frame mismatch')
        features.append(h);current.append(hand);poses.append(live['object_pose'][tick,env]);
        q.append(live['dof_position'][tick,env]);world_current.append(live['hand_keypoints'][tick,env])
    h=np.stack(features); current=np.stack(current); poses=np.stack(poses); q=np.stack(q);world_current=np.stack(world_current)
    proposal=TauProposal(torch.load(a.proposal/'displacement-best.pt',map_location='cpu',weights_only=False)).to(device).eval()
    ep=torch.tensor(np.unique(bank['episode'],return_inverse=True)[1],device=device)
    bh=proposal.encode(torch.tensor(bank['history'],device=device));bd=torch.tensor(bank['target']-bank['current'][:,None],device=device)
    packet=torch.load(a.evaluator/'T.pt',map_location=device,weights_only=False)
    if packet.get('arm') != 'T':
        raise ValueError('tau-only evaluator required')
    model=TrajectoryUtility(packet['architecture']['history_dim'],packet['architecture']['width'],packet['architecture']['layers']).to(device).eval()
    model.load_state_dict(packet['model']); mean, scale=packet['statistics']['trajectory']

    def score(tau):
        x=(tau.reshape(-1,24,33)-mean)/scale
        values=[]
        with torch.inference_mode():
            for start in range(0,len(x),128):
                value=torch.tensor(x[start:start+128],device=device)
                values.append(model(torch.zeros(len(value),1442,device=device),value,
                                    torch.zeros(len(value),24,12,device=device),False).cpu().numpy())
        return np.concatenate(values)

    observed=data['pw_hand_future'][flat]
    observed_score=score(observed).reshape(18,7)
    replay=float(np.max(np.abs(observed_score-original['T'])))
    if replay > 1e-5:
        raise ValueError('frozen observed score replay differs')
    with torch.inference_mode():
        prediction=proposal(torch.tensor(h,device=device),torch.tensor(current,device=device)).cpu().numpy()
        anchor=np.arange(0,len(flat),7)
        ids, distances=retrieve_rows(proposal.encode(torch.tensor(h[anchor],device=device)),bh,ep,8)
        retrieved=bd[ids].cpu().numpy()+current[anchor,None,None]
    # Verify runtime preprocessing/decoding against prior saved predictions.
    with np.load(a.proposal/'test-candidates.npz',allow_pickle=False) as s:
        lookup={(str(e),int(t)):i for i,(e,t) in enumerate(zip(s['episode'],s['tick']))}
        old_ids=[lookup[(str(data['episode'][i]),int(data['tick'][i]))] for i in flat]
        prediction_replay=float(np.max(np.abs(prediction-s['displacement'][old_ids])))
    if prediction_replay > 1e-5:
        raise ValueError('proposal inference replay differs')
    raw=np.concatenate((np.repeat(current[anchor,None,None],24,axis=2),prediction[anchor,None],retrieved,observed[anchor,None]),1)
    labels=data['label'][flat].reshape(18,7)
    donor_map={int(v):i for i,v in enumerate(flat)}
    donor=np.asarray([donor_map[int(v)] for v in original['donor']])
    forecast_score=score(prediction).reshape(18,7)
    observed_metrics=panel_metrics(labels,observed_score)
    forecast_metrics=panel_metrics(labels,forecast_score)
    forecast_shuffle=panel_metrics(labels,score(prediction[donor]).reshape(18,7))
    raw_score=score(raw).reshape(18,11);raw_choices=choose_generated(raw_score[:,:10])
    output.mkdir(parents=True)
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    for path in (urdf,Path(__file__).resolve(),TASK/'src/consequence_evaluator/proposal_runtime.py',
                 TASK/'src/consequence_evaluator/proposal_history.py',TASK/'src/consequence_evaluator/tau_projection.py',
                 TASK/'src/consequence_evaluator/reset_kinematics.py',TASK/'src/consequence_evaluator/fixed_wrist_decoder.py',
                 TASK/'src/consequence_evaluator/trajectory_utility.py',TASK/'tools/run/probe_measured_history_tau.py',
                 TASK/'tools/run/train_history_to_tau.py',TASK/'src/consequence_evaluator/reference_tracking.py',
                 TASK/'src/consequence_evaluator/object_relative_servo.py',TASK/'src/consequence_evaluator/tau_tracking.py'):
        hashes[str(path)]=sha(path)
    manifest=dict(schema='ref2dex.generated-tau-audit.v1',status='RUNNING',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashes,physical_gpu=a.gpu,gpu_before=before,seconds_cap=a.seconds,
        input_contract='Measured t-3:t only; query future never enters generation/retrieval/choice',
        selector_contract='Ten generated candidates; observed future is separate comparator',
        names=NAMES,anchors=data['source_env'][flat[anchor]].tolist(),ticks=data['tick'][flat[anchor]].tolist(),
        projection_iterations=300,observed_score_replay_error=replay,proposal_replay_error=prediction_replay,
        diagnostic_label_contract='Recorded-policy forecast substitution only; no execution Y for generated alternatives')
    write(output/'manifest.json',manifest)
    print(json.dumps(dict(stage='frozen-score',elapsed_s=time.monotonic()-begin,observed=observed_metrics,
                         forecast=forecast_metrics,forecast_shuffle=forecast_shuffle,gpu=gpu_state(a.gpu))),flush=True)

    def monitor(iteration,elapsed):
        gpu=gpu_state(a.gpu)
        if time.monotonic()-begin > a.seconds:
            raise TimeoutError('audit deadline')
        prior={r.split(',')[0].strip() for r in before['processes'].splitlines()}
        for row in gpu['processes'].splitlines():
            pid,memory=row.split(',')
            if pid.strip() not in prior|{str(os.getpid())} and int(memory)>512:
                raise RuntimeError('foreign GPU work; stop own audit')
        print(json.dumps(dict(stage='projection',iteration=iteration,elapsed_s=elapsed,
            eta_s=elapsed/iteration*(300-iteration),gpu=gpu,torch_peak_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)

    try:
        pool_world=np.einsum('nij,nctpj->nctpi',poses[anchor,:3,:3],raw)+poses[anchor,None,None,None,:3,3]
        forecast_world=np.einsum('nij,ntpj->ntpi',poses[:,:3,:3],prediction)+poses[:,None,None,:3,3]
        raw_world=np.concatenate((pool_world.reshape(-1,24,11,3),forecast_world))
        init_hand=np.concatenate((np.repeat(world_current[anchor],11,axis=0),world_current))
        init_q=np.concatenate((np.repeat(q[anchor],11,axis=0),q))
        proj=project_tau(init_hand,raw_world,init_q,urdf,device,iterations=300,
            deadline_s=min(720,a.seconds-(time.monotonic()-begin)-10),monitor=monitor)
        if not all(np.isfinite(proj[k]).all() for k in ('q','points')):
            raise ValueError('nonfinite native projection')
        fingers=proj['q'][:,1:,list(FINGERS)]
        if np.any(fingers < -1e-6) or np.any(fingers > np.asarray(FINGER_LIMITS)+1e-6):
            raise ValueError('projected independent finger limits violated')
        coupling_error=max(float(np.max(np.abs(proj['q'][:,1:,d]-proj['q'][:,1:,p]*ratio)))
            for d,p,ratio in ((7,6,1.05),(9,8,1.05),(11,10,1.05),(13,12,1.05),(16,15,.6),(17,15,.8)))
        if coupling_error > 1e-6:
            raise ValueError('projected finger coupling violated')
        fitted=proj['points'][:,1:]
        delta=fitted-raw_world
        coordinate_mm=np.sqrt(np.mean(delta**2,axis=(1,2,3)))*1000
        roots=[0,1,3,5,7,9]
        palm_mm=np.sqrt(np.mean(np.sum(delta[:,:,roots]**2,-1),axis=(1,2)))*1000
        repeated_poses=np.concatenate((np.repeat(poses[anchor],11,axis=0),poses))
        local=np.einsum('nji,ntpj->ntpi',repeated_poses[:,:3,:3],fitted-repeated_poses[:,None,None,:3,3])
        projected_pool=local[:198].reshape(18,11,24,11,3);projected_forecast=local[198:]
        projected_score=score(projected_pool).reshape(18,11);projected_choices=choose_generated(projected_score[:,:10])
        projected_proxy=panel_metrics(labels,score(projected_forecast).reshape(18,7))
        geometry=(coordinate_mm[:198].reshape(18,11)<=5)&(palm_mm[:198].reshape(18,11)<=3)
        observed_pass=int(geometry[:,10].sum());chosen_pass=int(geometry[np.arange(18),raw_choices].sum())
        forecast_drop=forecast_metrics['pairwise_accuracy']-forecast_shuffle['pairwise_accuracy']
        joint=bool(observed_pass>=15 and chosen_pass>=17 and forecast_metrics['pairwise_accuracy']>=.70 and forecast_drop>=.03)
        geometry_summary={name:dict(coordinate_rmse_median_mm=float(np.median(coordinate_mm[:198].reshape(18,11)[:,i])),
            coordinate_rmse_max_mm=float(coordinate_mm[:198].reshape(18,11)[:,i].max()),
            palm_point_rmse_median_mm=float(np.median(palm_mm[:198].reshape(18,11)[:,i])),pass_count=int(geometry[:,i].sum())) for i,name in enumerate(NAMES)}
        result=dict(status='PROMISING' if joint else ('UNCLEAR' if observed_pass<15 else 'UNPROMISING'),
            joint_screen=joint,observed_usable_count=observed_pass,raw_score_selected_geometry_pass=chosen_pass,
            observed_metrics=observed_metrics,forecast_substitution=forecast_metrics,forecast_shuffle=forecast_shuffle,
            forecast_shuffle_drop=float(forecast_drop),projected_forecast_diagnostic=projected_proxy,
            geometry=geometry_summary,raw_choices=raw_choices.tolist(),projected_choices=projected_choices.tolist(),
            observed_score_replay_error=replay,proposal_replay_error=prediction_replay,
            future_q_finite=True,independent_finger_limits_pass=True,coupling_max_error=coupling_error,
            elapsed_s=time.monotonic()-begin,projection_s=proj['elapsed_s'],
            claim='Frozen score/geometry Probe; no generated-candidate execution labels, native success or Cm benefit')
        np.savez_compressed(output/'audit.npz',raw=raw,projected=projected_pool,raw_score=raw_score,
            projected_score=projected_score,raw_choices=raw_choices,projected_choices=projected_choices,
            geometry_pass=geometry,coordinate_rmse_mm=coordinate_mm,palm_point_rmse_mm=palm_mm,
            forecast=prediction,projected_forecast=projected_forecast,forecast_score=forecast_score,
            observed_score=observed_score,label=labels,source_rows=ids.cpu().numpy(),distances=distances.cpu().numpy(),
            q=proj['q'],anchor_hand=current[anchor],anchor_rows=flat[anchor],projection_start=proj['chosen_start'])
        for path,digest in hashes.items():
            if sha(path)!=digest:
                raise ValueError('input drift: '+path)
        write(output/'result.json',result)
        manifest.update(status='COMPLETED',elapsed_s=result['elapsed_s'],projection_s=proj['elapsed_s'])
        print(json.dumps(result),flush=True)
    except Exception as error:
        manifest.update(status='FAILED',error=repr(error),elapsed_s=time.monotonic()-begin)
        raise
    finally:
        write(output/'manifest.json',manifest)


if __name__ == '__main__':
    main()
