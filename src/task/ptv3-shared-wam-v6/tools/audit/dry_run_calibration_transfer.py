"""Preflight frozen calibration transfer protocol. No training or physics option.

One real nominal observation forward per checkpoint on CPU checks compatibility;
held-out and perturbed-initial performance remain NOT_EVALUATED.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
RUN=ROOT/'outputs/ptv3-shared-wam-v6/single-tau-calibration-20261022-r1'
ADMISSION=ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true',required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/wam-rl-policy-v3/src')]
    import torch
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from ptv3_shared_wam_v6.tau_contract import gt_reference_tau
    torch.set_num_threads(2)
    calibration=json.loads((RUN/'manifest.json').read_text())
    name=calibration['source_sequence']
    admission=json.loads((ADMISSION/'manifest.json').read_text())
    original=next(r for r in admission['records'] if r['sequence']==name)
    # Predetermine the distinct same-object case lexically, without model scores.
    candidates=sorted((r for r in admission['records'] if
        r['sequence']!=name and Path(r['source']).resolve()!=Path(original['source']).resolve()
        and r['split'] in ('val','dev') and r['engineering']=='PASS' and r['frames']+37<=600),
        key=lambda r:(r['object']!=original['object'],r['sequence']))
    if not candidates:raise ValueError('no bounded distinct DEV trajectory')
    other=candidates[0]
    with np.load(original['source']+'/trajectory.npz') as z:
        original_q=z['q'][0].copy();original_hand=z['hand'][0].copy()
    with np.load(other['source']+'/trajectory.npz') as z:
        if len(z['action'])+1 != len(z['hand']) or not np.isfinite(z['hand']).all():
            raise ValueError('held-out recording contract')
        initial_q_difference=float(np.max(np.abs(z['q'][0]-original_q)))
        initial_hand_difference=float(np.max(np.abs(z['hand'][0]-original_hand)))
        other_states=len(z['hand'])
    assert initial_hand_difference>0
    other_task=json.loads((ADMISSION/other['sidecar']/'raw-task.json').read_text())
    with np.load(RUN/'source-pd/trajectory.npz') as z:
        # Exactly the old 81-state GT definition, not the newly extended channel.
        x=z['observation'][47:48].copy();tau,valid=gt_reference_tau(z['hand'][:,0],47)
        x[:,421:685]=tau.reshape(1,-1);x[:,685]=valid
    checkpoint_paths=[Path(calibration['base_checkpoint']),RUN/'calibrated.pt']
    predictions=[];contracts=[]
    for checkpoint in checkpoint_paths:
        packet=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=MeasuredInspirePolicy(with_tau=True).eval().requires_grad_(False)
        model.load_state_dict(packet['model'],strict=True)
        with torch.inference_mode():prediction=model.actor(torch.as_tensor(x))
        if prediction.shape!=(1,12) or not torch.isfinite(prediction).all():raise ValueError('actor interface')
        predictions.append(prediction.numpy())
        contracts.append(dict(checkpoint=str(checkpoint),input_dim=686,action_dim=12,
            hidden_dims=list(model.actor.hidden_dims),finite_forward=True,
            state_keys=len(packet['model'])))
    assert contracts[0]['hidden_dims']==contracts[1]['hidden_dims']
    report=dict(status='DRY_RUN_PASS',task='ptv3-shared-wam-v6',
        run_id='s2b-calibration-transfer-preflight-20261022',
        decision='Does frozen single-trajectory calibration damage behavior away from its fit distribution?',
        classification='Decision; execution deferred, this run is engineering preflight only',
        distinct_sequence=other['sequence'],distinct_source=other['source'],distinct_states=other_states,
        distinct_object=other['object'],same_object_as_calibration=other['object']==original['object'],
        distinct_raw_terminal_mode=other_task['terminal_mode'],
        distinct_initial_max_q_difference=initial_q_difference,
        distinct_initial_max_hand_difference_m=initial_hand_difference,
        no_overlap_scope='different sequence and source recording from the calibration batch; WAM/ancestor lifetime exposure NOT certified',
        checkpoints=contracts,nominal_interface_samples_per_checkpoint=1,
        nominal_action_max_difference=float(np.abs(predictions[0]-predictions[1]).max()),
        minimum_protocol=dict(cells=[
            dict(label='seen-nominal',sequence=name,initial='same captured full initial/native reset state'),
            dict(label='seen-shifted-initial',sequence=name,
                initial='coherently shift hand root +5mm in world x; preserve object/table; recompute all native rigid bodies, q/dq, caches and previous actions; verify actual measured state'),
            dict(label='different-trajectory',sequence=other['sequence'],initial='its own declared measured initial state')],
            arms=['source-PD feasibility witness','frozen baseline R + GT-tau','same frozen calibrated R + GT-tau'],
            repeats_per_cell_arm=1,maximum_controls_per_rollout=600,maximum_rollouts=9,
            maximum_environment_controls=5400,optimizer_updates=0,wam_inference_calls=0,
            gpu_plan='co-locate native and frozen R on one idle physical4/5/6; keep two cards globally idle; split across two cards only if that idle-card floor still holds',
            wall_limit_s=900,output_limit_gib=1,minimum_free_gib=20,
            stop='abort only own episode on physical safety or malformed reset; no retry/tuning/new fitting',
            reference='uninterrupted source-PD witness per cell; endpoint + elapsed1s +8 observed future; declare final-PD hold; never splice or use live future',
            paired_initial='all three arms share and compare visible+hidden native reset state; source trajectories differ by cell',
            no_tau_fallback='same frozen executor zero-tau/valid=false; report actual mask each control',
            task_inputs='same declared narrow lift5cm/force/coupled/unsupported elapsed1s task in every arm; full original RELEASE scored separately, never substitute',
            metrics=['source feasibility','ever/final narrow success','first loss vs actual valid tau',
                     'longest elapsed hold','clip/safety/domain/fallback counts','frozen action RMSE on fixed reference states'],
            decision_rule='If any source-positive shifted/distinct cell loses baseline success after calibration, flag observed regression; paired hold decrease >=0.2s is a warning. All source-negative cells are inconclusive. A pass is only a small-screen pass, not a population non-inferiority claim.',
            expansion='two repeats of the failing cell only if a regression changes the next decision; no training; do not reuse for checkpoint selection'),
        training_started=False,simulator_started=False,gpu_allocated=False,optimizer_created=False,
        shifted_initial_reset_verified=False,transfer_evaluated=False,
        transfer_verdict='NOT_EVALUATED',python=sys.executable,python_version=sys.version.split()[0],
        numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
        limitations=['Nominal single-row forwards are compatibility checks, not transfer RMSE or physical performance.',
                     'Prefer same-object DEV; if unavailable use lexically first bounded different-object DEV and record the object/trajectory confound. Broader evaluation remains future evidence.',
                     'Historical source-PD original81 channel remains the fit target; extended reference is not silently treated as old calibration supervision.'])
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if a.output:
        if TASK not in a.output.resolve().parents:raise ValueError('Task-local output required')
        with a.output.open('x') as stream:stream.write(text)
    print(text)
    return report


if __name__=='__main__':main()
