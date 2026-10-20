"""Check a real Current421 expansion and its first tau-gradient update on admitted rows."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('REF2DEX_RUN_ROOT', TASK.parents[2])).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--tau-cache', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=6)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:
        raise ValueError('fresh v3 audit output required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
    import sys
    sys.path.insert(0, str(TASK/'src'))
    import torch
    from wam_rl_policy_v3.measured_execution import GOAL_CONTRACT, TAU_LAYOUT_SCHEMA
    from wam_rl_policy_v3.measured_policy import MeasuredInspirePolicy
    from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus

    torch.set_num_threads(2); torch.manual_seed(246)
    receipt = json.loads((args.tau_cache/'manifest.json').read_text())
    if receipt.get('status') != 'COMPLETED':
        raise ValueError('completed immutable cache required for warmstart audit')
    layout = json.loads((args.tau_cache/'layout.json').read_text())
    if layout.get('schema') != TAU_LAYOUT_SCHEMA or layout.get('goal_contract') != GOAL_CONTRACT:
        raise ValueError('corrected shared-goal tau cache required')
    tau = np.load(args.tau_cache/'train-tau.npy', mmap_mode='r')
    valid = np.load(args.tau_cache/'train-valid.npy', mmap_mode='r')
    corpus = TeacherRolloutCorpus(args.manifest, 'train')
    by_name = {r['sequence']: i for i,r in enumerate(corpus.records)}
    pieces = []; objects = set(); sequences = []
    for row in layout['records']:
        if row['split'] != 'train' or row['object'] in objects:
            continue
        episode = corpus.episode(by_name[row['sequence']])
        ids = np.flatnonzero(episode['active'] & valid[row['lo']:row['hi']])[:32]
        if not len(ids):
            continue
        pieces.append(dict(measured=episode['measured'][ids],task=episode['task'][ids],action=episode['action'][ids],
            points=np.repeat(episode['object_points'][None],len(ids),axis=0),tau=np.asarray(tau[row['lo']+ids]).reshape(-1,8,11,3)))
        objects.add(row['object']); sequences.append(row['sequence'])
        if len(objects) == 8:
            break
    if len(objects) != 8:
        raise ValueError('eight real object episodes with valid tau required')
    values = {key: torch.as_tensor(np.concatenate([p[key] for p in pieces]), device='cuda') for key in pieces[0]}
    mask = torch.ones(len(values['action']),device='cuda',dtype=torch.bool)
    packet = torch.load(args.parent, map_location='cpu', weights_only=False)
    if packet.get('schema') != 'ref2dex.inspire-measured-current-bc.v1':
        raise ValueError('real Current421 checkpoint required')
    parent = MeasuredInspirePolicy().cuda().eval(); parent.load_state_dict(packet['model'],strict=True)
    scheduled = MeasuredInspirePolicy(with_tau=True).cuda().eval(); scheduled.initialize_current(packet)
    with torch.no_grad():
        expected = parent(values['measured'], values['task'], values['points'])
        predicted = scheduled(values['measured'], values['task'], values['points'], tau=values['tau'], tau_valid=mask)
        equality_error = float((expected-predicted).abs().max())
        new_nonzero = int(torch.count_nonzero(scheduled.actor.net[0].weight[:,421:]))
        assert new_nonzero == 0 and equality_error < 3e-6
        shape = parent.shape_report(values['points'])
    scheduled.train()
    prediction = scheduled(values['measured'],values['task'],values['points'],tau=values['tau'],tau_valid=mask)
    loss = ((prediction-values['action'])/scheduled.actor.y_scale).abs().mean()
    optimizer = torch.optim.AdamW(scheduled.parameters(),lr=3e-4,weight_decay=1e-5)
    optimizer.zero_grad(set_to_none=True); loss.backward()
    gradient = scheduled.actor.net[0].weight.grad
    tau_gradient = float(gradient[:,421:685].abs().max())
    valid_gradient = float(gradient[:,685].abs().max())
    assert tau_gradient > 0 and valid_gradient > 0
    torch.nn.utils.clip_grad_norm_(scheduled.parameters(),1.); optimizer.step()
    assert torch.count_nonzero(scheduled.actor.net[0].weight[:,421:]) > 0
    result = dict(status='COMPLETED',schema='ref2dex.measured-warmstart-audit.v1',
        git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,torch=torch.__version__,gpu=args.gpu,parent=str(args.parent.resolve()),parent_step=packet['step'],
        source_manifest=str(args.manifest.resolve()),tau_cache=str(args.tau_cache.resolve()),sequences=sequences,
        objects=len(objects),rows=len(mask),initial_output_max_error=equality_error,initial_added_nonzero_weights=new_nonzero,
        first_tau_gradient_max=tau_gradient,first_valid_gradient_max=valid_gradient,shape_report=shape,
        optimizer_steps=1,checkpoint_saved=False,task_success_claim=False)
    out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
