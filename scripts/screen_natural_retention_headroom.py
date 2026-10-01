"""Reused-data descriptive late-loss screen; only randomized training arm0."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'src/task/CmResidual/research/contact_response/output'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    begin = time.monotonic()
    torch.set_num_threads(2)
    root = BASE/'P-20261002-support-feature-policy-training-r1'
    out = BASE/'P-20261002-natural-retention-headroom-r1'
    if out.exists():
        raise FileExistsError(out)
    parent = json.loads((root/'run_manifest.json').read_text())
    assert parent['run_status'] == 'FAILED' and parent['last_update'] == 12
    hashes = {}
    rows = []
    for seed in range(529, 541):
        panel = root/f's{seed}'
        for name in ('initial.pt', 'trace.pt', 'audited.pt', 'panel_audit.json'):
            path = panel/name
            digest = sha(path)
            assert parent['input_sha256'][str(path)] == digest
            hashes[str(path)] = digest
        initial = torch.load(panel/'initial.pt', map_location='cpu', weights_only=False)
        trace = torch.load(panel/'trace.pt', map_location='cpu', weights_only=False)
        audited = torch.load(panel/'audited.pt', map_location='cpu', weights_only=False)
        mask = initial['arm_assignment'].numpy() == 0
        ids = np.flatnonzero(mask)
        assert len(ids) == 96
        motion = initial['motion'].numpy()[ids]
        roots = trace['object_root'].numpy()[:,ids]
        clear = trace['clearance'].numpy()[:,ids]
        rise = roots[:,:,2] - initial['initial_height'].numpy()[ids]
        valid = (rise >= np.float32(.03)) & (clear >= np.float32(.02))
        relative_speed = roots[:,:,9] - trace['native_dq'].numpy()[:,ids,2]
        for j, env in enumerate(ids):
            start = int(initial['lift_start'][motion[j]])
            stop = int(initial['phase_stop'][motion[j]])
            assert np.array_equal(trace['progress'][:,env].numpy(), np.arange(1,203))
            run = 0
            acquired = None
            first_loss = None
            downward = None
            for i in range(start-1, stop+30):
                run = run+1 if valid[i,j] else 0
                if acquired is None and run >= 5:
                    acquired = i+1
                if acquired is not None:
                    if first_loss is None and not valid[i,j]:
                        first_loss = i+1
                    if downward is None and relative_speed[i,j] <= -.02:
                        downward = i+1
            outcome = bool(audited['physical105'][env])
            assert outcome == bool(valid[stop-75:stop+30,j].all())
            rows.append(dict(seed=seed, environment=int(env), motion=int(motion[j]),
                physical105=outcome, acquired_tick=acquired, first_loss_tick=first_loss,
                relative_downward_tick=downward, phase_stop=stop))
        if time.monotonic()-begin > 120:
            raise TimeoutError('descriptive screen budget')
    summaries = {}
    for motion in range(3):
        cohort = [r for r in rows if r['motion'] == motion]
        loss = [r for r in cohort if r['acquired_tick'] is not None and not r['physical105']]
        summaries[str(motion)] = dict(n=len(cohort), physical105=sum(r['physical105'] for r in cohort),
            acquired5=sum(r['acquired_tick'] is not None for r in cohort),
            acquired5_but_fail105=len(loss), never_acquired5=sum(r['acquired_tick'] is None for r in cohort),
            first_loss_after_acquisition_ticks=[r['first_loss_tick']-r['acquired_tick'] for r in loss if r['first_loss_tick'] is not None],
            observable_downward_before_first_loss=sum(r['relative_downward_tick'] is not None and r['first_loss_tick'] is not None and r['relative_downward_tick'] < r['first_loss_tick'] for r in loss),
            loss_before_plateau_end=sum(r['first_loss_tick'] is not None and r['first_loss_tick']<=r['phase_stop'] for r in loss))
    for p,digest in hashes.items():
        assert sha(Path(p)) == digest
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    decision = ROOT/'docs/decisions/D-20261002-natural-retention-headroom.md'
    hashes[str(decision)] = sha(decision)
    result = dict(run_status='COMPLETED', classification='DESCRIPTIVE_REUSED_DATA',
        trajectories=len(rows), training_seeds=list(range(529,541)), arm=0,
        no_evaluation_data=True, no_physics_or_training=True, summaries=summaries,
        no_policy_utility_claim=True, wall_seconds=time.monotonic()-begin, input_sha256=hashes)
    out.mkdir()
    (out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))


if __name__ == '__main__':
    main()
