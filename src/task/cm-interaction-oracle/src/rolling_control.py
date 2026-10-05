"""Same-current-state rolling oracle contracts, independent of Isaac Gym."""
import numpy as np
import torch
from oracle_y_utility import CANDIDATES, utility, paired_bootstrap, stable_grasp_z
from rolling_y_audit import short_y

OFFSETS = tuple(range(0, 89, 8))
TOLERANCE = torch.tensor([1e-4]*5+[1e-5, 0.])


def candidate_scores(panels, rows):
    """All seven physical forks must share the actual current H, never later paths."""
    if len(panels) != len(CANDIDATES):
        raise ValueError('all seven candidates required')
    reference = panels[0]
    scores, labels = [], []
    for k, p in enumerate(panels):
        if p['candidate'] != k or p['post_window'] != 32:
            raise ValueError('candidate/window identity mismatch')
        for key in ('initial_fingerprint','simulation_contract','model_fingerprint',
                    'rms_fingerprint','rolling_offset'):
            if p[key] != reference[key]:
                raise ValueError('fork provenance mismatch: '+key)
        for key in ('triggers','rest_height','motion_id','start_frame','delta'):
            if not torch.equal(p[key],reference[key]):
                raise ValueError('fork identity mismatch: '+key)
        for key in ('before','history','actor_obs','hand_root'):
            if not torch.allclose(p[key][rows],reference[key][rows],rtol=0,atol=1e-4):
                raise ValueError('different current H: '+key)
        if not (p['full_world_prefix_errors'] <= TOLERANCE).all():
            raise ValueError('full-world replay failed')
        if not p['valid_steps'][rows].all():
            raise ValueError('incomplete lookahead; no filtering/truncation')
        y, _ = short_y(p['before'][rows,2],p['before'][rows,71]>.5,
                       p['height'][rows],p['pair'][rows],p['rest_height'][rows])
        labels.append(y); scores.append(utility(y))
    return torch.stack(labels,1), torch.stack(scores,1)


def mixed_plan(scores, rows, num_envs):
    if scores.shape != (len(rows), len(CANDIDATES)) or not torch.isfinite(scores).all():
        raise ValueError('finite complete scores required')
    choices = torch.zeros(num_envs,dtype=torch.long)
    choices[rows] = scores.argmax(-1)  # exact ties baseline then fixed arm order
    return choices


def execution_z(trace, rows, origin, rest):
    """Only a genuinely executed mixed path provides rolling Z, never fork mosaics."""
    after = trace['after_physical'][origin:origin+90,rows].transpose(0,1)
    done = trace['done'][origin:origin+90,rows].transpose(0,1)
    if after.shape[1] != 90 or done.any() or not torch.isfinite(after).all():
        raise ValueError('full actual90step nonterminal outcome required')
    return stable_grasp_z(after[:,:,2],after[:,:,71]>.5,rest)


def control_gate(baseline, rolling, motion, seed=268):
    baseline = np.asarray(baseline,dtype=int); rolling = np.asarray(rolling,dtype=int)
    if baseline.shape != rolling.shape or len(baseline) != len(motion) or not len(baseline):
        raise ValueError('paired outcomes required')
    gain = paired_bootstrap(rolling-baseline,seed=seed)
    adequate = len(baseline) >= 30 and len(np.unique(motion)) >= 2
    passed = adequate and gain['gain'] >= .05 and gain['lower95'] > 0
    return dict(status='PROMISING' if passed else ('UNPROMISING' if adequate else 'UNCLEAR'),
                passed=bool(passed),baseline_count=int(baseline.sum()),
                rolling_count=int(rolling.sum()),anchors=len(baseline),gain=gain,
                rescued=int(((baseline==0)&(rolling==1)).sum()),
                harmed=int(((baseline==1)&(rolling==0)).sum()),adequate_support=adequate)
