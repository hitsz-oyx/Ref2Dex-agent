"""Diagnostic only: reference accuracy on retained/lost lifted-nearby BC states."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK/'src'), str(TASK.parent/'shadow-finetune/src')]
from shadow_finetune import paths
from video_data_preparation.unidexgrasp_data import object_poses
from wam_rl_policy.coordination import lifted_nearby


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2)
    reports = []
    for run in a.runs:
        manifest = json.loads((run/'manifest.json').read_text())
        assert manifest['mode'] == 'bc' and manifest['coordination_kind'] == 'wam'
        with np.load(run/'first-episode.npz') as d:
            objects = torch.as_tensor(d['object']);hands = torch.as_tensor(d['hand'])
            poses = torch.as_tensor(object_poses(d['object']), dtype=torch.float32)
        relative = (hands-poses[:, :, None, :3, 3]) @ poses[:, :, :3, :3]
        values, prelift_values = [], []
        for tick in range(1, 184, 16):
            ref = torch.load(run/('coordination-input-%03d.pt' % tick), map_location='cpu', weights_only=False)
            now, future = tick-1, tick-1+16
            use = ref['enabled']
            retained = lifted_nearby(hands[future], poses[future, :, :3, 3], objects[0, :, 2])
            prelift = ref['admitted'] & (objects[now, :, 2] < objects[0, :, 2]+.03)
            for mask, destination in [(use, values), (prelift, prelift_values)]:
                for env in mask.nonzero().flatten().tolist():
                    destination.append(dict(retained=bool(retained[env]),
                        wam_mm=float((relative[future, env]-ref['relative_target'][env]).norm(dim=-1).mean()*1000),
                        geometry_mm=float((relative[future, env]-relative[now, env]).norm(dim=-1).mean()*1000),
                        reference_change_mm=float((ref['relative_target'][env]-relative[now, env]).norm(dim=-1).mean()*1000)))
        groups = {}
        for label, subset in [('all', values), ('retained', [v for v in values if v['retained']]),
                              ('lost', [v for v in values if not v['retained']]),
                              ('prelift_admitted', prelift_values),
                              ('prelift_future_lifted_nearby', [v for v in prelift_values if v['retained']])]:
            groups[label] = dict(n=len(subset), **{key:float(np.mean([v[key] for v in subset])) if subset else None
                for key in ['wam_mm', 'geometry_mm', 'reference_change_mm']})
        # Same actual BC transitions and same admission/phase gate isolate how
        # much the predicted target changes the scalar bonus, not the policy.
        wam_scores, geometry_scores = [], []
        for tick in range(1, 200):
            refresh = 1+((tick-1)//16)*16
            if tick == refresh:
                ref = torch.load(run/('coordination-input-%03d.pt' % refresh), map_location='cpu', weights_only=False)
            active = ref['enabled'] & lifted_nearby(hands[tick], poses[tick, :, :3, 3], objects[0, :, 2])
            for target, scores in [(ref['relative_target'], wam_scores),
                                   (relative[refresh-1], geometry_scores)]:
                score = torch.exp(-(relative[tick]-target).square().sum(-1).mean(-1)/.0018)
                scores.extend(score[active].tolist())
        ws, gs = np.asarray(wam_scores), np.asarray(geometry_scores)
        differential = dict(enabled_transitions=len(ws), weight=manifest['coordination_weight'],
            wam_score_mean=float(ws.mean()) if len(ws) else None,
            geometry_score_mean=float(gs.mean()) if len(gs) else None,
            bonus_abs_difference_mean=float(np.abs(ws-gs).mean()*manifest['coordination_weight']) if len(ws) else None,
            bonus_abs_difference_p95=float(np.quantile(np.abs(ws-gs), .95)*manifest['coordination_weight']) if len(ws) else None)
        reports.append(dict(run=str(run), groups=groups, same_transition_bonus=differential))
    result = dict(scope='Post-hoc BC transition diagnostic; future used only for labels, not online targets or checkpoint selection; lifted-nearby is not measured contact', runs=reports)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':main()
