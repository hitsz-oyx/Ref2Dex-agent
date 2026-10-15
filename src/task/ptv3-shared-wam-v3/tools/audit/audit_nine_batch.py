"""Replay all nine sources and compare main3 labels against the original collate."""
import argparse
import json
from pathlib import Path
import sys
TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.data import NineWindows,collate
from ptv3_shared_wam_v3.checkpoint import atomic_json
from ptv3_shared_wam.loading import forecasting_collate
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable audit receipt')
    recipe=json.loads((a.recipe/'recipe.json').read_text())
    dataset=NineWindows(recipe['main3_root'],recipe['native_root'],'train',recipe)
    panel=dataset.panel(1,17)
    samples=[dataset[ids[0]] for ids in panel.values()]
    mixed=collate(samples)
    old=forecasting_collate([s[0] for s in samples[:3]])
    checks={}
    for key in ('start_hand','start_hand_history','start_hand_valid','trajectory_target','trajectory_valid',
                'scene_points','scene_target','scene_valid','goal_object_delta'):
        value=old[key]
        view=mixed['model_inputs'][key][tuple(slice(0,n) for n in value.shape)]
        checks[key]=bool(torch.equal(value,view))
    if not all(checks.values()):raise ValueError('legacy label/goal/mask contract changed')
    for key in ('trajectory_target','scene_target'):
        if not torch.isfinite(mixed['model_inputs'][key]).all():raise ValueError('nonfinite mixed targets')
    if set(mixed['model_inputs']['source_id'].tolist())!={0,1,2,4,5,6,8,9,10}:
        raise ValueError('nine-source coverage incomplete')
    atomic_json(a.output,dict(status='PASS',source_ids=mixed['model_inputs']['source_id'].tolist(),
        main3_labels_identical=checks,provenance=mixed['provenance']))
    print(json.dumps(dict(status='PASS',checks=checks)),flush=True)


if __name__=='__main__':main()
