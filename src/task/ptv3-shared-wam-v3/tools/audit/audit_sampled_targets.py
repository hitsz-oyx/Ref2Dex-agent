"""Replay a logged step's labels to attribute unusually large mixed-batch loss."""
import argparse
import json
from pathlib import Path
import sys
TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.data import NineWindows,collate
from ptv3_shared_wam_v3.normalization import HorizonNormalizer
from ptv3_shared_wam_v3.objectives import forecast_loss
from ptv3_shared_wam_v3.checkpoint import atomic_json
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable target-audit receipt')
    c=json.loads((a.run/'config.json').read_text());r=json.loads((a.run/'recipe.json').read_text())
    n=HorizonNormalizer(json.loads((a.run/'horizon-stats.json').read_text()))
    world=json.loads((a.run/'manifest.json').read_text())['provenance']['world_size']
    torch.set_num_threads(1)
    d=NineWindows(r['main3_root'],r['native_root'],'train',r)
    report=[]
    for rank in range(world):
        windows=[d[i] for i in d.draw(c['batch_size'],c['seed'],a.step,rank)]
        for position,sample in enumerate(windows):
            b=collate([sample])['model_inputs']
            pred={key+'_normalized':torch.zeros_like(b[key+'_target']) for key in ('trajectory','scene')}
            loss,terms=forecast_loss(pred,b,c,n)
            valid=b['scene_valid'] & b['object_valid'][:,:,None,None]
            values=b['scene_target'][valid]
            report.append(dict(rank=rank,position=position,source_id=int(b['source_id'][0]),
                zero_prediction_normalized_loss=float(loss),hand=float(terms['hand']),scene=float(terms['scene']),
                scene_max_displacement_m=float(values.norm(dim=-1).max()),
                scene_rms_displacement_m=float(values.square().sum(-1).mean().sqrt()),
                provenance=sample[1]))
    report.sort(key=lambda x:x['scene'],reverse=True)
    atomic_json(a.output,dict(step=a.step,scope='CPU target replay, zero normalized prediction; no model inference or effect conclusion',
        maximum_scene_displacement_m=max(x['scene_max_displacement_m'] for x in report),
        windows=len(report),top20=report[:20]))
    print(json.dumps(dict(windows=len(report),top3=[{k:v for k,v in row.items() if k!='provenance'} for row in report[:3]])),flush=True)


if __name__=='__main__':main()
