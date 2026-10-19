"""Apply Shadow's DEV selection including the zero-expanded current BC baseline."""
import argparse
import json
import os
from pathlib import Path
import sys

TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--current',type=Path,required=True)
    p.add_argument('--scheduled',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh selection output required')
    a.output.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES='6',TMPDIR=str(ROOT/'tmp/inspire-shadow-tests'))
    import numpy as np
    import torch
    from wam_rl_policy_v2.inspire_shadow import SCHEMA,CONTRACT,CURRENT_DIM,expanded_model,initialize_scheduled
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    parent=torch.load(a.current,map_location='cpu',weights_only=False)
    candidate=torch.load(a.scheduled,map_location='cpu',weights_only=False)
    baseline=expanded_model(True);baseline.x_scale[CURRENT_DIM:-1]=.05;initialize_scheduled(baseline,parent)
    fitted=expanded_model(True);fitted.load_state_dict(candidate['model'],strict=True)
    panel=np.load(a.cache/'dev-panel.npy');valid=np.load(a.cache/'dev-include.npy');panel=panel[valid[panel]]
    x=torch.tensor(np.load(a.cache/'dev-scheduled.npy',mmap_mode='r')[panel],device='cuda');y=torch.tensor(np.load(a.cache/'dev-y.npy',mmap_mode='r')[panel],device='cuda')
    metrics={}
    with torch.inference_mode():
        for name,model in [('initial_current_expansion',baseline),('scheduled_best',fitted)]:
            error=(model.cuda()(x)-y).abs();metrics[name]=dict(command_mae=float(error.mean()),root_mae=float(error[:,:6].mean()),finger_mae=float(error[:,6:].mean()))
    selected=min(metrics,key=lambda k:metrics[k]['command_mae'])
    if selected=='initial_current_expansion':
        candidate['model']={k:v.cpu() for k,v in baseline.state_dict().items()};candidate['step']=0
        candidate['optimizer']={};candidate['config']=dict(candidate['config'],selection='Shadow baseline-inclusive DEV command MAE',selected_initial_current=True)
    else:candidate['config']=dict(candidate['config'],selection='Shadow baseline-inclusive DEV command MAE',selected_initial_current=False)
    candidate['config']['conditioning']=CONTRACT
    torch.save(candidate,a.output/'best.pt')
    report=dict(status='COMPLETED',selected=selected,selected_step=candidate['step'],metrics=metrics,panel_rows=len(panel),
        current=str(a.current.resolve()),scheduled=str(a.scheduled.resolve()),cache=str(a.cache.resolve()),
        scope='same fixed object-balanced DEV pool; no behavioral outcome selection',reason='include the initial baseline just as Shadow train_current_bc.py')
    (a.output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
