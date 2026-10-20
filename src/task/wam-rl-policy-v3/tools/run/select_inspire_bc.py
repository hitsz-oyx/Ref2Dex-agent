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
    p.add_argument('--scheduled',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh selection output required')
    a.output.mkdir();os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(ROOT/'tmp/inspire-shadow-tests'))
    import numpy as np
    import torch
    from wam_rl_policy_v3.inspire_shadow import SCHEMA,CONTRACT,CURRENT_DIM,TAU_SLICE,BC_ADMISSION,expanded_model,initialize_scheduled,bc_inputs,bc_admission
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    parent=torch.load(a.current,map_location='cpu',weights_only=False)
    candidate=torch.load(a.scheduled,map_location='cpu',weights_only=False)
    if any(p['config'].get('training_admission')!=BC_ADMISSION for p in (parent,candidate)):
        raise ValueError('current and scheduled BC must use the task-active admission contract')
    baseline=expanded_model(True);baseline.x_scale[TAU_SLICE]=.05;initialize_scheduled(baseline,parent)
    fitted=expanded_model(True);fitted.load_state_dict(candidate['model'],strict=True)
    panel=np.load(a.cache/'dev-panel.npy');valid=bc_admission(a.cache,'dev',True);panel=panel[valid[panel]]
    x=torch.tensor(bc_inputs(a.cache,'dev',True)[panel],device='cuda');y=torch.tensor(np.load(a.cache/'dev-y.npy',mmap_mode='r')[panel],device='cuda')
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
