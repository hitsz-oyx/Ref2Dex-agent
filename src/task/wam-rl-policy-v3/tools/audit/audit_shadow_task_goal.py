"""Check that trained geometric goal columns affect BC commands, beyond phase bias."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=7)
    a=p.parse_args()
    if a.output.exists():raise ValueError('fresh goal audit required')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import numpy as np
    import torch
    from wam_rl_policy_v3.bc_mlp import TaskShadowMLP
    from wam_rl_policy_v3.shadow_task_contract import CURRENT_SCHEMA,SCHEDULED_SCHEMA,conditioning
    torch.set_num_threads(2)
    packet=torch.load(a.checkpoint.resolve(),map_location='cpu',weights_only=False)
    if packet.get('schema') not in (CURRENT_SCHEMA,SCHEDULED_SCHEMA) or not packet.get('task_columns_trained'):
        raise ValueError('actually trained task10 policy required')
    with_tau=packet['schema']==SCHEDULED_SCHEMA
    if packet['config']['conditioning']!=conditioning(with_tau):raise ValueError('goal contract changed')
    cache=json.loads((a.data/'manifest.json').read_text());panel=np.load(a.data/'dev-panel.npy')
    if cache['schema']=='ref2dex.shadow-task-labels.v1':
        robot=np.load(Path(cache['base_cache'])/'dev-x.npy',mmap_mode='r')[panel]
        task=np.load(a.data/'dev-task.npy',mmap_mode='r')[panel]
        x=np.concatenate((robot,task),-1)
    else:x=np.load(a.data/'dev-x.npy',mmap_mode='r')[panel,:575 if with_tau else 310]
    torch.manual_seed(packet['config']['seed'])
    initial=TaskShadowMLP(with_tau)
    if with_tau:
        parent=torch.load(packet['config']['parent'],map_location='cpu',weights_only=False)
        initial.initialize_task_current(parent)
    reference=initial.net[0].weight[:,300:310].detach().clone()
    model=TaskShadowMLP(with_tau).cuda().eval();model.load_state_dict(packet['model'],strict=True)
    changed=model.net[0].weight[:,300:310].detach().cpu()-reference
    original=[];without=[]
    with torch.inference_mode():
        for lo in range(0,len(x),1024):
            inputs=torch.as_tensor(x[lo:lo+1024],device='cuda')
            counterfactual=inputs.clone();counterfactual[:,300:306]=0
            # Progress and valid one-hot phase stay identical; only local SE(3) changes.
            original.append(model(inputs).cpu().numpy());without.append(model(counterfactual).cpu().numpy())
    delta=np.abs(np.concatenate(original)-np.concatenate(without))
    nonzero=np.linalg.norm(x[:,300:306],axis=-1)>1e-7
    report=dict(checkpoint=str(a.checkpoint.resolve()),data=str(a.data.resolve()),gpu=a.gpu,
        git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.executable,step=int(packet['step']),input_dim=model.input_dim,rows=len(x),nonzero_geometric_goal_rows=int(nonzero.sum()),
        weight_movement={name:float(changed[:,sl].abs().max()) for name,sl in
            [('translation',slice(0,3)),('rotation',slice(3,6)),('progress',slice(6,7)),('phase',slice(7,10))]},
        geometric_goal_command_delta_mean=float(delta[nonzero].mean()) if nonzero.any() else None,
        geometric_goal_command_delta_max=float(delta.max()),zero_goal_control_max=float(delta[~nonzero].max()) if (~nonzero).any() else None,
        trained=bool(packet['task_columns_trained']),scope='offline geometric-goal dependence; not method gain or counterfactual execution success')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
