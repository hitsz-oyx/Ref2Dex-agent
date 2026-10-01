#!/usr/bin/env python3
"""GPU frozen checkpoint/selector wiring and held proposal support audit."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from run_paired_evaluator_resolution import sha,gpu_admission


def run(args):
    admission=gpu_admission(args.gpu);os.environ['CUDA_VISIBLE_DEVICES']=admission['uuid']
    import torch
    from analyze_executable_contact_opportunity import split_group
    from src.task.CmResidual.plan_consequence_selector import FrozenPlanSelector
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();selector=FrozenPlanSelector(args.checkpoint,'cuda');m=json.loads((args.run/'run_manifest.json').read_text())
    if m['run_status']!='COMPLETED' or args.output.exists():raise ValueError('terminal input/unique audit required')
    for k,v in m['input_sha256'].items():
        if sha(Path(k))!=v:raise ValueError('input drift')
    counts={mode:dict(rows=0,ood=0,proposals=0,gain_rows=0,gain_risk_rows=0,gain_risk_contact_rows=0,selected=[0]*8,actual_selected_matches=0,actual_base_matches=0,episodes=set()) for mode in selector.models}
    for p in m['phases']:
        path=Path(p['directory'])/'records.pt'
        if sha(path)!=p['result']['record_sha256']:raise ValueError('record drift')
        b=torch.load(path,map_location='cpu',weights_only=False)
        held=torch.tensor([split_group(int(i),int(j))>=70 for i,j in zip(b['motion_id'],b['start_frame'])]);clear=(b['initial_clearance']>=.002)&(b['state'][:,38]-b['rest_z']>=.03);rows=(held&clear).nonzero().flatten()
        for mode in selector.models:
            for begin in range(0,len(rows),96):
                ids=rows[begin:begin+96];choice,d=selector.choose(mode,b['history'][ids].cuda(),b['native_observation'][ids,0].cuda(),b['candidate_actions'][ids].cuda(),b['hold_target'][ids,3:6].cuda(),b['rest_z'][ids].cuda())
                if not torch.isfinite(d['score_mm']).all():raise ValueError('nonfinite forecast score')
                c=counts[mode];chosen=choice.cpu();c['rows']+=len(ids);c['ood']+=int(d['ood'].sum());c['proposals']+=int((chosen!=4).sum());c['selected']=[a+v for a,v in zip(c['selected'],torch.bincount(chosen,minlength=8).tolist())]
                c['gain_rows']+=int(d['gain_gate'].any(-1).sum());c['gain_risk_rows']+=int((d['gain_gate']&d['risk_gate']).any(-1).sum());c['gain_risk_contact_rows']+=int((d['gain_gate']&d['risk_gate']&d['contact_gate']).any(-1).sum())
                active=chosen!=4;c['actual_selected_matches']+=int(((b['assignment'][ids]==chosen)&active).sum());c['actual_base_matches']+=int(((b['assignment'][ids]==4)&active).sum());c['episodes'].update(b['episode_id'][int(i)] for i in ids[active])
    for c in counts.values():c['proposal_episodes']=len(c.pop('episodes'))
    result=dict(run_status='COMPLETED',kind='FROZEN_MODEL_SELECTOR_ENGINEERING_AND_COVERAGE',checkpoint_sha256=sha(args.checkpoint),selector_source_sha256=sha(ROOT/'src/task/CmResidual/plan_consequence_selector.py'),gpu=admission,modes=counts,elapsed_seconds=time.monotonic()-start,scope='held observed-state proposal support only; factual one-action matches are not individual counterfactuals or actual MPC utility')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);run(p.parse_args())
