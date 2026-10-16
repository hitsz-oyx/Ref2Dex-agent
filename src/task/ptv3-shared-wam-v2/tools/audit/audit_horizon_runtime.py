"""Check the actual loss function used by horizon runs without model/GPU inference."""
import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace


def isolated_loss(path):
    import torch
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='robot_loss')
    namespace=dict(forecast_loss=lambda *a:(torch.tensor(0.),{}),
                   target_tensor=lambda *a:torch.zeros(1,8,15))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    model=SimpleNamespace(action_representation='diagnostic',forecaster=SimpleNamespace(normalizer=None),
                          retargeter=SimpleNamespace(action_scale=torch.ones(8,15)))
    prediction=dict(action=torch.ones(1,8,15));prediction['action'][:,0]=0
    values={}
    for steps in (None,1,8,24):
        config=dict(anchor_weight=0.)
        if steps is not None:config['action_supervision_steps']=steps
        try:values[str(steps)]=float(namespace['robot_loss'](prediction,{},config,model)[1]['action'])
        except ValueError as error:values[str(steps)]=dict(error=str(error))
    return dict(source=str(path),function_lineno=node.lineno,
        fixture='(1,8,15); first action error0, all later action errors1; forecast0, anchor0',values=values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[5];out=a.output.resolve()
    if out.exists() or root/'outputs/ptv3-shared-wam-v2' not in out.parents:raise ValueError('Fresh Task output required')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    relative=Path('src/task/ptv3-shared-wam-v2/src/ptv3_shared_wam_v2/robot_model.py')
    current=isolated_loss(root/relative)
    sparse=isolated_loss(root/'tmp/shared-current-code-20261016-cont7k-sparse'/relative)
    assert current['values']['None']==current['values']['8']==.875
    assert current['values']['1']==0 and isinstance(current['values']['24'],dict)
    assert sparse['values']['1']==sparse['values']['8']==.875
    base=root/'outputs/ptv3-shared-wam-v2/shared-action-horizon-20261016-r1'
    runs={}
    for name in ('continuation-masked8-sparse','continuation-delta08-sparse','single-masked1-sparse','single-delta01-sparse'):
        status=json.loads((base/name/'process.json').read_text());command=status['command']
        source=next(x for x in command if x.endswith('/train_robot.py'))
        resume=command[command.index('--resume')+1]
        manifest=json.loads((base/name/'manifest.json').read_text())
        runs[name]=dict(training_source=source,resume_parent=resume,
            requested_steps=manifest['config'].get('action_supervision_steps'),
            sparse_loss_supervision_steps=8,
            horizon_classification='valid_eight_step' if name.startswith('continuation') else 'mixed_one_then_eight_not_valid_one_step_endpoint')
    common={}
    for name in ('continuation-masked8','continuation-delta08','single-masked1','single-delta01'):
        curve=json.loads((base/name/'curve.json').read_text());row=next(r for r in curve if r['step']==4000)
        common[name]=dict(checkpoint=str(base/name/'step4000.pt'),physical=row['physical']['PD_xyz_mm'])
    out.mkdir(parents=True)
    report=dict(status='CONFIRMED',scope='Isolated actual robot_loss function with controlled forecast/target dependencies; no model inference or new training',
        current_function=current,sparse_function=sparse,runs=runs,common_4000=common,
        correction='Forecast24 is separate from action8. Revoke24-step invalidation; sparse single endpoints mix1->8. Original results_summary.json retained unchanged.')
    (out/'correction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],current=current['values'],sparse=sparse['values'],output=str(out))))


if __name__=='__main__':main()
