"""Verify real-input forecasts and commands before any expanded-model update."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--width',type=int,default=512)
    p.add_argument('--layers',type=int,default=16)
    a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or ROOT/'outputs/ptv3-shared-wam' not in out.parents:raise ValueError('fresh owned output required')
    if subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    scratch=ROOT/'tmp'/out.name;scratch.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),TMPDIR=str(scratch),PYTHONDONTWRITEBYTECODE='1',
        TORCH_EXTENSIONS_DIR=str(ROOT/'tmp/shared-wam-extensions'))
    import numpy as np
    import torch
    from ptv3_shared_wam.checkpoint import load_checkpoint,atomic_json
    from ptv3_shared_wam.model import SharedWAM
    from ptv3_shared_wam.robot import RobotPair
    from ptv3_shared_wam.expansion import expand_forecaster
    from ptv3_shared_wam.scene import SceneHistoryEncoder
    from ptv3_shared_wam.rl_robot import RLRobotWindows
    from ptv3_shared_wam.data import shared_collate
    from ptv3_shared_wam.loading import to_device
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(124)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=load_checkpoint(a.checkpoint)
    source=RobotPair(SharedWAM(parent['stats'],dict(parent['config'],vectorized_padding=False)))
    source.load_state_dict(parent['model'],strict=True)
    source.forecaster.scene._sample_tokens=SceneHistoryEncoder._sample_tokens.__get__(source.forecaster.scene)
    target=RobotPair(SharedWAM(parent['stats'],dict(parent['config'],interaction_width=a.width,interaction_layers=a.layers,
        vectorized_padding=True,checkpoint_interaction=True)))
    state={k[len('forecaster.'):]:v for k,v in parent['model'].items() if k.startswith('forecaster.')}
    transfer=expand_forecaster(target.forecaster,state,parent['config'])
    rstate={k[len('retargeter.'):]:v for k,v in parent['model'].items() if k.startswith('retargeter.')}
    target.retargeter.load_state_dict(rstate,strict=True)
    source=source.cuda().eval();target=target.cuda().eval()
    dataset=RLRobotWindows(a.source,'train')
    indices=dataset.sample_indices(np.random.default_rng(124),8)
    batch=to_device(shared_collate([dataset[int(i)] for i in indices]),torch.device('cuda'))
    captured={}
    def capture(module,args,kwargs):
        captured.update(tokens=args[0].detach(),padding=kwargs['src_key_padding_mask'].detach())
    hook=source.forecaster.interaction.register_forward_pre_hook(capture,with_kwargs=True)
    with torch.no_grad():before,after=source(batch),target(batch)
    hook.remove()
    differences={key:float((before[key]-after[key]).abs().max()) for key in before}
    # Isolate the expansion's mathematical equivalence from FP32 kernel and
    # sparse-backbone numeric variation using identical actual input tokens.
    with torch.no_grad():
        tokens=captured['tokens'].double();padding=captured['padding']
        left=source.forecaster.interaction.double()(tokens,src_key_padding_mask=padding)
        repeat=a.width//source.forecaster.interaction_width
        right=target.forecaster.interaction.double()(tokens.repeat(1,1,repeat),src_key_padding_mask=padding)
        right=right.reshape(*right.shape[:-1],repeat,128).mean(-2)
        math_error=float((left-right).abs().max())
    passed=math_error<1e-9 and all(np.isfinite(value) and value< (1e-4 if key in ('scene','trajectory') else 5e-5)
               for key,value in differences.items())
    out.mkdir(parents=True)
    report=dict(status='PASSED' if passed else 'FAILED',precision='FP32; TF32 disabled',samples=8,
        parent_checkpoint=str(a.checkpoint.resolve()),source=str(a.source.resolve()),gpu=a.gpu,
        transfer=transfer,max_absolute_difference=differences,
        shared_same_input_fp64_max_abs=math_error,pipeline_position_tolerance_m=1e-4,
        note='metres for scene/trajectory, normalized command units for actions')
    atomic_json(out/'report.json',report);print(json.dumps(report),flush=True)
    if not passed:raise AssertionError('expanded initialization changed actual predictions')


if __name__=='__main__':main()
