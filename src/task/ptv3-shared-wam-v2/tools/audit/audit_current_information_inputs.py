"""Check that the trained added inputs affect predictions; not a new control arm."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK/'tools/run'))


def run(out, gpu):
    if subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('assigned GPU occupied')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    scratch = TASK.parents[2]/'tmp'/out.name/'input-audit'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    import numpy as np
    import torch
    from current_information import InformationRetargeter, pack
    from probe_current_information import evaluate
    from ptv3_shared_wam_v2.robot_data import V2RLRobotWindows
    from ptv3_shared_wam.action_targets import GEOMETRY_LIFTED
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    result = json.loads((out/'result.json').read_text())
    source = result['provenance']['both']
    directory = Path(source['directory'])
    initial = torch.load(directory/'initial_R.pt', map_location='cpu', weights_only=False)
    c = initial['config']
    dataset = V2RLRobotWindows(initial['parent_provenance']['source'], 'val', 32, GEOMETRY_LIFTED,
        dict(schema=c['encoder_workspace_admission'], origin_m=c['voxel_origin_m'], voxel_m=c['voxel_m']))
    packed = pack(dataset)
    panel = np.load(directory/'panels.npz')['dev_indices']
    data = {k:torch.as_tensor(v[panel], device='cuda') for k,v in packed.items()}
    stats = np.load(directory/'extra_train_stats.npz')
    model = InformationRetargeter(initial['model'], stats['mean'], stats['std']).cuda()
    saved = torch.load(directory/('both-step%d.pt'%source['selected_step']), map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    indices = torch.arange(len(panel), device='cuda')
    rows, predictions = {}, {}
    for arm in ['both', 'current_reference', 'current_geometry', 'baseline']:
        rows[arm], predictions[arm] = evaluate(model, data, indices, arm, dataset.controller, dataset.records)
    np.testing.assert_allclose(rows['both']['xyz_mean_mm'],
        result['arms']['both']['selected']['panels']['val']['xyz_mean_mm'], rtol=1e-4, atol=1e-5)
    scale = model.base.action_scale.detach().cpu().numpy()
    differences = {arm:float(np.mean(abs(pred-predictions['both'])/scale))
                   for arm,pred in predictions.items() if arm != 'both'}
    assert all(value > 1e-6 for value in differences.values()), differences
    payload = dict(scope='inference input sensitivity of selected both model; not separately trained comparisons',
                   normalized_all8_prediction_change=differences, mask_metrics=rows,
                   final_extra_weight_norm=float(model.extra[-1].weight.detach().norm()))
    (out/'input_sensitivity.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps(differences))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()
    run(args.output, args.gpu)
