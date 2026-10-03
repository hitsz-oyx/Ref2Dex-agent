"""Four measured states and tiny causal-fit engineering checks, CPU only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.task.CmResidual.surface_execution import area_hand_samples,episode_split,selected_rows,fit_execution,predict_execution
from src.task.CmResidual.v118_planner import TorchInspireKinematics,QUERY_LINKS
from src.task.CmResidual.dexplore_cm_geometry import dexplore_root_pose
from scripts.audit_surface_execution import independent_links,independent_samples


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();d=a.source/'s655'
    initial=torch.load(d/'initial.pt',map_location='cpu',weights_only=False);trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);m=json.loads((d/'physical_metadata.json').read_text())
    train,held=episode_split(initial['motion'].numpy(),initial['arm_assignment'].numpy());rows=selected_rows(initial,trace,train[:2])
    fit=fit_execution(rows,'cpu');pred=predict_execution(rows,fit,initial['native_lower'].numpy(),initial['native_upper'].numpy())
    changed=dict(rows,next_q=rows['next_q']+.5,next_obj=rows['next_obj']+.5)
    altered=predict_execution(changed,fit,initial['native_lower'].numpy(),initial['native_upper'].numpy())
    assert all(np.array_equal(pred[k],altered[k]) for k in pred if k!='oracle')
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf';geometry=area_hand_samples(urdf);tri,link,bary=independent_samples(urdf,QUERY_LINKS)
    assert np.array_equal(link,geometry['links']) and np.array_equal(bary,geometry['barycentric'])
    assert np.allclose(tri,geometry['triangle_vertices'],atol=2e-8,rtol=1e-7)
    ids=torch.tensor([0,40,100,180]);env=torch.tensor([0,256,512,767]);q=trace['native_q'][ids,env]
    fk=TorchInspireKinematics(urdf,'cpu');poses=fk.forward(q[:,None])[:,0];independent=independent_links(q.numpy(),urdf)
    maximum=max(float(np.abs(independent[name]-poses[:,i].numpy()).max()) for i,name in enumerate(QUERY_LINKS));assert maximum<2e-6
    world=dexplore_root_pose(trace['hand_root'][ids,env])[:,None]@poses
    names=[m['native_body_names'][i] for i in m['contact_body_ids']];selected=[QUERY_LINKS.index(n) for n in names]
    sdk=float((world[:,selected,:3,3]-trace['hand_body_position'][ids,env]).abs().max());assert sdk<2e-4
    print(json.dumps(dict(engineering_smoke='PASS',tiny_cpu_reason='4states/32closed-fitrows avoid GPU startup',
                         independent_fk_max=maximum,sdk_origin_max_m=sdk,causal_future_label_isolation=True,global_area_sampler_reconstructed=True)),flush=True)


if __name__=='__main__':main()
