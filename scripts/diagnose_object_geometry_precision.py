"""One same-data FP64 binding check; original failed scalar limits remain fixed."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_contact_response_probe import sha,admission

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();source=a.source.resolve();out=a.output.resolve();assert ROOT in out.parents and not out.exists();m=json.loads((source/'run_manifest.json').read_text());old=json.loads((source/'binding/results.json').read_text());assert m['run_status']=='COMPLETED' and not old['binding_valid']
    gpu=admission(4);os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid'];import numpy as np;import torch
    from src.task.CmResidual.object_frame_kinematics import UrdfKinematics
    from scripts.analyze_static_hold_feasibility import rotation
    out.mkdir();begin=time.monotonic();torch.set_num_threads(2);d=source/'s576';meta=json.loads((d/'physical_metadata.json').read_text());t=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);assert sha(d/'trace.pt')==old['raw_trace_sha256']
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf';assert sha(asset)==old['asset_sha256'];fk=UrdfKinematics(asset,meta['native_dof_names']);names=[meta['native_body_names'][j] for j in meta['contact_body_ids']];q=t['native_q'].reshape(-1,18);root=t['hand_root'].reshape(-1,13);actual=t['hand_body_position'].reshape(-1,5,3).numpy().astype(np.float64);actual_quat=t['hand_body_quaternion'].reshape(-1,5,4).numpy();sums=np.zeros((5,3));square=np.zeros((5,3));maximum=np.zeros(5);commonmax=0.;rotation_error=0.;local_min=np.full((5,3),np.inf);local_max=-local_min
    for start in range(0,len(q),4096):
        if time.monotonic()-begin>60:raise TimeoutError('fixed precision diagnosis budget')
        n=min(4096,len(q)-start);pos,rot=fk.forward(q[start:start+n].double().cuda(),root[start:start+n].double().cuda());pred=torch.stack([pos[k] for k in names],1).cpu().numpy();r=torch.stack([rot[k] for k in names],1).cpu().numpy();error=actual[start:start+n]-pred
        sums+=error.sum(0);square+=(error**2).sum(0);maximum=np.maximum(maximum,np.max(np.abs(error),(0,2)));local=np.einsum('nbji,nbj->nbi',r,error);local_min=np.minimum(local_min,local.min(0));local_max=np.maximum(local_max,local.max(0));common=error.mean(1,keepdims=True);commonmax=max(commonmax,float(np.abs(error-common).max()));sdk=rotation(actual_quat[start:start+n].reshape(-1,4)).reshape(n,5,3,3);rotation_error=max(rotation_error,float(np.abs(r-sdk).max()))
    error=float(maximum.max());result=dict(run_status='COMPLETED',engineering_only=True,gpu=gpu,double_link_origin_max_error_m=error,double_rotation_max_error=rotation_error,unchanged_position_limit_m=5e-6,unchanged_rotation_limit=2e-5,double_binding_pass=bool(error<=5e-6 and rotation_error<=2e-5),per_body_max_error_m=dict(zip(names,maximum.tolist())),per_body_mean_error_world_m=(sums/len(q)).tolist(),per_body_rms_world_m=np.sqrt(square/len(q)).tolist(),local_error_range_m=(local_max-local_min).tolist(),noncommon_error_max_m=commonmax,no_physics_optimizer_or_frame_correction=True,original_failed_binding_retained=True,original_manifest_sha256=sha(source/'run_manifest.json'),original_binding_sha256=sha(source/'binding/results.json'),raw_trace_sha256=sha(d/'trace.pt'),asset_sha256=sha(asset),script_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
