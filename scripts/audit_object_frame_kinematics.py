"""GPU bulk FK, independent SDK binding and fixed semantic alternatives only."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.run_contact_response_probe import sha
from scripts.analyze_static_hold_feasibility import rotation
from src.task.CmResidual.object_frame_kinematics import UrdfKinematics,object_frame_points

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();assert ROOT in out.parents and not out.exists();out.mkdir();begin=time.monotonic();torch.set_num_threads(2);assert torch.cuda.is_available()
    d=a.test/'s576';meta=json.loads((d/'physical_metadata.json').read_text());trace=torch.load(d/'trace.pt',map_location='cpu',weights_only=False);assert json.loads((d/'panel_audit.json').read_text())['run_status']=='COMPLETED'
    asset=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf';fk=UrdfKinematics(asset,meta['native_dof_names']);ids=meta['contact_body_ids'];names=[meta['native_body_names'][j] for j in ids];com=torch.tensor([meta['hand_body_com'][j] for j in ids],device='cuda')
    q=trace['native_q'].reshape(-1,18);root=trace['hand_root'].reshape(-1,13);point=trace['hand_body_position'].reshape(-1,5,3).numpy();quat=trace['hand_body_quaternion'].reshape(-1,5,4).numpy();obj=trace['object_root'].reshape(-1,13);root_expected=torch.zeros_like(root);root_expected[:,6]=1;root_error=float((root-root_expected).abs().max());errors={'link_origin':0.,'center_of_mass':0.,'rotation_matrix':0.};derived=[]
    for start in range(0,len(q),4096):
        n=min(4096,len(q)-start);positions,rotations=fk.forward(q[start:start+n].cuda(),root[start:start+n].cuda());pos=torch.stack([positions[name] for name in names],1);rot=torch.stack([rotations[name] for name in names],1);center=pos+(rot@com[None,...,None]).squeeze(-1)
        raw=point[start:start+n];errors['link_origin']=max(errors['link_origin'],float(np.abs(pos.cpu().numpy()-raw).max()));errors['center_of_mass']=max(errors['center_of_mass'],float(np.abs(center.cpu().numpy()-raw).max()))
        sdk_rotation=rotation(quat[start:start+n].reshape(-1,4)).reshape(n,5,3,3);errors['rotation_matrix']=max(errors['rotation_matrix'],float(np.abs(rot.cpu().numpy()-sdk_rotation).max()))
        derived.append(object_frame_points(pos,obj[start:start+n].cuda()).cpu())
    binding='link_origin' if errors['link_origin']<=5e-6 else 'center_of_mass' if errors['center_of_mass']<=5e-6 else None
    passes=root_error<=1e-7 and errors['rotation_matrix']<=2e-5 and binding is not None
    result=dict(run_status='COMPLETED',engineering_only=True,binding_valid=passes,binding=binding,position_tolerance_m=5e-6,rotation_matrix_tolerance=2e-5,root_tolerance=1e-7,maximum_error=errors,root_invariance_error=root_error,native_post_step_poses=len(q)*5,contact_body_names=names,no_models_or_policy_updates=True,not_actual_contact_points=True,old_poses_not_measured=True,asset_sha256=sha(asset),raw_trace_sha256=sha(d/'trace.pt'),metadata_sha256=sha(d/'physical_metadata.json'),script_sha256=sha(Path(__file__)),wall_seconds=time.monotonic()-begin)
    # Store explicit link-origin geometry even if SDK binding fails; never label
    # it as measured SDK position or usable scientific data on failure.
    torch.save(dict(link_origin_object_frame=torch.cat(derived).reshape(202,768,5,3),binding_valid=passes,sdk_position_semantics=binding),out/'derived_geometry.pt');(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
