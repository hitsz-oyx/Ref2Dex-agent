"""Independent NumPy URDF FK audit of continuous-angle reset/PD equivalence."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
NATIVE_LINKS=('hand_base_link','thumb_proximal_base','thumb_tip','index_proximal','index_tip','middle_proximal','middle_tip','ring_proximal','ring_tip','pinky_proximal','pinky_tip')
def fk(urdf,dof_names,body_names,q):
 q=np.atleast_2d(q).astype(np.float64);xml=ET.parse(urdf).getroot();joints=xml.findall('joint');children={}
 for j in joints:children.setdefault(j.find('parent').get('link'),[]).append(j)
 roots=set(l.get('name') for l in xml.findall('link'))-set(j.find('child').get('link') for j in joints)
 if len(roots)!=1:raise ValueError('URDF topology')
 root=roots.pop();out={root:np.repeat(np.eye(4)[None],len(q),0)}
 def visit(parent):
  for j in children.get(parent,[]):
   origin=j.find('origin');xyz=np.array([float(v) for v in origin.get('xyz','0 0 0').split()]) if origin is not None else np.zeros(3);rpy=np.array([float(v) for v in origin.get('rpy','0 0 0').split()]) if origin is not None else np.zeros(3);T=np.repeat(np.eye(4)[None],len(q),0);T[:,:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();T[:,:3,3]=xyz
   if j.get('type')!='fixed':
    axis=np.array([float(v) for v in j.find('axis').get('xyz').split()]);axis/=np.linalg.norm(axis);value=q[:,dof_names.index(j.get('name'))]
    if j.get('type')=='prismatic':T[:,:3,3]+=np.einsum('bij,bj->bi',T[:,:3,:3],value[:,None]*axis)
    else:T[:,:3,:3]=T[:,:3,:3]@Rotation.from_rotvec(value[:,None]*axis).as_matrix()
   child=j.find('child').get('link');out[child]=out[parent]@T;visit(child)
 visit(root);return np.stack([out[k] for k in body_names],axis=1)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--branch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=json.loads((a.teacher/'manifest.json').read_text());b=json.loads((a.branch/'manifest.json').read_text());urdf=Path('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf');names=b['native_initial_audit']['native_dof_names'];bodies=m['observation_contract']['body_names'];z=np.load(a.teacher/'sequences/s7_cubemedium_pass_1/trajectory.npz');trace=np.load(a.branch/'trajectory.npz');controls=np.load(a.branch/'controls.npz');joint_by_name={j.get('name'):j for j in ET.parse(urdf).getroot().findall('joint')};joint_contract=[dict(index=i,name=name,kind=joint_by_name[name].get('type'),limit=joint_by_name[name].find('limit').attrib) for i,name in enumerate(names[:6])]
 raw=fk(urdf,names,bodies,z['q']);actual=fk(urdf,names,bodies,trace['q'][:,0]);normalized=z['q'][0].astype(float).copy();normalized[3:6]=(normalized[3:6]+np.pi)%(2*np.pi)-np.pi;equiv=fk(urdf,names,bodies,normalized);palm=bodies.index('hand_base_link');target=fk(urdf,names,bodies,controls['target'][:,0]);raw_target=fk(urdf,names,bodies,controls['raw_target'][:,0]);observed_target=fk(urdf,names,bodies,controls['actual_pd'][:,0]);hand_ids=[bodies.index(k) for k in NATIVE_LINKS]
 def angle(a,b):return float(Rotation.from_matrix(a.T@b).magnitude())
 out=dict(status='COMPLETED',joint_contract=joint_contract,native_dof_properties=b['native_initial_audit']['native_dof_properties'],independent_fk_vs_teacher_hand_max_error=float(np.abs(raw[:,hand_ids][:,:,:3,3]-z['hand']).max()),independent_fk_vs_branch_hand_max_error=float(np.abs(actual[:,hand_ids][:,:,:3,3]-trace['hand'][:,0]).max()),reset_wrap_all_link_transform_max_error=float(np.abs(raw[0]-equiv[0]).max()),normalized_initial_vs_raw_pose_max_error=float(np.abs(raw[0]-actual[0]).max()),raw_first_tick_palm_rotation_rad=angle(raw[0,palm,:3,:3],raw[1,palm,:3,:3]),branch_first_tick_palm_rotation_rad=angle(actual[0,palm,:3,:3],actual[1,palm,:3,:3]),raw_first_tick_palm_translation_m=float(np.linalg.norm(raw[1,palm,:3,3]-raw[0,palm,:3,3])),branch_first_tick_palm_translation_m=float(np.linalg.norm(actual[1,palm,:3,3]-actual[0,palm,:3,3])),all_target_raw_vs_equivalent_fk_max_error=float(np.abs(raw_target-target).max()),all_actual_pd_vs_equivalent_target_fk_max_error=float(np.abs(observed_target-target).max()),source_reference_clipped_controls=int((controls['clipped']&controls['within_reference']).sum()),clipped_controls=int(controls['clipped'].sum()),maximum_raw_pd_numeric_difference=float(np.abs(controls['raw_target']-controls['actual_pd']).max()),maximum_equivalent_pd_numeric_difference=float(np.abs(controls['target']-controls['actual_pd']).max()))
 out['registered_engineering_gate']=bool(out['branch_first_tick_palm_rotation_rad']<.1 and out['source_reference_clipped_controls']==0 and out['all_target_raw_vs_equivalent_fk_max_error']<1e-6 and out['all_actual_pd_vs_equivalent_target_fk_max_error']<1e-6)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'continuous-wrist-fk-audit.json').write_text(json.dumps(out,indent=2)+'\n');(a.output/'inspire_hand_right.urdf').write_bytes(urdf.read_bytes());print(json.dumps(out,indent=2))
if __name__=='__main__':main()
