"""One teacher-history WAM forecast, fitted MANO surfaces, no BC/PPO execution."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
TASK=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('REF2DEX_RUN_ROOT',TASK.parents[2])).resolve()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--audit-manifest',type=Path,required=True);p.add_argument('--geometry-cache',type=Path,required=True)
 p.add_argument('--wam',type=Path,required=True);p.add_argument('--mano-root',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--sequence',default='s1_apple_lift');p.add_argument('--tick',type=int)
 p.add_argument('--gpu',type=int,default=7);p.add_argument('--fit-steps',type=int,default=300)
 a=p.parse_args();out=a.output.resolve()
 if out.exists() or ROOT/'outputs/wam-rl-policy-v3' not in out.parents:raise ValueError('fresh task preview required')
 out.mkdir(parents=True);os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',REF2DEX_DEEP_OPT='all')
 sys.path.insert(0,str(TASK/'src'))
 import numpy as np,torch,trimesh
 from scipy.spatial.transform import Rotation
 from wam_rl_policy_v3.teacher_dataset import TeacherRolloutCorpus
 from wam_rl_policy_v3.robot_adapter import InspireAdapter
 from wam_rl_policy_v3.wam_v4_runtime import V4Runtime,collate_v4,v4_online_samples
 from wam_rl_policy_v3.measured_execution import task_wam_goal
 from wam_rl_policy_v3.mano_preview import forecast_world,fit_mano,load_mano,MANO_BASE,MANO_TIPS
 torch.set_num_threads(2);torch.manual_seed(246);started=time.monotonic()
 corpus=TeacherRolloutCorpus(a.audit_manifest.resolve(),'train');idx=next(i for i,r in enumerate(corpus.records) if r['sequence']==a.sequence)
 episode=corpus.episode(idx);row=corpus.records[idx];source=Path(row['source']);n=len(episode['task'])
 candidates=np.flatnonzero(episode['active']&(episode['task'][:,8]>.5)&(np.arange(n)>=3)&(np.arange(n)+24<n))
 tick=int(candidates[0]) if a.tick is None and len(candidates) else a.tick
 if tick is None or not 3<=tick<n-24 or not episode['active'][tick]:raise ValueError('active H4 anchor and24 reference frames required')
 with np.load(source/'trajectory.npz') as z:
  poses=z['obj'][tick-3:tick+25].copy();hands=InspireAdapter.canonical_hand(z['hand'][tick-3:tick+25]);q=z['q'][tick:tick+25].copy()
 with np.load(corpus.root/row['sidecar']/'admission.npz') as z:table_pose=z['table_pose'].copy()
 env_objects=[dict(object_code='dexplore/'+row['object'],scale=1.)]
 runtime=V4Runtime(a.wam.resolve(),env_objects,a.geometry_cache.resolve(),dense_manifest=a.geometry_cache/'surface-manifest.json',allow_geometry_replacement=True,device='cuda:0')
 goal,error=task_wam_goal(poses[3],episode['local_goal_pose'][tick],episode['task'][tick],runtime.geometry[0][2])
 objects=np.c_[poses[:4,:3,3],Rotation.from_matrix(poses[:4,:3,:3]).as_quat()][None]
 sample=v4_online_samples(objects,hands[:4][None],runtime.geometry,goal_delta=goal[None])[0]
 if not runtime.workspace_valid([sample])[0]:raise ValueError('selected current WAM workspace invalid')
 batch={k:v.to(runtime.device) for k,v in collate_v4([sample]).items()}
 with torch.inference_mode(),torch.autocast('cuda',enabled=False):delta=runtime.predict_hand(batch)['trajectory'][0,:,0].float().cpu().numpy()
 targets=forecast_world(hands[3],delta,poses[3]);del runtime,batch;torch.cuda.empty_cache()
 # Geometry smoke: a known rigidly transformed MANO surface must remain recoverable.
 layer=load_mano(a.mano_root,'cuda:0')
 with torch.no_grad():
  r=layer(global_orient=torch.zeros((1,3),device='cuda'),hand_pose=torch.zeros((1,12),device='cuda'),betas=torch.zeros((1,10),device='cuda'))
  ref=torch.cat((r.joints[:,MANO_BASE],r.vertices[:,MANO_TIPS]),1);ref=(ref-r.joints[:,:1]).cpu().numpy()
 rot=Rotation.from_euler('xyz',[.2,-.3,.4]).as_matrix();ref=ref@rot.T+np.array([.1,.2,.8])
 check=fit_mano(ref,a.mano_root,steps=100)
 if check['fit_max_mm']>5:raise ValueError('MANO synthetic geometry smoke failed: '+str(check['fit_max_mm']))
 del layer,r;torch.cuda.empty_cache()
 fitted=fit_mano(targets,a.mano_root,steps=a.fit_steps)
 mesh_dir=out/'mano-assets';mesh_dir.mkdir()
 for i,vertices in enumerate(fitted['vertices']):
  # Keep the surface in world coordinates; each fixed actor is shown at identity.
  trimesh.Trimesh(vertices=vertices,faces=fitted['faces'],process=False).export(mesh_dir/('%02d.obj'%i))
  (mesh_dir/('%02d.urdf'%i)).write_text('<robot name="mano"><link name="surface"><visual><geometry><mesh filename="%02d.obj"/></geometry></visual></link></robot>'%i)
 np.savez_compressed(out/'preview.npz',target_hand=targets,wam_delta_anchor=delta,fit_hand=fitted['landmarks'],fit_vertices=fitted['vertices'],fit_faces=fitted['faces'],fit_errors_m=fitted['errors_m'],teacher_hand=hands[3:],teacher_q=q,object_poses=poses[3:],goal_pose=episode['local_goal_pose'][tick],table_pose=table_pose)
 report=dict(status='COMPLETED',task='wam-rl-policy-v3',experiment_id='P-20261020-wam-mano-preview',run_id=out.name,git_commit=os.environ.get('REF2DEX_CODE_COMMIT') or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=sys.executable,torch=torch.__version__,gpu=a.gpu,sequence=a.sequence,object=row['object'],source=str(source),source_manifest=str(a.audit_manifest.resolve()),tick=tick,anchor_selection='first current TRACK with H4 and24 in-prefix reference frames' if a.tick is None else 'explicit anchor',wam=str(a.wam.resolve()),wam_steps=24,wam_hz=30,hand='right only; absent left is not rendered',wam_inputs='teacher measured H4 + object CAD + current TaskController spatial goal; no future hand input',reference_role='future teacher q/object/hand used only for comparison after prediction',task_wam_goal_error_m=error,mano_root=str(a.mano_root),mano_fit={k:v for k,v in fitted.items() if k in ['scale','raw_scale','fit_mean_mm','fit_max_mm','shape','pose']},synthetic_fit_max_mm=check['fit_max_mm'],per_frame_fit_mean_mm=(fitted['errors_m'].mean(-1)*1000).tolist(),per_frame_prediction_error_mm=(np.linalg.norm(targets-hands[3:],axis=-1).mean(-1)*1000).tolist(),elapsed_s=time.monotonic()-started,scope='approximate fitted visualization; mesh is not direct WAM output or execution evidence')
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
