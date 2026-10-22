"""BC of bounded recovery intent on REAL recorded drift states; no DAgger."""
import argparse,json,sys,time,subprocess,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[5];TASK=ROOT/'src/task/wam-rl-policy-v3';sys.path.insert(0,str(TASK/'src'))


def prepare(run,out):
    from scipy.spatial.transform import Rotation
    from wam_rl_policy_v3.grasp_geometry import GraspGeometry,JointAdmission,grasp_candidates
    from wam_rl_policy_v3.grasp_repair import support_plane
    from wam_rl_policy_v3.pose_consumer import pose7,grasp_intent_from_retargeter
    from wam_rl_policy_v3.intent_residual import residual_observation,RESIDUAL_SCALE
    catalog=json.loads((ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json').read_text())['records']
    x=[];y=[];sequences=[];sources=[];seen=set();rejected=0
    for path in sorted(run.glob('g*-resume2/manifest.json')):
        manifest=json.loads(path.read_text())
        for r in manifest['runs']:
            if r['arm']!='A1' or r['sequence'] in seen:continue
            seen.add(r['sequence']);folder=path.parent/r['id'];audit=json.loads((folder/'native-audit.json').read_text());control=json.loads((folder/'consumer.json').read_text())
            record=next(c for c in catalog if c['sequence']==r['sequence'])
            with np.load(Path(record['source'])/'canonical.npz') as z:points=z['points'].copy()
            with np.load(ROOT/'outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/sequences'/r['sequence']/'admission.npz') as z:bounds=z['table_bounds'].copy()
            admission=JointAdmission(audit['physical_joint_limits'],audit['controller']['offset'],audit['controller']['scale']);g=GraspGeometry.from_points(points)
            with np.load(folder/'trajectory.npz') as z:d={k:z[k].copy() for k in ('q','objects','palm_pose','table_pose','hand','adapter_raw_action','observation','tau_valid')}
            goal=pose7(d['objects'][0,0]);goal[2,3]+=.08;count=0
            for t in range(0,len(control),8):
                if control[t]['phase'] not in ('approach','close'):continue
                obj=pose7(d['objects'][t,0]);palm=pose7(d['palm_pose'][t,0]);q=d['q'][t,0];hand=d['hand'][t,0]
                intent=grasp_intent_from_retargeter(q,palm,obj,goal,np.clip(d['adapter_raw_action'][t],-1,1),admission.offset,admission.scale,tau_valid=bool(d['tau_valid'][t]))
                normal,height=support_plane(d['table_pose'][t,0],bounds)
                target,info=grasp_candidates(palm,hand,obj,points,normal,height,obj@intent.grasp_pose_object,admission.fingers(intent.preshape_q),q,admission)
                if target is None:rejected+=1;continue
                local=np.linalg.inv(obj)@target;error=local[:3,3]-intent.approach_pose_object[:3,3]
                if np.linalg.norm(error)<=g.position_tolerance_m:continue
                rotation=Rotation.from_matrix(local[:3,:3]@intent.approach_pose_object[:3,:3].T).as_rotvec()
                label=np.r_[error,rotation,admission.fingers(intent.preshape_q)-intent.preshape_q]/RESIDUAL_SCALE
                x.append(residual_observation(d['observation'][t],intent));y.append(np.clip(label,-.98,.98));sequences.append(r['sequence']);sources.append(dict(trajectory=str(folder/'trajectory.npz'),tick=t));count+=1
                if count>=40:break
    unique=sorted(set(sequences));dev={s for i,s in enumerate(unique) if i%3==2};mask=np.array([s in dev for s in sequences])
    if not mask.any() or mask.all():raise ValueError('need held-out sequences for offline BC evaluation')
    np.savez_compressed(out/'dataset.npz',x=np.array(x,np.float32),y=np.array(y,np.float32),dev=mask)
    metadata=dict(n=len(x),train=int((~mask).sum()),dev=int(mask.sum()),sequences=unique,dev_sequences=sorted(dev),rejected_no_admissible_candidate=rejected,
        label='bounded geometry recovery INTENT minus frozen R intent on measured drift states; analytic teacher labels, no new expert/DAgger collection',sources=sources)
    (out/'dataset.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({k:v for k,v in metadata.items() if k!='sources'}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path);p.add_argument('--prepare-only',action='store_true');p.add_argument('--gpu',type=int,choices=[4,5,6],default=4);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);os.environ['TMPDIR']=str(ROOT/'tmp')
    if a.dataset is None:
        if a.run is None:raise ValueError('recorded run required')
        prepare(a.run.resolve(),out);dataset=out/'dataset.npz'
    else:dataset=a.dataset.resolve()
    if a.prepare_only:return
    occupied=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if int(occupied)>64:raise RuntimeError('GPU occupied; queue BC')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from wam_rl_policy_v3.intent_residual import IntentResidualPolicy,RESIDUAL_CONTRACT
    torch.set_num_threads(2);torch.manual_seed(292);np.random.seed(292)
    with np.load(dataset) as z:x=torch.tensor(z['x'],device='cuda');y=torch.tensor(z['y'],device='cuda');dev=torch.tensor(z['dev'],device='cuda')
    policy=IntentResidualPolicy().cuda();optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4)
    def metrics():
        with torch.no_grad():prediction=policy(x)[0].tanh();loss=(prediction-y).square().mean(-1)
        return dict(train_mse=float(loss[~dev].mean()),dev_mse=float(loss[dev].mean()),maximum_normalized_residual=float(prediction.abs().max()))
    initial=metrics();train=torch.nonzero(~dev,as_tuple=False).flatten();start=time.monotonic();curves=[];gpu_samples=[]
    for update in range(400):
        if time.monotonic()-start>300:break
        ids=train[torch.randint(len(train),(min(128,len(train)),),device='cuda')];mean,_=policy(x[ids]);loss=(mean.tanh()-y[ids]).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
        if update%50==0:
            eta=(time.monotonic()-start)/(update+1)*(399-update)
            curves.append(dict(update=update,eta_s=eta,**metrics()))
            print(json.dumps(curves[-1]),flush=True)
        if update in (0,200,399):gpu_samples.append(dict(update=update,reading=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)))
    torch.cuda.synchronize();final=metrics();torch.save(dict(model=policy.state_dict(),contract=RESIDUAL_CONTRACT,seed=292,dataset=str(dataset)),out/'bc.pt')
    result=dict(status='COMPLETED',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dataset=str(dataset),seed=292,updates=update+1,wall_s=time.monotonic()-start,initial=initial,final=final,curve=curves,
        gpu=a.gpu,gpu_samples=gpu_samples,python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,gpu_memory_peak_bytes=torch.cuda.max_memory_allocated(),scope='offline intent recovery fit; held-out sequences, no physical efficacy claim; frozen WAM/R unchanged; DAgger deferred')
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
