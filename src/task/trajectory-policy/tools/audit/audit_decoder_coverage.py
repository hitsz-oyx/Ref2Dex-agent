"""Frozen48D geometry coverage; this is not a learned policy evaluation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
TASK=Path(__file__).resolve().parents[2];ROOT=TASK.parents[2]
sys.path[:0]=[str(TASK/'src'),str(ROOT/'src/task/consequence-evaluator/src')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('geometry','reference','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);a=p.parse_args();started=time.monotonic()
    a.output=a.output.resolve()
    if a.output.exists() or ROOT/'outputs/trajectory-policy' not in a.output.parents:raise ValueError('fresh task output required')
    before=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    util,used=map(int,before.strip().split(','))
    if util>10 or used>512:raise ValueError('GPU not idle')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch
    from trajectory_policy.decoder import TrajectoryDecoder
    from trajectory_policy.inputs import load_geometry
    from consequence_evaluator.data import sha
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    urdf=ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf'
    g,initial,hand,hashes=load_geometry(a.geometry,a.reference,urdf)
    for file in (Path(__file__),TASK/'src/trajectory_policy/decoder.py',TASK/'src/trajectory_policy/inputs.py'):
        hashes[str(file.resolve())]=sha(file)
    a.output.mkdir(parents=True)
    m=dict(status='RUNNING',schema='ref2dex.trajectory-decoder-coverage.v1',input_sha256=hashes,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),physical_gpu=a.gpu,
        gpu_before=before.strip(),claim='Oracle hand-derived geometry labels only; not H-to-c inference')
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    try:
        ticks=np.arange(0,542,8);indices=np.minimum(ticks[:,None]+np.arange(1,25),542)
        current=g['q'][ticks];obj=np.repeat(initial['object_pose'][None],len(ticks),0)
        decoder=TrajectoryDecoder(urdf,'cuda:0');c=decoder.encode(g['q'][indices],current,obj)
        with torch.no_grad():decoded=decoder.decode(c,current,obj,1/30)
        points=decoded['hand'].cpu().numpy();target=hand[indices];dense=g['fitted_points'][indices]
        def errors(actual,reference):
            e=np.linalg.norm(actual-reference,axis=-1)
            return dict(point_3d_rmse_mm=float(np.sqrt((e**2).mean())*1000),
                prefix8_point_3d_rmse_mm=float(np.sqrt((e[:,:8]**2).mean())*1000),
                prefix8_palm_max_mm=float(e[:,:8,0].max()*1000),point_max_mm=float(e.max()*1000),
                per_point_3d_rmse_mm=(np.sqrt((e**2).mean((0,1)))*1000).tolist())
        result=dict(status='UNCLEAR',knots=[1,8,16,24],windows=len(ticks),raw_gt=errors(points,target),
            dense_geometry=errors(points,dense),native_evaluation_required=True)
        np.savez_compressed(a.output/'coverage.npz',ticks=ticks,c=c,q=decoded['q'].cpu().numpy(),hand=points,target=target,dense=dense)
        if time.monotonic()-started>120:raise TimeoutError('coverage audit exceeded120s')
        if any(sha(path)!=digest for path,digest in hashes.items()):raise ValueError('source drift')
        m.update(status='COMPLETED',elapsed_s=time.monotonic()-started,torch_peak_bytes=torch.cuda.max_memory_allocated())
        (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    except BaseException as e:
        m.update(status='FAILED',error=repr(e),elapsed_s=time.monotonic()-started);raise
    finally:(a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')


if __name__=='__main__':main()
