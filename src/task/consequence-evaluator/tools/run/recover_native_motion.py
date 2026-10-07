"""Recover one original corrected GRAB reference with bounded local writes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from consequence_evaluator.contracts import is_within


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',default='s3_airplane_lift')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=900)
    a=parser.parse_args()
    output=a.output.resolve()
    if (not is_within(output,ROOT/'outputs/consequence-evaluator') or output.exists()
            or a.sequence!='s3_airplane_lift' or not 1<=a.seconds<=900):
        parser.error('fresh task-owned s3 output and <=900s required')
    if shutil.disk_usage(ROOT).free<20*2**30:
        raise RuntimeError('disk below20GiB reserve')
    source=Path('/home2/wyy/oyx_ws/Ref2Dex')
    legacy=source/'data/processed_data/dexplore_grab/sequences'
    raw=source/'data/raw_data/GRAB'
    reference=source/'data/processed_data/inspire_geometric_dexplore'/a.sequence/'interaction_hand_inspire.pt'
    builder=ROOT/'src/task/CmResidual/tools/data/build_dexplore_v120_motion_input.py'
    converter=ROOT/'third_party/DExplore/data_processing/convert_grab.py'
    models=source/'data/raw_data/ARCTIC/arctic/models'
    skeletons=Path('/home2/wyy/oyx_ws/InterAct/simulation/intermimic/data/assets/smplx')
    robots=Path('/home2/wyy/oyx_ws/_external/dex_urdf_official_v120/robots/hands')
    files=[Path(__file__),builder,reference,legacy/a.sequence/'motion.npz',legacy/a.sequence/'object.npz',
           raw/'grab/s3/airplane_lift.npz',raw/'objects/airplane/mesh.obj',
           raw/'tools/subject_meshes/female/s3.ply',skeletons/'smplx_grab_s3.xml',
           *sorted(p for p in converter.parent.rglob('*') if p.is_file() and p.suffix in ('.py','.json')),
           *sorted(p for p in models.rglob('*') if p.is_file()),
           *sorted(p for p in robots.rglob('*') if p.is_file())]
    frozen={str(p.resolve()):digest(p) for p in files}
    output.mkdir(parents=True)
    (output/'tmp').mkdir();(output/'cache').mkdir()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TMPDIR=str(output/'tmp'),
             XDG_CACHE_HOME=str(output/'cache'),PYTHONDONTWRITEBYTECODE='1',
             DEXPLORE_ENABLE_SAPIEN='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    record=dict(status='INITIALIZING',task='consequence-evaluator',run_id=output.name,pid=os.getpid(),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                sources=frozen,seconds_budget=a.seconds,output_budget_bytes=2*2**30,
                device='cpu',cpu_reason='GPU0 trains the parent; GPUs1/2 train PointWorld and3-7 are foreign. One bounded geometry conversion avoids competing for parent GPU memory.',
                sequence=a.sequence,exact_historical_bytes=False)
    record['python']=sys.executable
    record['dependencies']={name:importlib.metadata.version(name) for name in ('torch','numpy','smplx','dex-retargeting','sapien','pin')}
    started=time.monotonic();process=None
    def save():
        record['elapsed_s']=time.monotonic()-started
        (output/'run_manifest.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    def stopped(signum,frame):
        raise KeyboardInterrupt('owned recovery stopped')
    signal.signal(signal.SIGTERM,stopped)
    def run(name,command):
        nonlocal process
        record.update(status='RUNNING',phase=name,command=command);save()
        with (output/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record['child_pid']=process.pid;save()
            while process.poll() is None:
                remaining=a.seconds-(time.monotonic()-started)
                if remaining<=0:
                    raise TimeoutError('fixed motion recovery deadline')
                if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>2*2**30:
                    raise RuntimeError('motion recovery exceeded2GiB')
                try:process.wait(timeout=min(10,remaining))
                except subprocess.TimeoutExpired:pass
            if process.returncode:
                raise RuntimeError(name+' exited '+str(process.returncode))
    def build(name,extra):
        return [sys.executable,str(builder),'--sequence',a.sequence,'--legacy-root',str(legacy),
                '--raw-root',str(raw/'grab'),'--raw-object-root',str(raw/'objects'),
                '--raw-tools-root',str(raw/'tools'),'--skeleton-root',str(skeletons),
                '--output-root',str(output/name),'--manifest',str(output/(name+'_manifest.json')),*extra]
    def convert(name,target):
        return [sys.executable,str(converter),'--robot','inspire','--grab_dir',str(output/name),
                '--original_grab_dir',str(raw/'grab'),'--object_dir',str(output/name/'objects'),
                '--smplx_model_dir',str(models),'--skeleton_dir',str(output/name/'skeletons'),
                '--dex_retarget_dir',str(robots),'--output_dir',str(output/target),
                '--filter',a.sequence,'--retarget-iterations','1','--retarget-stride','1']
    save()
    try:
        run('build_baseline',build('baseline_input',[]))
        run('convert_baseline',convert('baseline_input','baseline_converted'))
        baseline=output/'baseline_converted'/a.sequence/'interaction_hand_inspire.pt'
        run('build_corrected',build('corrected_input',['--baseline-tensor',str(baseline),'--reference-tensor',str(reference)]))
        run('convert_corrected',convert('corrected_input','corrected_converted'))
        import torch
        corrected=output/'corrected_converted'/a.sequence/'interaction_hand_inspire.pt'
        b,c,r=[torch.load(p,map_location='cpu',weights_only=True) for p in (baseline,corrected,reference)]
        if any(x.shape!=(543,598) or x.dtype!=torch.float32 or not torch.isfinite(x).all() for x in (b,c,r)):
            raise ValueError('native corrected tensor shape/dtype/finite mismatch')
        relative_error=((c[:,51:54]-c[:,198:201])-(r[:,51:54]-r[:,198:201])).abs().max().item()
        table_error=((c[:,198:201]-c[:,238:241])-(b[:,198:201]-b[:,238:241])).abs().max().item()
        alignment=json.loads((output/'corrected_input_manifest.json').read_text())
        if (relative_error>1e-5 or table_error>1e-5
                or alignment['body_translation_alignment']['method']!='right_hand_object_relative_inverse_rotation_x90'):
            raise ValueError('corrected right-wrist/object alignment failed')
        if any(digest(p)!=sha for p,sha in frozen.items()):
            raise RuntimeError('motion source/code drift')
        audit=dict(right_wrist_object_error_m=relative_error,
                   object_table_relative_error_m=table_error,
                   contact_values=torch.unique(c[:,205:238]).tolist(),
                   left_active=int((c[:,206:222]>.5).sum()),right_active=int((c[:,222:238]>.5).sum()),
                   baseline_sha256=digest(baseline),corrected_sha256=digest(corrected),
                   limitation='reference reconstruction; not robot rollout or grasp evidence')
        (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        record.update(status='COMPLETED',corrected_tensor=str(corrected),corrected_sha256=digest(corrected))
    except BaseException as error:
        record.update(status='TIMED_OUT' if isinstance(error,TimeoutError) else 'FAILED',error=repr(error));raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        save()


if __name__=='__main__':
    main()
