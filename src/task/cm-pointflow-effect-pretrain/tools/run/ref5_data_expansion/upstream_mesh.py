"""Run original ObjectForesight step8 multi-image reconstruction and texturing.

Manual clean-frame/mask inputs replace EPIC-specific automatic selection.
Sampling, Gaussian reconstruction, mesh postprocessing and GLB texture baking
are executed by the original vendored entrypoint with its default settings.
"""
import argparse
import faulthandler
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from native_data import sha


class MeshDeadlineExceeded(BaseException):
    """Do not let recoverable upstream error handlers swallow a deadline."""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'video', 'masks', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--clean-start', type=int, required=True)
    p.add_argument('--clean-end', type=int, required=True)
    p.add_argument('--mask-start', type=int, required=True)
    p.add_argument('--mask-end', type=int, required=True)
    p.add_argument('--seconds', type=int, default=1800)
    p.add_argument('--diagnostics', action='store_true',
                   help='log original postprocessing phases and periodically dump Python stacks')
    p.add_argument('--mesh-face-budget',type=int,
                   help='optional face budget using original to_glb simplification parameter')
    a = p.parse_args()
    if not 0 <= a.clean_start < a.clean_end or not 1 <= a.seconds <= 1800:
        p.error('invalid clean interval / time budget')
    if a.mesh_face_budget is not None and not 10000 <= a.mesh_face_budget <= 100000:
        p.error('pilot mesh face budget must be10000..100000')
    a.root, a.output = a.root.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    obj = a.output/'object_0'
    obj.mkdir()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    masks = np.load(a.masks)['masks']
    clean = range(a.clean_start, a.clean_end+1)
    selected = range(a.mask_start,a.mask_end+1)
    if (not 0 <= a.mask_start <= a.clean_start < a.clean_end <= a.mask_end < len(masks)
            or len(clean) < max(3,len(selected)//5)):
        raise ValueError('clean inputs do not meet original step8 coverage gate')
    np.savez_compressed(obj/'masks.npz', **{str(i):masks[i] for i in selected})
    np.savez_compressed(obj/'clean_masks.npz', **{str(i):masks[i] for i in clean})
    cap = cv2.VideoCapture(str(a.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,a.clean_start)
    crops = {}
    try:
        for i in clean:
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('native clean frame decode failed')
            mask = cv2.resize(masks[i].astype('uint8'),(bgr.shape[1],bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            x,y,w,h = cv2.boundingRect(mask)
            rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            crops[str(i)] = np.where(mask[y:y+h,x:x+w,None]>0,
                                     rgb[y:y+h,x:x+w],0)
    finally:
        cap.release()
    np.savez_compressed(obj/'clean_cropped_frames.npz',**crops)
    vendor = a.root/'vendor/ObjectForesight-Data'
    sys.path.insert(0,str(vendor))
    sys.path.insert(0,str(vendor/'trellis'))
    torch.hub.set_dir(str(a.root/'torch-hub'))
    # Preserve the original pretrained factory. Redirect only the DINO source
    # from GitHub to the already acquired, pinned local copy and checkpoint.
    original_hub_load = torch.hub.load
    def local_hub_load(repo, model, *args, **kwargs):
        if repo == 'facebookresearch/dinov2':
            kwargs['source'] = 'local'
            return original_hub_load(str(a.root/'vendor/dinov2'),model,*args,**kwargs)
        return original_hub_load(repo,model,*args,**kwargs)
    torch.hub.load = local_hub_load
    source = vendor/'step8_trellis.py'
    spec = importlib.util.spec_from_file_location('objectforesight_step8',source)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    simplification=[]
    if a.mesh_face_budget is not None:
        original_to_glb=upstream.postprocessing_utils.to_glb
        def bounded_to_glb(appearance,mesh,*args,**kwargs):
            faces=int(mesh.faces.shape[0])
            original_ratio=float(kwargs.get('simplify',.95))
            effective=max(original_ratio,1-a.mesh_face_budget/faces)
            choice=dict(raw_faces=faces,original_simplify=original_ratio,
                        effective_simplify=effective,face_budget=a.mesh_face_budget)
            simplification.append(choice)
            (a.output/'simplification_choice.json').write_text(json.dumps(choice,indent=2)+'\n')
            print(json.dumps(choice),flush=True)
            kwargs['simplify']=effective
            return original_to_glb(appearance,mesh,*args,**kwargs)
        upstream.postprocessing_utils.to_glb=bounded_to_glb
    if a.diagnostics:
        faulthandler.enable()
        faulthandler.dump_traceback_later(45,repeat=True)
        def instrument(name):
            original=getattr(upstream.postprocessing_utils,name)
            def measured(*args,**kwargs):
                tick=time.monotonic()
                print(json.dumps(dict(diagnostic_phase=name,event='START',
                                      shapes=[list(x.shape) for x in args if hasattr(x,'shape')])),flush=True)
                if name=='parametrize_mesh':
                    np.savez_compressed(a.output/'mesh_before_uv.npz',vertices=args[0],faces=args[1])
                result=original(*args,**kwargs)
                print(json.dumps(dict(diagnostic_phase=name,event='END',elapsed_s=time.monotonic()-tick)),flush=True)
                return result
            setattr(upstream.postprocessing_utils,name,measured)
        for name in ['postprocess_mesh','_fill_holes','parametrize_mesh','bake_texture']:
            instrument(name)
    old_argv = sys.argv
    try:
        sys.argv=[str(source)]
        cfg=upstream.load_config()
    finally:
        sys.argv=old_argv
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # Original defaults: 10 images, 40 steps, guidance10, simplify0.7,
    # texture2048, 100 Gaussian views, 1000 visibility/hole-fill views.
    started = time.monotonic()
    report = dict(status='RUNNING',pid=os.getpid(),training_allowed=False,
                  upstream_sha256=sha(source),script_sha256=sha(Path(__file__)),
                  input_sha256={str(x.resolve()):sha(x) for x in (a.video,a.masks)},
                  clean_source_frames=list(clean),config=vars(cfg),seconds_budget=a.seconds,
                  selected_source_frames=[a.mask_start,a.mask_end],
                  local_dino_source=True,core='original step8 process_single_object and save_3d_models')
    report['diagnostics']=a.diagnostics
    report['mesh_face_budget']=a.mesh_face_budget
    target=a.output/'mesh_manifest.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    def deadline(signum,frame):
        raise MeshDeadlineExceeded('original ObjectForesight mesh deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(a.seconds)
    try:
        pipeline=upstream.TrellisImageTo3DPipeline.from_pretrained(str(a.root/'models/trellis'))
        pipeline.cuda()
        flux=upstream.init_flux_upscaler(cfg) if cfg.flux_sr_enable else None
        report['flux_available']=flux is not None
        upstream.process_single_object(str(a.video),obj,pipeline,flux,cfg)
        mesh=obj/'trellis/model.glb'
        gaussian=obj/'trellis/gaussian.ply'
        if not mesh.exists() or not gaussian.exists():
            raise ValueError('original step8 did not emit textured GLB/Gaussian outputs')
        report.update(status='COMPLETED',mesh_sha256=sha(mesh),gaussian_sha256=sha(gaussian))
    except BaseException as e:
        report.update(status='TIMED_OUT' if isinstance(e,MeshDeadlineExceeded) else 'FAILED',error=repr(e))
        raise
    finally:
        signal.alarm(0)
        if a.diagnostics:
            faulthandler.cancel_dump_traceback_later()
        torch.hub.load=original_hub_load
        report['elapsed_s']=time.monotonic()-started
        report['simplification_choices']=simplification
        target.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
