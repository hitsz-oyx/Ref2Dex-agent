"""Bounded TRELLIS mesh reconstruction from an inspected SAM2 mask.

Uses the ObjectForesight vendored implementation. Only the mesh decoder is
loaded; Gaussian/texturing/rendering stages are unnecessary for geometry.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from native_data import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frame', type=int, default=0)
    p.add_argument('--steps', type=int, default=25)
    a = p.parse_args()
    if not 1 <= a.steps <= 25:
        p.error('steps must be 1..25')
    a.root = a.root.resolve(); a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2); cv2.setNumThreads(2)
    os.environ.setdefault('ATTN_BACKEND', 'flash_attn')
    os.environ.setdefault('SPARSE_ATTN_BACKEND', 'flash_attn')
    torch.hub.set_dir(str(a.root/'torch-hub'))
    sys.path.insert(0, str(a.root/'vendor/ObjectForesight-Data/trellis'))
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines.base import Pipeline
    from trellis.pipelines import samplers
    from torchvision import transforms
    import trimesh

    class LocalDinoPipeline(TrellisImageTo3DPipeline):
        @classmethod
        def from_pretrained(cls, path):
            # Upstream's static factory hard-codes its class, so explicitly
            # preserve its initialization while overriding only DINO source.
            base = Pipeline.from_pretrained(path)
            result = cls(); result.__dict__ = base.__dict__
            args = base._pretrained_args
            for stage in ('sparse_structure', 'slat'):
                spec = args[stage+'_sampler']
                setattr(result, stage+'_sampler', getattr(samplers,spec['name'])(**spec['args']))
                setattr(result, stage+'_sampler_params', spec['params'])
            result.slat_normalization = args['slat_normalization']
            result._init_image_cond_model(args['image_cond_model'])
            return result

        def _init_image_cond_model(self, name):
            model = torch.hub.load(str(a.root/'vendor/dinov2'), name,
                                   source='local', pretrained=True)
            self.models['image_cond_model'] = model.eval()
            self.image_cond_model_transform = transforms.Compose([
                transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])])

    report = dict(status='RUNNING', stage='TRELLIS_MESH', frame=a.frame,
                  metric_scale_available=False, training_allowed=False,
                  source_video_sha256=sha(a.video), mask_sha256=sha(a.masks),
                  script_sha256=sha(Path(__file__)))
    try:
        cap = cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
        ok, bgr = cap.read(); cap.release()
        if not ok:
            raise ValueError('reference frame decode failed')
        data = np.load(a.masks)
        mask = data['masks'][a.frame]
        mask = cv2.resize(mask.astype('uint8'), (bgr.shape[1],bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        if mask.sum() < 100:
            raise ValueError('reference mask too small')
        rgba = np.concatenate((cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),
                               (mask.astype('uint8')*255)[...,None]),axis=-1)
        image = Image.fromarray(rgba); image.save(a.output/'reference_rgba.png')
        # Keep original pretrained configuration; derive a local mesh-only view.
        config = json.loads((a.root/'models/trellis/pipeline.json').read_text())
        keep = {'sparse_structure_flow_model','sparse_structure_decoder',
                'slat_flow_model','slat_decoder_mesh'}
        names = config['args']['models']
        if not keep <= names.keys():
            raise ValueError(f'unexpected TRELLIS model names: {names.keys()}')
        config['args']['models'] = {k:v for k,v in names.items() if k in keep}
        local = a.output/'mesh_pipeline'; local.mkdir()
        (local/'ckpts').symlink_to(a.root/'models/trellis/ckpts',target_is_directory=True)
        (local/'pipeline.json').write_text(json.dumps(config,indent=2)+'\n')
        pipeline = LocalDinoPipeline.from_pretrained(str(local)); pipeline.cuda()
        with torch.inference_mode():
            out = pipeline.run(image, seed=42, formats=['mesh'],
                               sparse_structure_sampler_params={'steps':a.steps},
                               slat_sampler_params={'steps':a.steps})
        result = out['mesh'][0]
        if not result.success:
            raise ValueError('empty reconstructed mesh')
        vertices = result.vertices.detach().float().cpu().numpy()
        faces = result.faces.detach().cpu().numpy()
        colors = None
        if result.vertex_attrs is not None:
            colors = (result.vertex_attrs[:,:3].detach().float().cpu().numpy().clip(0,1)*255).astype('uint8')
        mesh = trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=colors,process=False)
        if not np.isfinite(vertices).all() or mesh.area <= 0:
            raise ValueError('invalid reconstructed mesh')
        mesh.export(a.output/'mesh.ply')
        report.update(status='COMPLETED',vertices=len(vertices),faces=len(faces),
                      watertight=mesh.is_watertight,bounds=mesh.bounds.tolist(),
                      mesh_sha256=sha(a.output/'mesh.ply'))
    except Exception as e:
        report.update(status='FAILED',error=repr(e)); raise
    finally:
        report['elapsed_s'] = time.monotonic()-start
        (a.output/'mesh_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
