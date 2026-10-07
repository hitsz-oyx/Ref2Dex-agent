"""Execute existing original-core adapters serially for one curated EgoDex clip.

This is command orchestration only. Source inputs are hashed, completed stages
can resume, failed stage directories are preserved, and every GPU stage requires
an empty GPU. Source runtime_env.sh before invoking with the isolated interpreter.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stage_status(returncode,state):
    status=state.get('status','MISSING_RESULT')
    if returncode != 0:
        return status if status in ['FAILED','TIMED_OUT'] else 'PROCESS_FAILED'
    if status in ['COMPLETED','STAGED','ENGINEERING_PASS']:
        return 'COMPLETED'
    return 'INCOMPLETE_RESULT' if status in ['RUNNING','PREPARED'] else status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=3)
    p.add_argument('--seconds', type=int, default=1800)
    p.add_argument('--mesh-diagnostics',action='store_true')
    p.add_argument('--mesh-face-budget',type=int)
    a = p.parse_args()
    if not 1 <= a.seconds <= 1800 or os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.gpu):
        p.error('bounded deadline and matching CUDA_VISIBLE_DEVICES required')
    root = Path(os.environ['REF5_OUTPUT_ROOT']).resolve()
    out = a.output.resolve()
    if not out.is_relative_to(root):
        p.error('output must stay in the owned output root')
    inputs = json.loads(a.inputs.read_text())
    if inputs['training_allowed'] is not False:
        raise ValueError('official test input cannot be enabled for training')
    source = inputs['source']
    for name in ['video', 'hdf5']:
        if digest(source[name]) != source['source_sha256'][name]:
            raise ValueError('curated source changed: '+name)
    out.mkdir(parents=True, exist_ok=True)
    target = out/'clip_run_manifest.json'
    report = (json.loads(target.read_text()) if target.exists() else
              dict(status='PREPARED', training_allowed=False,
                   inputs_sha256=digest(a.inputs), source_key=source['key'], stages=[]))
    if report['inputs_sha256'] != digest(a.inputs):
        raise ValueError('resume inputs changed; use a fresh run directory')
    report.update(pid=os.getpid(), seconds_budget=a.seconds, script_sha256=digest(__file__))
    tools = Path(__file__).resolve().parent
    started = time.monotonic()
    def save():
        report['elapsed_this_invocation_s'] = time.monotonic()-started
        target.write_text(json.dumps(report, indent=2)+'\n')
    def stage(name, tool, args, result, gpu=True):
        result = out/result
        previous = next((x for x in report['stages'] if x['name']==name), None)
        if result.exists():
            state = json.loads(result.read_text())
            if previous is None or previous.get('status') != 'COMPLETED':
                raise ValueError('unattributed/failed existing stage; preserve it and use a fresh output')
            if state['status'] not in ['COMPLETED', 'STAGED', 'ENGINEERING_PASS']:
                raise ValueError('stage is terminal without accepted result: '+name)
            return state
        if gpu:
            query = subprocess.run(['nvidia-smi','-i',str(a.gpu),
                                    '--query-compute-apps=pid','--format=csv,noheader'],
                                   check=True, text=True, capture_output=True).stdout.strip()
            if query:
                report.update(status='RESOURCE_WAIT', stage=name, occupied_gpu_pids=query.splitlines())
                save()
                raise SystemExit(77)
        if shutil.disk_usage(root).free < 20*2**30:
            raise RuntimeError('free disk below20GiB stop condition')
        remaining = a.seconds-(time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('clip deadline')
        command = [sys.executable,str(tools/tool),*[str(x) for x in args]]
        record = dict(name=name,status='RUNNING',command=command,
                      script_sha256=digest(tools/tool),log=str(out/(name+'.log')))
        report['stages'].append(record)
        report.update(status='RUNNING',stage=name)
        report.pop('occupied_gpu_pids',None)
        env = os.environ.copy()
        if name in ['track','export']:
            env['LD_PRELOAD']='/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0'
            env['PYOPENGL_PLATFORM']='egl'
            env['EGL_DEVICE_ID']=str(a.gpu)
        tick = time.monotonic()
        with Path(record['log']).open('x') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            record['pid']=process.pid
            save()
            try:
                code=process.wait(timeout=remaining)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
                raise
        record.update(returncode=code,elapsed_s=time.monotonic()-tick)
        state = json.loads(result.read_text()) if result.exists() else dict(status='MISSING_RESULT')
        record['status']=stage_status(code,state)
        record['result_sha256']=digest(result) if result.exists() else None
        save()
        print(json.dumps(record),flush=True)
        if record['status'] != 'COMPLETED':
            report['status']=record['status'];save();raise SystemExit(2)
        return state
    video,hdf5 = source['video'],source['hdf5']
    try:
        stage('masks','track_masks.py',['--video',video,'--model',root/'models/sam2-small',
              '--output',out/'masks','--point',*inputs['point'],'--box',*inputs['box'],
              '--prompt-frame',inputs['prompt_frame']], 'masks/mask_manifest.json')
        stage('mesh','upstream_mesh.py',['--root',root,'--video',video,
              '--masks',out/'masks/masks.npz','--output',out/'mesh',
              '--clean-start',inputs['mesh_clean_frames'][0],'--clean-end',inputs['mesh_clean_frames'][1],
              '--mask-start',inputs['mesh_selected_frames'][0],'--mask-end',inputs['mesh_selected_frames'][1],
              *(['--diagnostics'] if a.mesh_diagnostics else []),
              *(['--mesh-face-budget',a.mesh_face_budget] if a.mesh_face_budget else [])],
              'mesh/mesh_manifest.json')
        stage('depth','estimate_depth.py',['--root',root,'--video',video,'--hdf5',hdf5,
              '--output',out/'depth','--start-frame',inputs['depth_start'],'--frames',inputs['depth_frames']],
              'depth/depth_manifest.json')
        stage('refine','refine_depth.py',['--root',root,'--video',video,
              '--depth',out/'depth/depth.npz','--output',out/'refined'],'refined/refinement_manifest.json')
        stage('stage','upstream_object.py',['stage','--root',root,'--video',video,'--hdf5',hdf5,
              '--mesh',out/'mesh/object_0/trellis/model.glb','--masks',out/'masks/masks.npz',
              '--depth',out/'refined/depth.npz','--output',out/'pose',
              '--clean-frames',','.join(str(x) for x in inputs['pose_clean_frames']),
              '--height',inputs['pose_image_size'][0],'--width',inputs['pose_image_size'][1]],
              'pose/adapter_manifest.json',gpu=False)
        stage('track','upstream_object.py',['run','--root',root,'--output',out/'pose',
              '--egl-device',a.gpu],'pose/upstream_run_manifest.json')
        stage('export','export_upstream_object.py',['--root',root,'--input',out/'pose',
              '--output',out/'native','--hdf5',hdf5,'--depth',out/'refined/depth.npz',
              '--masks',out/'masks/masks.npz','--sequence','egodex_test_'+source['key'].replace('-','_'),
              '--object-key','egodex_'+source['key'].replace('-','_')],'native/pose_manifest.json')
        audit=stage('audit','audit_object.py',['--input',out/'native','--hdf5',hdf5,
              '--depth',out/'refined/depth.npz'],'native/object_audit.json',gpu=False)
        report['status']=audit['verdict'];save()
    except Exception as error:
        report.update(status='FAILED',error=repr(error));save();raise
    finally:
        save()


if __name__ == '__main__':
    main()
