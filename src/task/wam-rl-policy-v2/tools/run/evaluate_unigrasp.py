#!/usr/bin/env python3
"""Bounded, evaluation-only wrapper for the pinned UniGraspTransformer release.

Runs upstream run_online.py --test with strict checkpoint loading. The explicitly
selected upstream dynamic-reset option avoids the unavailable static-init archive;
this is a Preview4 compatibility probe, not reproduction of the paper protocol.
"""
import argparse
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[5]
PIN = '58753be89024267bb2348d0bf9a224fadbe5f35a'
FPS_URL = 'https://mirrors.pku.edu.cn/files/dl-release/UniDexGrasp_CVPR2023/dexgrasp_policy/assets/meshdatav3_pc_fps.zip'


def write(path, data):
    part = path.with_suffix(path.suffix + '.partial')
    part.write_text(json.dumps(data, indent=2) + '\n')
    part.replace(path)


class RemoteZip(io.RawIOBase):
    """Read selected official ZIP entries with bounded HTTP ranges."""
    def __init__(self, url, size=619907863):
        self.url, self.size, self.position, self.transferred = url, size, 0, 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0: raise ValueError('Negative ZIP seek')
        return self.position

    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size)
        if count <= 0: return b''
        if count > 16*2**20 or self.transferred+count > 32*2**20:
            raise ValueError('Remote ZIP metadata/entry budget exceeded')
        start, end = self.position, self.position+count-1
        request = urllib.request.Request(self.url+'?ref2dex_range=%d-%d' % (start, end),
            headers={'Range': 'bytes=%d-%d' % (start, end)})
        with urllib.request.urlopen(request, timeout=45) as response:
            expected = 'bytes %d-%d/%d' % (start, end, self.size)
            if response.status != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Server did not return requested ZIP byte range')
            data = response.read(count+1)
        if len(data) != count: raise ValueError('Incomplete ZIP range')
        self.position += count; self.transferred += count
        return data


def prepare(args):
    source = ROOT/'third_party/UniGraspTransformer'
    actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=source, text=True).strip()
    if actual != PIN: raise ValueError('Upstream source revision changed')
    out = args.output
    out.mkdir(parents=True); (out/'tmp').mkdir()
    upstream = out/'runtime/UniGraspTransformer'
    shutil.copytree(source, upstream, ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    assets = out/'runtime/Assets'; assets.mkdir()
    (out/'runtime/Logs').mkdir()
    base = ROOT/'outputs/wam-rl-policy-v2/official-scheduled-mixed-20261018-r4/assets-rank-0'
    for folder in ['mjcf','urdf','textures']:
        (assets/folder).symlink_to((source/'dexgrasp/hand_assets'/folder).resolve(), target_is_directory=True)
    for folder in ['meshdatav3_scaled','meshdatav3_pc_feat']:
        (assets/folder).symlink_to((base/folder).resolve(), target_is_directory=True)
    (assets/'datasetv4.1_posedata.npy').symlink_to((base/'datasetv4.1_posedata.npy').resolve())
    rows = (source/'results/configs/train_set_results.yaml').read_text().splitlines()
    candidates=[]
    for index,row in enumerate(rows):
        match=re.search(r"'([^']+)'\s*:\s*\[([\d.]+)\]",row)
        if match and match[1] == args.object_code and float(match[2]) == args.scale: candidates.append(index)
    if len(candidates) != 1: raise ValueError('Unique upstream object/scale row required')
    index = candidates[0]
    name = 'meshdatav3_pc_fps/'+args.object_code+'/coacd/pc_fps1024_%03d.npy' % round(args.scale*100)
    remote = RemoteZip(FPS_URL)
    with zipfile.ZipFile(remote) as archive:
        archive_name = name if name in archive.namelist() else name.replace('meshdatav3_pc_fps/', 'meshdatav3_pc_fps_new/', 1)
        info = archive.getinfo(archive_name)
        if info.file_size > 2**20: raise ValueError('Unexpected point-cloud payload')
        data = archive.read(archive_name)
    dest=assets/name;dest.parent.mkdir(parents=True);dest.write_bytes(data)
    patches=[]
    def replace(relative, old, new):
        path=upstream/relative;text=path.read_text()
        if text.count(old) != 1: raise ValueError('Compatibility patch anchor changed: '+relative)
        path.write_text(text.replace(old,new))
        patches.append(dict(path=relative,old=old,new=new))
    replace('dexgrasp/run_online.py','import numpy as np\n','import numpy as np\nnp.float = float\n')
    replace('dexgrasp/autoencoding/PN_Model.py','from maniskill_learn.networks.backbones.pointnet import',
        'from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import')
    replace('dexgrasp/cfg/train/universal_policy_state_based.yaml','  static_init: True','  static_init: False')
    if args.preserve_training_hand_pose:
        replace('dexgrasp/tasks/state_based_grasp.py',
            'shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_dims.z + 0.2)',
            'shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.08, self.table_dims.z + 0.2)')
    replace('dexgrasp/algo/pn_utils/maniskill_learn/utils/meta/__init__.py','from .magic_utils import *','')
    # Guard every optimizer step and collect passive evidence at upstream readout.
    hook='''    assert args.test and not args.train, 'Evaluation-only runner requires --test'
    import torch
    from pathlib import Path
    import json, time
    audit_out = Path(__file__).resolve().parents[3]
    def forbidden_step(*a, **kw):
        raise RuntimeError('Optimizer updates forbidden in evaluation-only probe')
    torch.optim.Adam.step = forbidden_step
    torch.optim.AdamW.step = forbidden_step
    from tasks.hand_base.vec_task import VecTaskPython
    original_reset, original_step = VecTaskPython.reset, VecTaskPython.step
    audit_start = time.monotonic()
    audit_task = None
    def observed_reset(self):
        global audit_task
        audit_task = self.task
        obs = original_reset(self)
        np.savez(audit_out/'initial-state.npz',observation=obs.cpu().numpy(),
            root=self.task.root_state_tensor.cpu().numpy())
        return obs
    def observed_step(self, actions, id=0):
        result = original_step(self, actions, id)
        if id == self.task.max_episode_length-2:
            scores = self.task.successes.detach().cpu().numpy()
            report = dict(episodes=self.num_envs,successes=int(scores.sum()),
                success_rate=float(scores.mean()),per_env_success=scores.tolist(),
                optimizer_steps=0,torch=torch.__version__,observation_dim=self.num_obs,
                elapsed_s=time.monotonic()-audit_start,
                reset_protocol='upstream Modes.static_init=False; official static-init archive unavailable',
                simulation='IsaacGym Preview4; released policy trained for Preview3')
            (audit_out/'evaluation.json').write_text(json.dumps(report,indent=2))
        return result
    VecTaskPython.reset, VecTaskPython.step = observed_reset, observed_step
'''
    replace('dexgrasp/run_online.py','    # start train or test process\n',hook+'\n    # start train or test process\n')
    replace('dexgrasp/run_online.py','        train()',
        '        try:\n            train()\n        finally:\n            if audit_task is not None:\n                audit_task.gym.destroy_sim(audit_task.sim)\n')
    model_folder=out/'runtime/Logs/Results/results_distill/random/universal_policy/state_based/distill_0000_3199_seed0'
    model_folder.mkdir(parents=True)
    (model_folder/'model_best.pt').symlink_to((source/'results/state_based/model_best_isaacgym_3.pt').resolve())
    command=[sys.executable,'-u','run_online.py','--task=StateBasedGrasp','--algo=dagger_value',
        '--seed='+str(args.seed),'--rl_device=cuda:0','--sim_device=cuda:0','--pipeline=gpu',
        '--num_envs='+str(args.envs),'--max_iterations=10000','--config=universal_policy_state_based.yaml',
        '--headless','--test','--test_iteration=1','--model_dir=distill_0000_3199',
        '--object_scale_file=train_set_results.yaml','--start_line='+str(index),'--end_line='+str(index+1)]
    manifest=dict(status='PREPARED',run_id=out.name,experiment_id='P-20261019-unigrasp-published-eval',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        upstream_commit=PIN,mode='evaluation-only',gpu=args.gpu,envs=args.envs,seed=args.seed,
        object_code=args.object_code,scale=args.scale,upstream_row=index,command=command,
        preserve_training_hand_pose=args.preserve_training_hand_pose,
        checkpoint=str(source/'results/state_based/model_best_isaacgym_3.pt'),
        seconds=args.seconds,artifact_limit_bytes=4*2**30,remote_zip_transferred=remote.transferred,
        compatibility_patches=patches,limitations=['Preview3 policy evaluated in Preview4',
            'Missing released static-init data; explicitly use upstream dynamic reset option'],
        missing_static_init_url='https://drive.google.com/file/d/14InPt0JFm6UNc-bdWCa4NkOVtATi-_M6/view',
        missing_static_init_issue='https://github.com/microsoft/UniGraspTransformer/issues/26')
    write(out/'manifest.json',manifest)
    return upstream,command,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--envs',type=int,default=256)
    parser.add_argument('--seed',type=int,default=246)
    parser.add_argument('--seconds',type=int,default=900)
    parser.add_argument('--object-code',default='core/jar-724fb5eb32401b96ed681532841d3af9')
    parser.add_argument('--scale',type=float,default=.06)
    parser.add_argument('--preserve-training-hand-pose',action='store_true',help='Preserve trained static-reset hand y=0.08 while using dynamic object resets')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume-prepared',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if not 1<=args.envs<=1000 or not 1<=args.seconds<=1800:parser.error('Bounded evaluation required')
    if ROOT/'outputs/wam-rl-policy-v2' not in args.output.parents:parser.error('Task-owned output required')
    if args.resume_prepared:
        manifest=json.loads((args.output/'manifest.json').read_text());command=manifest['command'];upstream=args.output/'runtime/UniGraspTransformer'
        if manifest['mode']!='evaluation-only' or manifest['gpu']!=args.gpu or manifest['envs']!=args.envs:raise ValueError('Prepared run arguments changed')
    else:
        if args.output.exists():raise FileExistsError(args.output)
        upstream,command,manifest=prepare(args)
    if args.prepare_only:return
    occupied=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:raise RuntimeError('Assigned GPU occupied')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MAX_JOBS='2',
        TMPDIR=str(args.output/'tmp'),XDG_CACHE_HOME=str(args.output/'tmp/cache'),
        TORCH_EXTENSIONS_DIR=str(ROOT/'outputs/wam-rl-policy/published-legacy-reset-audit-20261019-r1/torch_extensions'),
        PYTHONDONTWRITEBYTECODE='1',LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1]/'lib'),
        PYTHONPATH=os.pathsep.join([str(ROOT/'outputs/wam-rl-policy-v2/unigrasp-eval-20261019-r1/deps'),
            str(upstream/'pytorch_kinematics'),str(upstream/'dexgrasp'),str(upstream/'dexgrasp/algo/pn_utils')]))
    started=time.monotonic()
    with (args.output/'worker.log').open('a') as log:
        child=subprocess.Popen(command,cwd=upstream/'dexgrasp',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        manifest.update(status='RUNNING',pid=child.pid);write(args.output/'manifest.json',manifest)
        while child.poll() is None:
            elapsed=time.monotonic()-started
            total=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file() and not p.is_symlink())
            if elapsed>args.seconds or total>4*2**30 or shutil.disk_usage(args.output).free<20*2**30:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
                manifest['stop_reason']='budget';break
            with (args.output/'resources.jsonl').open('a') as f:
                usage=subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
                f.write(json.dumps(dict(elapsed_s=elapsed,gpu=usage))+'\n')
            time.sleep(10)
        code=child.wait()
    has_result=(args.output/'evaluation.json').is_file()
    manifest.update(status='COMPLETED' if code==0 and has_result else 'FAILED',exit_code=code,elapsed_s=time.monotonic()-started)
    if code==0 and not has_result:manifest['stop_reason']='upstream exited without evaluation result'
    write(args.output/'manifest.json',manifest)
    if code or not has_result:raise SystemExit(code or 1)


if __name__=='__main__': main()
