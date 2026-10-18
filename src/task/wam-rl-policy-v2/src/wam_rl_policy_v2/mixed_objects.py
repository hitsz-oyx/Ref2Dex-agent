"""TRAIN-only object partition and measured simulator identity binding."""
import json
from pathlib import Path
import numpy as np


def partition_objects(records,world_size,per_rank,seed=0):
    candidates={}
    for row in records:
        if row['split']=='train':
            candidates.setdefault(row['object_code'],dict(object_code=row['object_code'],scale=float(row['scale']),split='train',raw_episode=row.get('raw_episode')))
    codes=sorted(candidates)
    if len(codes)<world_size*per_rank:raise ValueError('Insufficient TRAIN objects')
    order=np.random.RandomState(seed).permutation(len(codes))[:world_size*per_rank]
    return [[candidates[codes[int(i)]] for i in order[r*per_rank:(r+1)*per_rank]] for r in range(world_size)]


def read_objects(path,envs):
    rows=json.loads(Path(path).read_text())['objects']
    if not rows or envs%len(rows) or len({r['object_code'] for r in rows})!=len(rows):
        raise ValueError('Distinct objects and equal environments per object required')
    if any(r['split']!='train' or not np.isfinite(r['scale']) or r['scale']<=0 for r in rows):
        raise ValueError('TRAIN-only positive-scale objects required')
    return rows


def bind_environment_objects(task,rows):
    declared={r['object_code']:float(r['scale']) for r in rows}
    codes=[task.object_code_list[int(i)] for i in task.object_id_buf.detach().cpu().tolist()]
    scales=[float(task.object_scale_buf[i]) for i in range(len(codes))]
    result=[]
    for code,scale in zip(codes,scales):
        if code not in declared or not np.isclose(scale,declared[code],atol=1e-6):
            raise ValueError('Simulator object identity differs from TRAIN manifest')
        result.append(dict(object_code=code,scale=declared[code]))
    if set(codes)!=set(declared):raise ValueError('Some declared objects have no environments')
    return result


def prepare_assets(rows,destination):
    """Reuse prepare_scale_panels.py's read-only link/selected-prior scheme."""
    assets=Path(destination);assets.mkdir(parents=True)
    sources={};priors={};closure=set()
    for row in rows:
        episode=Path(row['raw_episode']);batch=episode.parent.parent
        source_manifest=batch/'manifest.json'
        if source_manifest.exists():source=Path(json.loads(source_manifest.read_text())['asset_root'])
        else:source=Path(json.loads((episode.parent/'result.json').read_text())['asset_root'])
        code,scale=row['object_code'],row['scale']
        row['source_asset_root']=str(source)
        for name in ('mjcf','textures'):
            if not (assets/name).exists():(assets/name).symlink_to((source/name).resolve(),target_is_directory=True)
        if source not in sources:sources[source]=np.load(source/'datasetv4.1_posedata.npy',allow_pickle=True).item()
        priors[code]={scale:sources[source][code][scale]}
        for name in ('meshdatav3_scaled','meshdatav3_pc_feat'):
            target=(source/name/code).resolve()
            if not target.is_dir():raise FileNotFoundError(target)
            dest=assets/name/code;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(target,target_is_directory=True)
            closure.update(p.resolve() for p in target.rglob('*') if p.is_file())
        if not (assets/'meshdatav3_scaled'/code/'coacd'/('coacd_%03d.urdf'%round(scale*100))).exists():raise ValueError('Missing selected scale')
    np.save(assets/'datasetv4.1_posedata.npy',priors)
    size=sum(p.stat().st_size for p in closure)
    if size>512*2**20:raise ValueError('512MiB shard asset budget')
    return size
