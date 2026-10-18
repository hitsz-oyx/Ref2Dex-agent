"""TRAIN-only object partition and measured simulator identity binding."""
import json
from pathlib import Path
import numpy as np


def partition_objects(records,world_size,per_rank,seed=0):
    candidates={}
    for row in records:
        if row['split']=='train':
            candidates.setdefault(row['object_code'],dict(object_code=row['object_code'],scale=float(row['scale']),split='train'))
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
    scales=task.object_scale_buf.detach().cpu().tolist()
    result=[]
    for code,scale in zip(codes,scales):
        if code not in declared or not np.isclose(scale,declared[code],atol=1e-6):
            raise ValueError('Simulator object identity differs from TRAIN manifest')
        result.append(dict(object_code=code,scale=declared[code]))
    if set(codes)!=set(declared):raise ValueError('Some declared objects have no environments')
    return result
