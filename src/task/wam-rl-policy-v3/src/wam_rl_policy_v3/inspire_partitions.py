"""Disjoint full-TRAIN object/motion partitions for one shared Inspire PPO."""
import random


def partition_inspire(records,world_size,seed):
    train=[r for r in records if r['split']=='train']
    names=sorted({r['object'] for r in train})
    if not 1<=world_size<=6 or len(names)<world_size:raise ValueError('one to six nonempty TRAIN partitions required')
    if len({r['sequence'] for r in train})!=len(train):raise ValueError('duplicate TRAIN sequence')
    random.Random(seed).shuffle(names)
    partitions=[]
    for rank in range(world_size):
        selected=names[rank::world_size]
        partitions.append(dict(rank=rank,object_names=selected,
            sequence_names=[r['sequence'] for r in train if r['object'] in selected]))
    if sum(len(p['sequence_names']) for p in partitions)!=len(train):raise ValueError('incomplete TRAIN partition')
    return partitions
