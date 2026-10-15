"""TRAIN-only sufficient statistics, independent of a future source recipe."""
import numpy as np


class Moments:
    def __init__(self):
        self.count = np.zeros(24, np.int64)
        self.mean = np.zeros((24, 3), np.float64)
        self.m2 = np.zeros((24, 3), np.float64)

    def update(self, values, valid):
        values = np.asarray(values, np.float64).reshape(24, -1, 3)
        valid = np.asarray(valid, bool).reshape(24, -1)
        if not np.isfinite(values[valid]).all():
            raise ValueError('nonfinite valid displacement in moments')
        for h in range(24):
            x = values[h, valid[h]]
            if not len(x):
                continue
            n, mean = len(x), x.mean(0)
            total = self.count[h] + n
            delta = mean-self.mean[h]
            self.m2[h] += ((x-mean)**2).sum(0) + delta**2*self.count[h]*n/total
            self.mean[h] += delta*n/total
            self.count[h] = total

    def combine(self, other):
        n = np.asarray(other['count'], np.int64)
        mean, m2 = np.asarray(other['mean']), np.asarray(other['m2'])
        total = self.count+n
        delta = mean-self.mean
        fraction = np.divide(n,total,out=np.zeros(24),where=total>0)
        self.m2 += m2 + delta**2*(self.count*fraction)[:,None]
        self.mean += delta*fraction[:,None]
        self.count = total

    def result(self):
        return dict(count=self.count.tolist(),mean=self.mean.tolist(),m2=self.m2.tolist())


def sequence_moments(sequence, ticks, geometries, target_slot, workspace=None, chunk_size=64):
    """Exact filtered ticks; labels match pose_mesh_sample without building features."""
    ticks = np.asarray(ticks, np.int64)
    if ticks.ndim != 1 or not len(ticks) or not 1 <= chunk_size <= 256:
        raise ValueError('nonempty window ticks and bounded chunks required')
    hand, scene = Moments(), Moments()
    incompatible = []
    poses = sequence['object_poses'].astype(np.float64)
    points = sequence['hand'].astype(np.float64)
    known, hand_valid = sequence['object_pose_valid'], sequence['hand_valid']
    minimum, maximum = np.full(3,np.inf), np.full(3,-np.inf)
    for start in range(0,len(ticks),chunk_size):
        current = ticks[start:start+chunk_size]
        if (current<3).any() or (current+24>=len(poses)).any():
            raise ValueError('statistics window outside H4/K24')
        future = current[:,None]+np.arange(1,25)
        history = current[:,None]+np.arange(-3,1)
        anchor = np.linalg.inv(poses[current,target_slot])
        rotation = anchor[:,:3,:3]
        present = hand_valid[history].all(1)
        valid = hand_valid[future] & present[:,None]
        delta = np.einsum('bij,bthpj->bthpi',rotation,points[future]-points[current,None])
        hand.update(np.moveaxis(delta.astype(np.float32),1,0),np.moveaxis(valid,1,0))
        outside = np.zeros(len(current),bool)

        def observe(cloud, mask):
            nonlocal minimum,maximum,outside
            valid_points = cloud[mask]
            if len(valid_points):
                if not np.isfinite(valid_points).all():
                    raise ValueError('nonfinite observed input')
                minimum = np.minimum(minimum,valid_points.min(0))
                maximum = np.maximum(maximum,valid_points.max(0))
            if workspace is not None:
                grid = np.floor((cloud.astype(np.float32)-np.asarray(workspace['origin_m'],np.float32))/workspace['voxel_m'])
                bad = ((grid<0)|(grid>workspace['max_grid'])).any(-1) & mask
                outside |= bad.reshape(len(current),-1).any(1)

        observed_hand = np.einsum('bij,bhpj->bhpi',rotation,points[current])+anchor[:,None,None,:3,3]
        observe(observed_hand,present)
        for slot,geometry in enumerate(geometries):
            active = known[history,slot].all(1)
            valid = known[future,slot] & active[:,None]
            query = geometry['query'].astype(np.float64)
            change = poses[future,slot,:3,:3]-poses[current,None,slot,:3,:3]
            world = np.einsum('btij,pj->btpi',change,query)+(poses[future,slot,:3,3]-poses[current,None,slot,:3,3])[:,:,None]
            delta = np.einsum('bij,btpj->btpi',rotation,world)
            mask = np.broadcast_to(valid[:,:,None],delta.shape[:-1])
            scene.update(np.moveaxis(delta.astype(np.float32),1,0),np.moveaxis(mask,1,0))
            relative = anchor @ np.where(active[:,None,None],poses[current,slot],np.eye(4))
            cloud = np.einsum('bij,pj->bpi',relative[:,:3,:3],geometry['surface'])+relative[:,None,:3,3]
            observe(cloud,np.broadcast_to(active[:,None],cloud.shape[:-1]))
        incompatible.extend(current[outside].tolist())
    return dict(windows=len(ticks),hand=hand.result(),scene=scene.result(),
        observed_min_m=minimum.tolist(),observed_max_m=maximum.tolist(),
        workspace_incompatible_ticks=incompatible)


def mix_moments(sources, weights, floor_m):
    """Expected masked point mean under source -> uniform-window sampling.

    Per-source counts/sums are divided by its number of windows before weighting.
    This matches a global masked mean; it does not claim equal per-window loss.
    """
    if (not weights or set(weights)-set(sources) or not np.isfinite(floor_m) or floor_m<=0
            or any(not np.isfinite(v) or v<=0 for v in weights.values())):
        raise ValueError('known sources, positive weights and std floor required')
    total_weight = sum(weights.values())
    result = {}
    for name in ('hand','scene'):
        count,sums,squares = np.zeros(24),np.zeros((24,3)),np.zeros((24,3))
        for sid,weight in weights.items():
            record = sources[sid]
            scale = weight/total_weight/record['windows']
            m = record[name]
            c,mean,m2 = np.asarray(m['count']),np.asarray(m['mean']),np.asarray(m['m2'])
            count += scale*c
            sums += scale*c[:,None]*mean
            squares += scale*(m2+c[:,None]*mean**2)
        if (count<=0).any():
            raise ValueError('every horizon requires valid TRAIN observations')
        mean = sums/count[:,None]
        std = np.sqrt(np.maximum(squares/count[:,None]-mean**2,0))
        result[name] = dict(mean=mean.tolist(),raw_std=std.tolist(),
            std=np.maximum(std,floor_m).tolist(),expected_valid_points_per_window=count.tolist())
    return result


def subtract_moments(total, removed):
    """Remove an explicitly identified subset, preserving population moments."""
    n,b = np.asarray(total['count'],np.int64),np.asarray(removed['count'],np.int64)
    remaining=n-b
    if (b<0).any() or (remaining<=0).any():
        raise ValueError('subtraction requires a proper observed subset at every horizon')
    mean=(n[:,None]*np.asarray(total['mean'])-b[:,None]*np.asarray(removed['mean']))/remaining[:,None]
    delta=np.asarray(removed['mean'])-mean
    m2=np.asarray(total['m2'])-np.asarray(removed['m2'])-delta**2*(remaining*b/n)[:,None]
    if (m2 < -1e-8*np.maximum(np.asarray(total['m2']),1)).any():
        raise ValueError('removed moments are not a valid subset')
    return dict(count=remaining.tolist(),mean=mean.tolist(),m2=np.maximum(m2,0).tolist())


def observed_workspace(sequence,ticks,geometries,target_slot,workspace,chunk_size=128):
    """Current input only: H4 determines presence; no future labels or outcomes."""
    ticks=np.asarray(ticks,np.int64)
    accepted=[]
    origin=np.asarray(workspace['origin_m'],np.float32)
    for start in range(0,len(ticks),chunk_size):
        current=ticks[start:start+chunk_size]
        history=current[:,None]+np.arange(-3,1)
        anchor=np.linalg.inv(sequence['object_poses'][current,target_slot])
        present=sequence['hand_valid'][history].all(1)
        hand=np.einsum('bij,bhpj->bhpi',anchor[:,:3,:3],sequence['hand'][current])+anchor[:,None,None,:3,3]
        outside=np.zeros(len(current),bool)
        def check(cloud,valid):
            nonlocal outside
            grid=np.floor((cloud.astype(np.float32)-origin)/workspace['voxel_m'])
            bad=((~np.isfinite(cloud).all(-1))|(grid<0).any(-1)|(grid>workspace['max_grid']).any(-1)) & valid
            outside |= bad.reshape(len(current),-1).any(1)
        check(hand,present)
        for slot,geometry in enumerate(geometries):
            active=sequence['object_pose_valid'][history,slot].all(1)
            relative=anchor@np.where(active[:,None,None],sequence['object_poses'][current,slot],np.eye(4))
            cloud=np.einsum('bij,pj->bpi',relative[:,:3,:3],geometry['surface'])+relative[:,None,:3,3]
            check(cloud,np.broadcast_to(active[:,None],cloud.shape[:-1]))
        accepted.extend((~outside).tolist())
    return np.asarray(accepted,bool)
