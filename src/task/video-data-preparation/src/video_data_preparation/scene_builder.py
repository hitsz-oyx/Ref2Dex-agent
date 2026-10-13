"""Current-only sampling; sparse queries are selected before tracking."""
import numpy as np
from .geometry import lift_tracks, transform_points


def farthest_indices(points, count):
    if not len(points) or count < 1:
        return np.empty(0, np.int64)
    distance = np.full(len(points), np.inf)
    selected = []
    current = int(np.argmax(np.linalg.norm(points-points.mean(0), axis=-1)))
    for _ in range(min(count, len(points))):
        selected.append(current)
        distance = np.minimum(distance, np.sum((points-points[current])**2, -1))
        distance[selected] = -1
        current = int(np.argmax(distance))
    return np.asarray(selected, np.int64)


def select_queries(depth, target_mask, hand_mask, intrinsic, camera, count=64,
                   erosion=2, depth_to_native=None):
    import cv2
    mask = np.asarray(target_mask, bool) & ~np.asarray(hand_mask, bool)
    if mask.shape != depth.shape or count < 32 or count > 128:
        raise ValueError("current masks and 32..128 queries required")
    if erosion:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((erosion*2+1,)*2, np.uint8)).astype(bool)
    y, x = np.nonzero(mask)
    # Bound current candidate selection cost independently of image resolution.
    ids = np.linspace(0, len(x)-1, min(len(x), 4096), dtype=np.int64)
    uv = np.column_stack((x[ids], y[ids])).astype(np.float32)
    world, valid = lift_tracks(uv[None], np.ones((1, len(uv)), bool), depth[None],
                               intrinsic, camera[None], depth_to_native)
    uv, world = uv[valid[0]], world[0, valid[0]]
    keep = farthest_indices(world, count)
    return uv[keep], world[keep]


def build_scene(depth, target_mask, hand_mask, intrinsic, camera, anchor,
                query_uv, query_world, hand_world, hand_valid, count=2048,
                voxel=.01, local_radius=.30, depth_to_native=None):
    """Include query points, then target/local/context quotas without hand voxels."""
    import cv2
    if not 32 <= count <= 4096 or voxel <= 0:
        raise ValueError("bounded scene count and positive voxel size required")
    height, width = depth.shape
    # Compute normals from neighboring pixels before sparse sampling.
    yy, xx = np.mgrid[:height, :width]
    uv = np.stack([xx, yy], -1).astype(np.float32).reshape(-1, 2)
    world, valid = lift_tracks(uv[None], np.ones((1, len(uv)), bool), depth[None],
                               intrinsic, camera[None], depth_to_native)
    cloud = world[0].reshape(height, width, 3)
    known = valid[0].reshape(height, width) & ~np.asarray(hand_mask, bool)
    dx, dy = np.zeros_like(cloud), np.zeros_like(cloud)
    dx[:, 1:-1] = cloud[:, 2:]-cloud[:, :-2]
    dy[1:-1] = cloud[2:]-cloud[:-2]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=-1)
    continuity = np.zeros_like(known)
    continuity[1:-1, 1:-1] = (known[1:-1, 2:] & known[1:-1, :-2]
                             & known[2:, 1:-1] & known[:-2, 1:-1])
    # Avoid normals bridging depth discontinuities.
    continuity &= (np.linalg.norm(dx, axis=-1) < .05) & (np.linalg.norm(dy, axis=-1) < .05)
    normal = np.where((continuity & (length > 1e-8))[..., None],
                      normal / np.maximum(length[..., None], 1e-8), 0.)
    direction = cloud-camera[:3, 3]
    normal *= np.where((normal*direction).sum(-1) > 0, -1., 1.)[..., None]
    positions = np.flatnonzero(known.reshape(-1))
    # Dense data never enters the model: one representative per current voxel.
    coordinates = transform_points(world[0, positions], anchor)
    _, first = np.unique(np.floor(coordinates/voxel).astype(np.int64), axis=0, return_index=True)
    positions, coordinates = positions[first], coordinates[first]
    workspace = (coordinates >= -2.).all(-1) & (coordinates < 2.).all(-1)
    positions, coordinates = positions[workspace], coordinates[workspace]
    target = np.asarray(target_mask, bool).reshape(-1)[positions]
    center = transform_points(query_world, anchor).mean(0)
    nearby = np.linalg.norm(coordinates-center, axis=-1) <= local_radius
    hands = transform_points(hand_world[hand_valid], anchor)
    if len(hands):
        nearby |= np.linalg.norm(coordinates[:, None]-hands[None], axis=-1).min(1) <= local_radius
    query_pixels = np.rint(query_uv).astype(int)
    query_flat = query_pixels[:, 1]*width+query_pixels[:, 0]
    used = np.isin(positions, query_flat)
    pools = [np.flatnonzero(target & ~used), np.flatnonzero(~target & nearby & ~used),
             np.flatnonzero(~target & ~nearby & ~used)]
    remaining = count-len(query_world)
    quotas = [max(0, count//4-len(query_world)), count//2, count//4]
    chosen = []
    for pool, quota in zip(pools, quotas):
        n = min(len(pool), quota, remaining)
        chosen.extend(pool[np.linspace(0, len(pool)-1, n, dtype=int)].tolist() if n else [])
        remaining -= n
    extras = np.setdiff1d(np.flatnonzero(~used), chosen)
    n = min(remaining, len(extras))
    if n:
        chosen.extend(extras[np.linspace(0, len(extras)-1, n, dtype=int)].tolist())
    chosen = np.asarray(chosen, int)
    query_xyz = transform_points(query_world, anchor)
    xyz = np.concatenate([query_xyz, coordinates[chosen]])
    features = np.zeros((len(xyz), 18), np.float32)
    normals = np.concatenate([normal.reshape(-1, 3)[query_flat],
                              normal.reshape(-1, 3)[positions[chosen]]]) @ anchor[:3, :3].T
    features[:, :3] = normals
    is_target = np.r_[np.ones(len(query_world), bool), target[chosen]]
    features[:, 12] = is_target
    return dict(xyz=xyz.astype(np.float32), features=features,
                scene_object=np.where(is_target, 0, -2).astype(np.int64))
