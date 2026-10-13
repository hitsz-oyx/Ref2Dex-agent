"""Optical Z lifting, native camera compensation and a fixed current anchor."""
import numpy as np

ARKIT_TO_ZUP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], np.float32)


def check_transform(matrix):
    matrix = np.asarray(matrix, np.float32)
    if matrix.shape[-2:] != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("finite SE(3) transform required")
    r = matrix[..., :3, :3]
    if (not np.allclose(matrix[..., 3, :], [0, 0, 0, 1], atol=1e-4)
            or not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(r), 1., atol=1e-3)):
        raise ValueError("proper rigid transform required")
    return matrix


def transform_points(points, matrix):
    matrix = check_transform(matrix)
    if matrix.ndim != 2:
        raise ValueError("one anchor transform required")
    return (np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def resize_intrinsic(intrinsic, original_wh, new_wh):
    k = np.asarray(intrinsic, np.float32).copy()
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(*original_wh, *new_wh) <= 0:
        raise ValueError("finite intrinsic and positive image sizes required")
    # OpenCV resize uses pixel centers, so update principal points consistently.
    sx, sy = np.asarray(new_wh) / np.asarray(original_wh)
    k[0, :2] *= sx
    k[1, :2] *= sy
    k[0, 2] = (k[0, 2] + .5) * sx - .5
    k[1, 2] = (k[1, 2] + .5) * sy - .5
    return k


def sample_depth(depth, uv):
    """Bilinear sampling requires four finite positive neighboring pixels."""
    depth, uv = np.asarray(depth, np.float32), np.asarray(uv, np.float32)
    if depth.ndim != 3 or uv.ndim != 3 or uv.shape[0] != len(depth) or uv.shape[2] != 2:
        raise ValueError("depth TxHxW and UV TxQx2 required")
    height, width = depth.shape[1:]
    if min(height, width) < 2:
        raise ValueError("depth map too small")
    good = np.isfinite(uv).all(-1)
    good &= (uv[..., 0] >= 0) & (uv[..., 0] < width - 1)
    good &= (uv[..., 1] >= 0) & (uv[..., 1] < height - 1)
    safe = np.where(good[..., None], uv, 0.)
    x, y = np.floor(safe[..., 0]).astype(int), np.floor(safe[..., 1]).astype(int)
    t = np.arange(len(depth))[:, None]
    values = np.stack([depth[t, y, x], depth[t, y, x+1],
                       depth[t, y+1, x], depth[t, y+1, x+1]], -1)
    good &= np.isfinite(values).all(-1) & (values > 0).all(-1)
    dx, dy = safe[..., 0]-x, safe[..., 1]-y
    weights = np.stack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy], -1)
    result = (np.nan_to_num(values, nan=0., posinf=0., neginf=0.) * weights).sum(-1)
    return np.where(good, result, 0.).astype(np.float32), good


def lift_tracks(uv, visibility, depth_m, intrinsic, cameras, depth_to_native=None):
    cameras = check_transform(cameras)
    depth_to_native = check_transform(np.eye(4) if depth_to_native is None else depth_to_native)
    z, valid = sample_depth(depth_m, uv)
    if cameras.shape != (len(z), 4, 4) or np.shape(visibility) != z.shape:
        raise ValueError("track/camera clocks differ")
    rays = np.concatenate([np.nan_to_num(uv), np.ones((*z.shape, 1))], -1)
    rays = rays @ np.linalg.inv(intrinsic).T
    native = transform_points(rays * z[..., None], depth_to_native)
    world = np.einsum("tij,tqj->tqi", cameras[:, :3, :3], native) + cameras[:, None, :3, 3]
    valid &= np.asarray(visibility, bool) & np.isfinite(world).all(-1)
    return np.where(valid[..., None], world, 0.).astype(np.float32), valid


def current_anchor(current_world):
    points = np.asarray(current_world, np.float32)
    if points.ndim != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("valid current target points required for anchor")
    anchor = np.eye(4, dtype=np.float32)
    anchor[:3, :3] = ARKIT_TO_ZUP
    anchor[:3, 3] = -ARKIT_TO_ZUP @ points.mean(0)
    return anchor


def calibrate_scale(raw_depth, hand_world, hand_valid, intrinsic, cameras,
                    depth_to_native=None, min_samples=12, max_relative_iqr=.25):
    """Approximate joint/surface median scale, using ONLY supplied frames."""
    cameras = check_transform(cameras)
    bridge = check_transform(np.eye(4) if depth_to_native is None else depth_to_native)
    inverse = np.linalg.inv(cameras @ bridge)
    camera_hand = np.einsum("tij,t...j->t...i", inverse[:, :3, :3], hand_world)
    camera_hand += inverse[:, None, None, :3, 3]
    projected = camera_hand @ np.asarray(intrinsic).T
    z = camera_hand[..., 2]
    uv = projected[..., :2] / np.where(z[..., None] > 0, z[..., None], np.nan)
    measured, good = sample_depth(raw_depth, uv.reshape(len(uv), -1, 2))
    good = good.reshape(z.shape) & np.asarray(hand_valid, bool) & (z > 0)
    ratios = z[good] / measured.reshape(z.shape)[good]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) < min_samples:
        raise ValueError("UNCALIBRATED: insufficient observed hand/depth samples")
    scale = float(np.median(ratios))
    spread = float(np.subtract(*np.quantile(ratios, [.75, .25])) / scale)
    if spread > max_relative_iqr:
        raise ValueError("UNCALIBRATED: hand/depth scale inconsistent")
    return dict(scale=scale, samples=len(ratios), relative_iqr=spread,
                method="approximate-native-hand-median-scale")
