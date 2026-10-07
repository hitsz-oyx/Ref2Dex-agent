"""Metric geometry helpers for the estimated-object engineering pipeline."""
import numpy as np
from scipy.spatial import ConvexHull, distance


def exact_mesh_diameter(model_pts=None, mesh=None, n_sample=None):
    """Exact Euclidean diameter using hull vertices and bounded distance blocks.

    Signature matches FoundationPose's helper. n_sample is unnecessary: the
    farthest pair belongs to the convex hull, so no random sample is needed.
    Projection is used only to identify hull indices for planar inputs; all
    final distances use original metric 3D points.
    """
    pts = np.asarray(mesh.vertices if mesh is not None else model_pts, dtype='float64')
    if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts) or not np.isfinite(pts).all():
        raise ValueError('finite Nx3 geometry required')
    centered = pts-pts.mean(0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    rank = int((singular > max(singular[0], 1.)*1e-12).sum())
    if rank == 0:
        return 0.
    if rank == 1:
        return float(np.ptp(centered @ axes[0]))
    hull = ConvexHull(centered @ axes[:rank].T)
    extreme = pts[hull.vertices]
    diameter = 0.
    for start in range(0, len(extreme), 512):
        diameter = max(diameter, float(distance.cdist(extreme[start:start+512], extreme).max()))
    return diameter
